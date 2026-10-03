"""Experimental SSD streaming adapter for GLM EXL3 checkpoints.

Uses ExLlamaV3's verified architecture and CUDA kernels, while loading linear
weights on demand. Routed experts are never loaded as a complete layer. A
bounded static cache admits non-expert matrices; streaming matrices are released
after each invocation. This avoids an LRU thrash over a model larger than VRAM.
Architecture, quantization and compute kernels are provided by ExLlamaV3.
"""

from __future__ import annotations

import contextlib
import os
import math
from pathlib import Path
import time

# CUDA graphs retain weight pointers and are incompatible with eviction.
for _key in ("EXL3_BC_MLA", "EXL3_BC_ATTN", "EXL3_BC_GDN", "EXL3_LOAD_ARENA"):
    os.environ[_key] = "0"

import torch
from exllamav3 import Cache, Config, Model, Tokenizer
from exllamav3.modules.linear import Linear
from exllamav3.modules.quant.exl3 import LinearEXL3
from .validation import validate_generation, final_answer


def tensor_bytes(obj):
    """Count unique CUDA storage owned by a native Linear implementation."""
    seen = set()
    size = 0
    for value in vars(obj).values():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            storage = value.untyped_storage()
            ptr = storage.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                size += storage.nbytes()
    return size


class StreamInner:
    bc = None
    bias = None

    def __init__(self, owner, manager):
        self.owner = owner
        self.manager = manager

    def forward(self, x, params, out_dtype=None):
        return self.manager.forward(self.owner, x, params, out_dtype)


class WeightManager:
    def __init__(self, cache_bytes, host_cache_bytes=0):
        if cache_bytes < 0 or host_cache_bytes < 0:
            raise ValueError("Cache budgets must be nonnegative")
        self.limit = cache_bytes
        self.host_limit = host_cache_bytes
        self.host_cached = {}
        self.host_bytes = 0
        self.host_hits = 0
        self.cached = {}
        self.bytes = 0
        self.loads = 0
        self.hits = 0
        self.loaded_bytes = 0
        self.load_seconds = 0.0
        self.original_load = Linear.load
        self.original_unload = Linear.unload

    def install_linear(self, linear, device, **kwargs):
        # Routers are small, persistent and their native weight is used directly
        # by ExLlama's bias-aware routing kernel.
        if linear.key.endswith(".gate"):
            return self.original_load(linear, device, **kwargs)
        linear.device = torch.device(device)
        linear.quant_type = "ssd_stream"
        linear.inner = StreamInner(linear, self)

    def forward(self, linear, x, params, out_dtype):
        key = id(linear)
        inner = self.cached.get(key)
        if inner is not None:
            self.hits += 1
            return inner.forward(x, params, out_dtype)
        proxy, kind = linear.inner, linear.quant_type
        t0 = time.perf_counter()
        try:
            host = self.host_cached.get(key)
            if host is None:
                self.original_load(linear, linear.device)
                inner = linear.inner
            else:
                self.host_hits += 1
                inner = LinearEXL3(
                    linear.config,
                    linear.in_features,
                    linear.out_features,
                    out_dtype=linear.out_dtype,
                    key=linear.key,
                    **{
                        name: value.to(linear.device, non_blocking=True)
                        if name not in ("mcg", "mul1")
                        else value
                        for name, value in host.items()
                    },
                )
            size = tensor_bytes(inner)
            self.loads += 1
            self.loaded_bytes += size
            self.load_seconds += time.perf_counter() - t0
            # Never admit routed expert weights: one pass must not evict the
            # permanent trunk, and the cache bound includes all native tensors.
            keep = ".experts." not in linear.key and self.bytes + size <= self.limit
            result = inner.forward(x, params, out_dtype)
            if keep:
                self.cached[key] = inner
                self.bytes += size
            else:
                if (
                    host is None
                    and isinstance(inner, LinearEXL3)
                    and ".experts." not in linear.key
                    and self.host_bytes + size + 16 <= self.host_limit
                ):
                    bundle = {}
                    for name in ("suh", "svh", "trellis", "bias", "mcg", "mul1"):
                        value = getattr(
                            inner, name + "_tensor" if name in ("mcg", "mul1") else name
                        )
                        if value is None:
                            continue
                        if value.is_cuda:
                            pinned = torch.empty_like(value, device="cpu", pin_memory=True)
                            pinned.copy_(value)
                            bundle[name] = pinned
                        else:
                            bundle[name] = value
                    cpu_size = sum(value.untyped_storage().nbytes() for value in bundle.values())
                    self.host_cached[key] = bundle
                    self.host_bytes += cpu_size
                # Complete kernels before dropping their backing allocations.
                torch.cuda.synchronize(linear.device)
                inner.unload()
            return result
        finally:
            linear.inner = proxy
            linear.quant_type = kind

    @contextlib.contextmanager
    def patch(self):
        manager = self
        original_load, original_unload = Linear.load, Linear.unload

        def load(linear, device, **kwargs):
            return manager.install_linear(linear, device, **kwargs)

        def unload(linear):
            if isinstance(linear.inner, StreamInner):
                host = manager.host_cached.pop(id(linear), None)
                if host is not None:
                    manager.host_bytes -= sum(
                        value.untyped_storage().nbytes() for value in host.values()
                    )
                native = manager.cached.pop(id(linear), None)
                if native is not None:
                    manager.bytes -= tensor_bytes(native)
                    native.unload()
                linear.inner = None
                linear.device = None
            else:
                original_unload(linear)

        Linear.load, Linear.unload = load, unload
        try:
            yield
        finally:
            Linear.load, Linear.unload = original_load, original_unload

    def metrics(self):
        return dict(
            weight_loads=self.loads,
            weight_cache_hits=self.hits,
            weight_cache_bytes=self.bytes,
            weight_cache_limit=self.limit,
            weight_upload_bytes=self.loaded_bytes,
            host_weight_cache_bytes=self.host_bytes,
            host_weight_cache_limit=self.host_limit,
            host_weight_cache_hits=self.host_hits,
            weight_load_seconds=round(self.load_seconds, 3),
        )


class StreamingEngine:
    def __init__(self, model_dir, context=512, cache_gib=-1, host_cache_gib=-1):
        if context < 256 or context % 256:
            raise ValueError("Context must be a positive multiple of 256")
        if any(not math.isfinite(v) or v < -1 for v in (cache_gib, host_cache_gib)):
            raise ValueError("Invalid GPU cache budget")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable")
        self.directory = Path(model_dir).resolve()
        self.context = context
        self.manager = WeightManager(
            max(0, int(cache_gib * 2**30)), max(0, int(host_cache_gib * 2**30))
        )
        self.config = Config.from_directory(str(self.directory))
        if self.config.architecture != "GlmMoeDsaForCausalLM":
            raise ValueError(
                f"This backend requires GlmMoeDsaForCausalLM, got {self.config.architecture}"
            )
        # Small chunks keep both weight reconstruction and prompt buffers bounded.
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=context)
        self.tokenizer = Tokenizer.from_config(self.config)
        from transformers import AutoTokenizer

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            str(self.directory), local_files_only=True
        )
        self.patch_context = self.manager.patch()
        self.patch_context.__enter__()
        try:
            self.model.load(device="cuda:0", progressbar=True)
            if cache_gib < 0:
                free, _ = torch.cuda.mem_get_info()
                self.manager.limit = max(0, free - int(1.5 * 2**30))
            if host_cache_gib < 0:
                import psutil

                self.manager.host_limit = max(
                    0, min(2 * 2**30, psutil.virtual_memory().available - 4 * 2**30)
                )
            print(f"GPU weight cache: {self.manager.limit / 2**30:.2f} GiB", flush=True)
            print(f"RAM weight cache: {self.manager.host_limit / 2**30:.2f} GiB", flush=True)
        except BaseException:
            try:
                self.model.unload()
                self.config.stc.close()
            finally:
                self.patch_context.__exit__(None, None, None)
            raise

    def close(self):
        try:
            self.model.unload()
            self.config.stc.close()
        finally:
            self.patch_context.__exit__(None, None, None)

    def forward_tokens(self, ids, position):
        # The runtime memoizes host lengths and device tensors in params. A new
        # dict per forward prevents stale DSA lengths and retained allocations.
        params = dict(
            attn_mode="flash_attn",
            cache=self.cache,
            past_len=position,
            batch_shape=(1, self.context),
        )
        return self.model.forward(ids, params=params)

    @torch.inference_mode()
    def generate(self, prompt, max_tokens=128, echo=True, history=None):
        validate_generation(self.context, max_tokens)
        # Use the published chat template, including its reasoning-effort contract.
        previous = list(history or [])
        while True:
            formatted = self.hf_tokenizer.apply_chat_template(
                previous + [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            ids = self.tokenizer.encode(formatted, encode_special_tokens=True)
            if ids.shape[-1] + max_tokens <= self.context or not previous:
                break
            previous = previous[2:]
        n_prompt = ids.shape[-1]
        if n_prompt + max_tokens > self.context:
            raise ValueError("Prompt plus output exceeds configured context")
        eos = self.config.config_dict.get("eos_token_id", [])
        eos = {eos} if isinstance(eos, int) else set(eos)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        # Small prompt chunks amortize SSD reads without reconstructing matrices.
        for position in range(0, n_prompt - 1, 16):
            self.forward_tokens(ids[:, position : min(position + 16, n_prompt - 1)], position)
        torch.cuda.synchronize()
        generated = []
        printed = ""
        first = None
        stopped_on_eos = False
        decode_started = time.perf_counter()
        last = ids[:, -1:]
        for step in range(max_tokens):
            logits = self.forward_tokens(last, n_prompt - 1 + step)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite model logits")
            token = int(logits[0, -1].argmax().item())
            if first is None:
                first = time.perf_counter()
            if token in eos:
                stopped_on_eos = True
                break
            generated.append(token)
            last = torch.tensor([[token]], dtype=torch.long)
            if echo:
                visible = self.tokenizer.decode(
                    torch.tensor([generated]), decode_special_tokens=True
                )[0].rstrip("\ufffd")
                if visible.startswith(printed):
                    print(visible[len(printed) :], end="", flush=True)
                    printed = visible
        torch.cuda.synchronize()
        ended = time.perf_counter()
        elapsed_decode = ended - decode_started
        output = (
            self.tokenizer.decode(
                torch.tensor([generated], dtype=torch.long), decode_special_tokens=True
            )[0]
            if generated
            else ""
        )
        if echo and output.startswith(printed):
            print(output[len(printed) :], end="", flush=True)
        answer = final_answer(output)
        import psutil

        ram = psutil.Process().memory_info()
        return dict(
            prompt=prompt,
            text=output,
            generated_tokens=len(generated),
            answer=answer,
            finish_reason="stop" if stopped_on_eos else "length",
            history=previous
            + [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
            prompt_tokens=n_prompt,
            decode_seconds=elapsed_decode,
            tokens_per_second=len(generated) / elapsed_decode if elapsed_decode else 0,
            time_to_first_token_seconds=(first - started) if first else None,
            total_seconds=ended - started,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
            peak_ram_bytes=getattr(ram, "peak_wset", ram.rss),
            model_directory=str(self.directory),
            backend="exl3-ssd-stream 0.1.0 / ExLlamaV3 1.5.3 / CUDA",
            **self.manager.metrics(),
        )
