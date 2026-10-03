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
from exllamav3.modules.quant.fp16 import LinearFP16
from .validation import validate_generation, final_answer
from .direct_io import DirectReader, BundleLoader, DiskEmbedding
from .host_buffer import bundle_bytes, pack_tensors


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
        self.resident_prefixes = set()
        self.resident_bytes = 0
        self.memory_trim_events = 0
        self.direct_reader = DirectReader()
        self.bundle_loader = BundleLoader(self.direct_reader)

    def install_linear(self, linear, device, **kwargs):
        self.bundle_loader.registry[linear.key] = linear
        # Routers are small, persistent and their native weight is used directly
        # by ExLlama's bias-aware routing kernel.
        if linear.key.endswith(".gate"):
            return self.original_load(linear, device, **kwargs)
        linear.device = torch.device(device)
        if any(linear.key.startswith(prefix) for prefix in self.resident_prefixes):
            native = self.bundle_loader.load(linear)
            if native is None:
                raise RuntimeError(f"Unsupported resident draft linear: {linear.key}")
            linear.inner = native
            linear.quant_type = "exl3" if isinstance(native, LinearEXL3) else "fp16"
            self.resident_bytes += tensor_bytes(native)
            return
        linear.quant_type = "ssd_stream"
        linear.inner = StreamInner(linear, self)

    def forward(self, linear, x, params, out_dtype):
        if (self.loads + self.hits) % 256 == 0:
            self.guard_ram()
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
                inner = self.bundle_loader.load(linear)
                if inner is None:
                    self.original_load(linear, linear.device)
                    inner = linear.inner
            else:
                self.host_hits += 1
                if "weight" in host:
                    inner = LinearFP16(
                        linear.in_features,
                        linear.out_features,
                        host["weight"].to(linear.device, non_blocking=True),
                        host["bias"].to(linear.device, non_blocking=True)
                        if "bias" in host
                        else None,
                        linear.full_in_features,
                        linear.full_out_features,
                        linear.first_in_feature,
                        linear.first_out_feature,
                        out_dtype=linear.out_dtype,
                        key=linear.key,
                    )
                else:
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
                    and isinstance(inner, (LinearEXL3, LinearFP16))
                    and ".experts." not in linear.key
                    and self.host_bytes + size + 16 <= self.host_limit
                ):
                    values = {}
                    names = (
                        ("suh", "svh", "trellis", "bias", "mcg", "mul1")
                        if isinstance(inner, LinearEXL3)
                        else ("weight", "bias")
                    )
                    for name in names:
                        value = getattr(
                            inner, name + "_tensor" if name in ("mcg", "mul1") else name
                        )
                        if value is None:
                            continue
                        values[name] = value
                    bundle = pack_tensors(values, self.host_limit - self.host_bytes)
                    if bundle is not None:
                        self.host_cached[key] = bundle
                        self.host_bytes += bundle_bytes(bundle)
                # Complete kernels before dropping their backing allocations.
                torch.cuda.synchronize(linear.device)
                inner.unload()
            return result
        finally:
            linear.inner = proxy
            linear.quant_type = kind

    def guard_ram(self):
        if os.environ.get("GLM_MEMORY_GUARD", "0") != "1":
            return
        import psutil

        reserve = int(float(os.environ.get("GLM_RAM_RESERVE_GIB", "1.25")) * 2**30)
        if psutil.virtual_memory().available >= reserve:
            return
        torch.cuda.synchronize()
        loader = self.bundle_loader
        while loader.expert_cache and psutil.virtual_memory().available < reserve:
            victim = min(loader.expert_cache, key=lambda k: loader.access_counts[k])
            removed = loader.expert_cache.pop(victim)
            loader.expert_cache_bytes -= removed[0].numel()
            del removed
            self.memory_trim_events += 1
        while self.host_cached and psutil.virtual_memory().available < reserve:
            victim = next(reversed(self.host_cached))
            removed = self.host_cached.pop(victim)
            self.host_bytes -= bundle_bytes(removed)
            del removed
            self.host_limit = min(self.host_limit, self.host_bytes)
            self.memory_trim_events += 1
        loader.expert_cache_limit = min(
            loader.expert_cache_limit,
            loader.expert_cache_bytes + max(0, psutil.virtual_memory().available - reserve),
        )

    def guard_gpu(self):
        if os.environ.get("GLM_MEMORY_GUARD", "0") != "1":
            return
        torch.cuda.empty_cache()
        reserve = int(float(os.environ.get("GLM_DYNAMIC_GPU_RESERVE_GIB", "0.125")) * 2**30)
        free, _ = torch.cuda.mem_get_info()
        if free >= reserve:
            return
        torch.cuda.synchronize()
        for key in list(reversed(self.cached)):
            native = self.cached[key]
            name = getattr(native, "key", "")
            if (
                name.endswith("lm_head")
                or f"model.layers.{getattr(self, 'mtp_layer', 78)}." in name
            ):
                continue
            self.bytes -= tensor_bytes(native)
            del self.cached[key]
            native.unload()
            del native
            torch.cuda.empty_cache()
            self.limit = min(self.limit, self.bytes)
            self.memory_trim_events += 1
            free, _ = torch.cuda.mem_get_info()
            if free >= reserve:
                break

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
                    torch.cuda.synchronize()
                    manager.host_bytes -= bundle_bytes(host)
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
        self._cache_owner = None
        self._cached_ids = None
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
        from exllamav3.modules.embedding import Embedding

        self.disk_embeddings = []
        for module in self.model.modules:
            if isinstance(module, Embedding):
                disk = DiskEmbedding(module, self.manager.direct_reader)
                self.disk_embeddings.append(disk)
                module.load, module.forward, module.unload = (
                    disk.load,
                    disk.forward,
                    disk.unload,
                )
        self.cache = Cache(self.model, max_num_tokens=context)
        self.tokenizer = Tokenizer.from_config(self.config)
        from .chat import ChatFormatter

        self.hf_tokenizer = ChatFormatter(self.directory)
        self.patch_context = self.manager.patch()
        self.patch_context.__enter__()
        try:
            self.model.load(device="cuda:0", progressbar=True)
            self.manager.bundle_loader.install_prefetch(self.model)
            if cache_gib < 0:
                free, _ = torch.cuda.mem_get_info()
                reserve = float(os.environ.get("GLM_GPU_RESERVE_GIB", "1.5"))
                self.manager.limit = max(0, free - int(reserve * 2**30))
            if host_cache_gib < 0:
                import psutil

                self.manager.host_limit = max(
                    0, min(5 * 2**30, psutil.virtual_memory().available - 3 * 2**30)
                )
            if os.environ.get("GLM_HEAD_FIRST", "0") == "1":
                head = self.model.modules[-1]
                required = sum(self.config.stc.get_tensor_sizes(head.key))
                if required <= self.manager.limit:
                    native = self.manager.bundle_loader.load(head)
                    if native is not None:
                        self.manager.cached[id(head)] = native
                        self.manager.bytes += tensor_bytes(native)
            print(f"GPU weight cache: {self.manager.limit / 2**30:.2f} GiB", flush=True)
            print(
                f"RAM weight cache: {self.manager.host_limit / 2**30:.2f} GiB",
                flush=True,
            )
        except BaseException:
            try:
                self.manager.bundle_loader.close()
                self.model.unload()
                self.config.stc.close()
                self.manager.direct_reader.close()
            finally:
                self.patch_context.__exit__(None, None, None)
            raise

    def close(self):
        try:
            torch.cuda.synchronize()
            self.manager.bundle_loader.close()
            self.model.unload()
            self.config.stc.close()
            self.manager.direct_reader.close()
            self.manager.bundle_loader.registry.clear()
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
        result = self.model.forward(ids, params=params)
        if os.environ.get("GLM_EMPTY_CACHE", "0") == "1":
            torch.cuda.empty_cache()
        return result

    def prefill_tokens(self, ids, position):
        params = dict(
            attn_mode="flash_attn",
            cache=self.cache,
            past_len=position,
            batch_shape=(1, self.context),
        )
        self.model.prefill(ids, params=params)
        if os.environ.get("GLM_EMPTY_CACHE", "0") == "1":
            torch.cuda.empty_cache()

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
        io_started = self.manager.bundle_loader.total_io_bytes
        reuse = 0
        flat_ids = ids[0].tolist()
        if previous and self._cache_owner is self and self._cached_ids:
            for old, new in zip(self._cached_ids, flat_ids[:-1]):
                if old != new:
                    break
                reuse += 1
        self._cache_owner, self._cached_ids = self, None
        torch.cuda.reset_peak_memory_stats()
        self.manager.bundle_loader.start_prefill()
        # Small prompt chunks amortize SSD reads without reconstructing matrices.
        chunk = int(os.environ.get("GLM_PREFILL_CHUNK", "128"))
        if not 1 <= chunk <= 128:
            raise ValueError("Prefill chunk must be 1..128")
        for position in range(reuse, n_prompt - 1, chunk):
            self.prefill_tokens(ids[:, position : min(position + chunk, n_prompt - 1)], position)
        torch.cuda.synchronize()
        self.manager.bundle_loader.finish_prefill()
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
        self._cached_ids = flat_ids + (generated if stopped_on_eos else generated[:-1])
        import psutil

        ram = psutil.Process().memory_info()
        return dict(
            prompt=prompt,
            text=output,
            generated_tokens=len(generated),
            answer=answer,
            finish_reason="stop" if stopped_on_eos else "length",
            history=previous
            + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            prompt_tokens=n_prompt,
            reused_prompt_tokens=reuse,
            token_ids=generated,
            decode_seconds=elapsed_decode,
            tokens_per_second=len(generated) / elapsed_decode if elapsed_decode else 0,
            time_to_first_token_seconds=(first - started) if first else None,
            total_seconds=ended - started,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
            peak_ram_bytes=getattr(ram, "peak_wset", ram.rss),
            model_directory=str(self.directory),
            backend="EXL3 SSD Stream optimized / ExLlamaV3 1.5.3 / CUDA",
            expert_cache_bytes=self.manager.bundle_loader.expert_cache_bytes,
            expert_cache_hits=self.manager.bundle_loader.expert_cache_hits,
            total_io_bytes=self.manager.bundle_loader.total_io_bytes,
            request_io_bytes=self.manager.bundle_loader.total_io_bytes - io_started,
            **self.manager.metrics(),
        )
