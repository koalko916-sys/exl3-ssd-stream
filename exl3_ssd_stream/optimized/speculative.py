"""Greedy MTP speculation with target verification and cache rollback by length."""

import time
import os
import torch
from exllamav3 import Model, Cache
from .validation import validate_generation, final_answer
from .ngram import lookup_draft
from exllamav3.generator.draft_confidence import DraftConfidenceCalibrator


class MTPRunner:
    def __init__(self, engine, window=1):
        self.engine = engine
        self.window = window
        self.active_window = window
        self.cooldown = 0
        self.grow_draft = os.environ.get("GLM_GROW_DRAFT", "0") == "1"
        self.cached_ids = None
        self.cached_hidden = torch.empty(
            (1, engine.context, engine.config.hidden_size),
            dtype=torch.float16,
            device="cpu",
            pin_memory=True,
        )
        self.draft = Model.from_config(engine.config, component="mtp")
        self.draft.attach_to(engine.model)
        self.cache = Cache(self.draft, max_num_tokens=engine.context)
        self.resident_draft = os.environ.get("GLM_MTP_RESIDENT", "0") == "1"
        if self.resident_draft:
            engine.manager.resident_prefixes.add(f"model.layers.{engine.config.num_hidden_layers}.")
        self.draft.load(device="cuda:0", progressbar=True)
        if self.resident_draft:
            free, _ = torch.cuda.mem_get_info()
            engine.manager.limit = max(
                engine.manager.bytes,
                engine.manager.bytes
                + free
                - int(float(os.environ.get("GLM_GPU_RESERVE_GIB", "0.75")) * 2**30),
            )
            print(
                f"Fully resident native MTP: {engine.manager.resident_bytes / 2**30:.3f} GiB; target GPU cache: {engine.manager.limit / 2**30:.3f} GiB",
                flush=True,
            )
        else:
            engine.manager.bundle_loader.install_prefetch(self.draft)
        self.ngram_window = int(os.environ.get("GLM_NGRAM_WINDOW", "0"))
        self.ngram_bootstrap = os.environ.get("GLM_NGRAM_BOOTSTRAP", "0") == "1"
        confidence = float(os.environ.get("GLM_DRAFT_CONFIDENCE", "0"))
        self.calibrator = (
            DraftConfidenceCalibrator(confidence, min_count=4, burn_in=32) if confidence else None
        )
        if os.environ.get("GLM_MTP_GPU_FIRST", "0") == "1":
            from .runtime import Linear, tensor_bytes

            manager = engine.manager
            for module in self.draft:
                if (
                    not isinstance(module, Linear)
                    or ".experts." in module.key
                    or module.key.endswith(".gate")
                ):
                    continue
                if id(module) in manager.cached:
                    continue
                native = manager.bundle_loader.load(module)
                if native is None:
                    continue
                size = tensor_bytes(native)
                if manager.bytes + size <= manager.limit:
                    manager.cached[id(module)] = native
                    manager.bytes += size
                else:
                    native.unload()
            print(
                f"MTP priority GPU cache + head: {manager.bytes / 2**30:.3f} GiB",
                flush=True,
            )

    def params(self, position, cache, **extra):
        return dict(
            attn_mode="flash_attn",
            cache=cache,
            past_len=position,
            batch_shape=(1, self.engine.context),
            **extra,
        )

    def target(self, ids, position):
        params = self.params(position, self.engine.cache, **self.draft.draft_verifier_params)
        logits = self.engine.model.forward(ids, params=params)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite target logits")
        return logits.argmax(-1).cpu()[0].tolist(), params["export_states"][-1]

    @torch.inference_mode()
    def generate(self, prompt, max_tokens=32, echo=True, history=None):
        e = self.engine
        validate_generation(e.context, max_tokens)
        previous = list(history or [])
        while True:
            formatted = e.hf_tokenizer.apply_chat_template(
                previous + [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            ids = e.tokenizer.encode(formatted, encode_special_tokens=True)
            if ids.shape[-1] + max_tokens <= e.context or not previous:
                break
            previous = previous[2:]
        n = ids.shape[-1]
        if n + max_tokens > e.context:
            raise ValueError("Prompt plus output exceeds configured context")
        eos = e.config.config_dict.get("eos_token_id", [])
        eos = {eos} if isinstance(eos, int) else set(eos)
        boundaries = set(eos)
        for marker in ("<|user|>", "<|assistant|>", "<|system|>", "<|observation|>"):
            marker_ids = e.tokenizer.encode(marker, encode_special_tokens=True)[0].tolist()
            if len(marker_ids) == 1:
                boundaries.add(marker_ids[0])
        start = time.perf_counter()
        io_started = e.manager.bundle_loader.total_io_bytes
        flat_ids = ids[0].tolist()
        reuse = 0
        if previous and e._cache_owner is self and self.cached_ids:
            for old, new in zip(self.cached_ids, flat_ids[:-1]):
                if old != new:
                    break
                reuse += 1
        e._cache_owner, self.cached_ids = self, None
        torch.cuda.reset_peak_memory_stats()
        e.manager.bundle_loader.start_prefill()
        carry = (
            self.cached_hidden[:, reuse - 1 : reuse].to("cuda:0", non_blocking=True)
            if reuse
            else None
        )
        for pos in range(reuse, n - 1, 128):
            chunk = ids[:, pos : min(pos + 128, n - 1)]
            params = self.params(
                pos, e.cache, last_tokens_only=1, **self.draft.draft_verifier_params
            )
            e.model.forward(chunk, params=params)
            hidden = params["export_states"][-1]
            self.cached_hidden[:, pos : pos + chunk.shape[-1]].copy_(hidden)
            if carry is None:
                carry = torch.zeros_like(hidden[:, :1])
            shifted = torch.cat((carry, hidden[:, :-1]), dim=1)
            self.draft.prefill(chunk, self.params(pos, self.cache, target_hidden=shifted))
            carry = hidden[:, -1:].clone()
        torch.cuda.synchronize()
        e.manager.bundle_loader.finish_prefill()
        decoding = time.perf_counter()
        if self.grow_draft:
            self.active_window = min(self.window, 8)
            self.cooldown = 0
        generated, accepted_total, drafted_total, rounds = [], 0, 0, 0
        lookup_tokens, lookup_accepted, lookup_rounds = 0, 0, 0
        round_metrics = []
        position, last = n - 1, ids[:, -1:]
        first, stopped = None, False
        printed = ""
        while len(generated) < max_tokens:
            e.manager.guard_ram()
            e.manager.guard_gpu()
            round_started = time.perf_counter()
            round_io_started = e.manager.bundle_loader.total_io_bytes
            round_reads_started = e.manager.bundle_loader.total_read_seconds
            import psutil

            available_ram = psutil.virtual_memory().available
            available_gpu, _ = torch.cuda.mem_get_info()
            lookup = lookup_draft(
                flat_ids + generated,
                min(self.ngram_window, max_tokens - len(generated) - 1),
            )
            for i, token in enumerate(lookup):
                if token in boundaries:
                    lookup = lookup[:i]
                    break
            width = (
                min(self.active_window, max_tokens - len(generated) - 1) if carry is not None else 0
            )
            draft_ids, draft_hidden = [], carry
            hybrid, hybrid_prefix = False, 0
            draft_scores, reach = [], 1.0
            draft_input = last
            if width == 0 or lookup:
                previous_hidden = (
                    carry
                    if carry is not None
                    else torch.zeros(
                        (1, 1, e.config.hidden_size),
                        dtype=torch.float16,
                        device="cuda:0",
                    )
                )
                self.draft.prefill(
                    last,
                    self.params(position, self.cache, target_hidden=previous_hidden),
                )
            for i in range(0 if lookup else width):
                state = self.draft.forward(
                    draft_input,
                    self.params(position + i, self.cache, target_hidden=draft_hidden),
                )
                sampling = {"export_draft_conf": True} if self.calibrator else {}
                candidate = int(self.draft.sample_from_state(state, sampling).item())
                if candidate in eos:
                    break
                draft_ids.append(candidate)
                draft_input = torch.tensor([[candidate]], dtype=torch.long)
                draft_hidden = state
                if self.ngram_bootstrap and self.ngram_window:
                    context_ids = flat_ids + generated + draft_ids
                    remaining = min(self.ngram_window, max_tokens - len(generated) - 1) - len(
                        draft_ids
                    )
                    extension = lookup_draft(context_ids, remaining, source_limit=len(flat_ids))
                    if not extension and i == width - 1 and remaining > 0:
                        near = lookup_draft(context_ids, 1, minimum=7, source_limit=len(flat_ids))
                        if near:
                            extension = near + lookup_draft(
                                context_ids + near,
                                remaining - 1,
                                source_limit=len(flat_ids),
                            )
                    for j, token in enumerate(extension):
                        if token in boundaries:
                            extension = extension[:j]
                            break
                    if extension:
                        hybrid, hybrid_prefix = True, len(draft_ids)
                        draft_ids.extend(extension)
                        lookup_rounds += 1
                        lookup_tokens += len(extension)
                        break
                if self.calibrator:
                    score = float(sampling["draft_conf"].item())
                    draft_scores.append(score)
                    reach *= self.calibrator.estimate(score)
                    if reach < self.calibrator.confidence:
                        break
            if lookup:
                draft_ids = lookup
                lookup_rounds += 1
                lookup_tokens += len(lookup)
            drafted_seconds = time.perf_counter() - round_started
            verify = torch.tensor([[int(last.item()), *draft_ids]], dtype=torch.long)
            choices, hidden = self.target(verify, position)
            matched = 0
            while matched < len(draft_ids) and choices[matched] == draft_ids[matched]:
                matched += 1
            if self.calibrator and not lookup:
                self.calibrator.decay_step()
                for index, score in enumerate(draft_scores[: matched + 1]):
                    self.calibrator.add_label(score, index < matched)
            emitted = choices[: matched + 1]
            drafted_total += len(draft_ids)
            accepted_total += matched
            if lookup:
                lookup_accepted += matched
            elif hybrid:
                lookup_accepted += max(0, matched - hybrid_prefix)
            rounds += 1
            if first is None:
                first = time.perf_counter()
            accepted_length = len(emitted)
            self.cached_hidden[:, position : position + accepted_length].copy_(
                hidden[:, :accepted_length]
            )
            if accepted_length > 1:
                self.draft.prefill(
                    verify[:, 1:accepted_length],
                    self.params(
                        position + 1,
                        self.cache,
                        target_hidden=hidden[:, : accepted_length - 1],
                    ),
                )
            carry = hidden[:, accepted_length - 1 : accepted_length].clone()
            for token in emitted:
                if token in eos:
                    stopped = True
                    break
                generated.append(token)
            round_metrics.append(
                dict(
                    source="hybrid" if hybrid else ("ngram" if lookup else "mtp"),
                    proposed=len(draft_ids),
                    accepted=matched,
                    emitted=len(emitted),
                    draft_seconds=drafted_seconds,
                    total_seconds=time.perf_counter() - round_started,
                    io_bytes=e.manager.bundle_loader.total_io_bytes - round_io_started,
                    reader_wait_seconds=e.manager.bundle_loader.total_read_seconds
                    - round_reads_started,
                    available_ram_bytes=available_ram,
                    available_gpu_bytes=available_gpu,
                )
            )
            if echo:
                visible = e.tokenizer.decode(torch.tensor([generated]), decode_special_tokens=True)[
                    0
                ].rstrip("\ufffd")
                if visible.startswith(printed):
                    print(visible[len(printed) :], end="", flush=True)
                    printed = visible
            if stopped:
                break
            if draft_ids and not lookup and (not hybrid or matched < hybrid_prefix):
                if matched == len(draft_ids):
                    self.active_window = (
                        min(self.window, max(4, len(draft_ids) * 2))
                        if self.grow_draft
                        else self.window
                    )
                elif matched == 0:
                    self.active_window, self.cooldown = 0, 4
                else:
                    self.active_window = (
                        min(self.window, max(2, matched))
                        if self.grow_draft
                        else min(self.window, 2)
                    )
            elif width == 0 and self.cooldown:
                self.cooldown -= 1
                if self.cooldown == 0:
                    self.active_window = min(self.window, 2)
            position += accepted_length
            last = torch.tensor([[emitted[-1]]], dtype=torch.long)
            torch.cuda.empty_cache()
            if os.environ.get("GLM_PROFILE_ROUNDS", "0") == "1":
                print(
                    f"ROUND {rounds}: accepted={matched}/{len(draft_ids)}, tokens={len(generated)}",
                    flush=True,
                )
        torch.cuda.synchronize()
        end = time.perf_counter()
        text = (
            e.tokenizer.decode(torch.tensor([generated]), decode_special_tokens=True)[0]
            if generated
            else ""
        )
        answer = final_answer(text)
        self.cached_ids = flat_ids + (generated if stopped else generated[:-1])
        if echo:
            if text.startswith(printed):
                print(text[len(printed) :], end="", flush=True)
        import psutil

        memory = psutil.Process().memory_info()
        return dict(
            prompt=prompt,
            text=text,
            answer=answer,
            generated_tokens=len(generated),
            prompt_tokens=n,
            reused_prompt_tokens=reuse,
            token_ids=generated,
            finish_reason="stop" if stopped else "length",
            tokens_per_second=len(generated) / (end - decoding),
            decode_seconds=end - decoding,
            time_to_first_token_seconds=first - start if first else None,
            total_seconds=end - start,
            draft_tokens=drafted_total,
            accepted_draft_tokens=accepted_total,
            verification_rounds=rounds,
            ngram_draft_tokens=lookup_tokens,
            ngram_accepted_tokens=lookup_accepted,
            ngram_rounds=lookup_rounds,
            round_metrics=round_metrics,
            expert_cache_evictions=e.manager.bundle_loader.expert_cache_evictions,
            resident_mtp_bytes=e.manager.resident_bytes,
            memory_trim_events=e.manager.memory_trim_events,
            backend=(
                "EXL3 SSD Stream MTP + prompt lookup"
                if self.ngram_window
                else "EXL3 SSD Stream MTP"
            )
            + " / ExLlamaV3 1.5.3 / CUDA",
            model_directory=str(e.directory),
            expert_cache_bytes=e.manager.bundle_loader.expert_cache_bytes,
            expert_cache_hits=e.manager.bundle_loader.expert_cache_hits,
            total_io_bytes=e.manager.bundle_loader.total_io_bytes,
            request_io_bytes=e.manager.bundle_loader.total_io_bytes - io_started,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
            peak_ram_bytes=getattr(memory, "peak_wset", memory.rss),
            history=previous
            + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            **e.manager.metrics(),
        )

    def close(self):
        self.draft.unload()
        self.cached_hidden = None
        self.cached_ids = None
