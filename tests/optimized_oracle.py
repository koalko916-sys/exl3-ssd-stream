import os
import argparse

os.environ["GLM_PREFETCH"] = "1"
os.environ["GLM_PREFETCH_WORKERS"] = "4"
os.environ["GLM_DYNAMIC_EXPERT_CACHE"] = "1"
from exl3_ssd_stream.optimized.chat import ChatFormatter
from exl3_ssd_stream.optimized.ngram import lookup_draft
from transformers import AutoTokenizer
import torch
from exl3_ssd_stream.optimized.runtime import Config, Linear
from exl3_ssd_stream.optimized.direct_io import DirectReader, BundleLoader

parser = argparse.ArgumentParser(description="Optimized Windows CUDA oracle")
parser.add_argument("--model", required=True)
MODEL = parser.parse_args().model
formatter = ChatFormatter(MODEL)
reference = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
cases = [
    [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
    [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Привет!"}],
    [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "7*8?"},
    ],
]
for messages in cases:
    for effort in ("low", "high"):
        for generation in (False, True):
            kwargs = dict(tokenize=False, add_generation_prompt=generation, reasoning_effort=effort)
            assert formatter.apply_chat_template(
                messages, **kwargs
            ) == reference.apply_chat_template(messages, **kwargs)
print("PASS: 12 official chat-template cases are byte-identical.", flush=True)
assert lookup_draft([1, 2, 3, 4, 5, 8, 1, 2, 3], 2, minimum=3) == [4, 5]
assert lookup_draft(list(range(1, 13)) + [20] + list(range(1, 9)), 4) == [9, 10, 11, 12]
assert lookup_draft(list(range(1, 13)) + [20] + list(range(1, 9)), 20, source_limit=12) == [
    9,
    10,
    11,
    12,
]
assert lookup_draft(list(range(1, 13)) + [20] + list(range(1, 9)), 20, source_limit=0) == []
assert lookup_draft([1, 2, 3, 4], 8) == []
assert lookup_draft([1, 2, 3, 1, 2, 3], 0) == []
print("PASS: ngram suffix, bounds and no-match cases.", flush=True)
del reference


@torch.inference_mode()
def check_experts():
    config = Config.from_directory(MODEL)
    reader = DirectReader()
    loader = BundleLoader(reader)
    originals = {}
    prefix = "model.layers.3.mlp"
    try:
        for expert in (0, 1, 2):
            for projection in ("up_proj", "gate_proj", "down_proj"):
                k, n = (2048, 6144) if projection == "down_proj" else (6144, 2048)
                key = f"{prefix}.experts.{expert}.{projection}"
                linear = Linear(config, key, k, n)
                linear.load(torch.device("cuda:0"))
                originals[key] = linear.inner
                loader.registry[key] = linear
        loader.expert_cache_limit = 30 * 2**20
        loader.hot_experts = {f"{prefix}.experts.{expert}.up_proj" for expert in (0, 1)}
        for selected in ([0, 1], [0, 1], [2], [2], [2], [0, 2]):
            loader.begin_prefetch(prefix, torch.tensor([selected]))
            for expert in sorted(selected):
                for projection in ("up_proj", "gate_proj", "down_proj"):
                    key = f"{prefix}.experts.{expert}.{projection}"
                    linear = loader.registry[key]
                    x = torch.randn(1, 3, linear.in_features, device="cuda:0", dtype=torch.float16)
                    expected = originals[key].forward(x, {})
                    native = loader.load(linear)
                    actual = native.forward(x, {})
                    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
            assert not loader.pending and not loader.prefetch_pending and not loader.prefetch_queue
            assert loader.expert_cache_bytes <= loader.expert_cache_limit
        assert loader.expert_cache_evictions > 0
        assert loader.expert_cache_hits > 0
        print(
            "PASS: four-reader prefetch, dynamic cache hits and eviction; exact real expert outputs.",
            flush=True,
        )
    finally:
        torch.cuda.synchronize()
        loader.close()
        reader.close()
        config.stc.close()


check_experts()


def check_memory_guards():
    import weakref
    from unittest.mock import patch
    from types import SimpleNamespace
    from exl3_ssd_stream.optimized.runtime import WeightManager, LinearFP16, tensor_bytes
    from exl3_ssd_stream.optimized.host_buffer import allocate_bytes

    manager = WeightManager(1 << 20, 1 << 20)
    os.environ["GLM_MEMORY_GUARD"] = "1"
    try:
        host = allocate_bytes(1024)
        host_ref = weakref.ref(host)
        manager.host_cached[1] = {"weight": host}
        manager.host_bytes = 1024
        del host
        with patch(
            "psutil.virtual_memory",
            side_effect=lambda: SimpleNamespace(available=0 if manager.host_bytes else 4 * 2**30),
        ):
            manager.guard_ram()
        assert manager.host_bytes == 0 and not manager.host_cached and host_ref() is None
        matrix = torch.arange(256, dtype=torch.float16).view(16, 16)
        native = LinearFP16(16, 16, matrix.cuda(), None, 16, 16, 0, 0, key="test.safe_matrix")
        x = torch.ones(1, 1, 16, device="cuda:0", dtype=torch.float16)
        before = native.forward(x, {})
        weight_ref = weakref.ref(native.weight)
        manager.cached[1] = native
        manager.bytes = tensor_bytes(native)
        del native
        os.environ["GLM_DYNAMIC_GPU_RESERVE_GIB"] = "12"
        manager.guard_gpu()
        assert not manager.cached and manager.bytes == 0 and weight_ref() is None
        reloaded = LinearFP16(16, 16, matrix.cuda(), None, 16, 16, 0, 0, key="test.safe_matrix")
        torch.testing.assert_close(before, reloaded.forward(x, {}), atol=0, rtol=0)
        reloaded.unload()
        print(
            "PASS: RAM and GPU cache pressure releases backing storage; reloaded computation identical.",
            flush=True,
        )
    finally:
        os.environ["GLM_DYNAMIC_GPU_RESERVE_GIB"] = "0.125"
        manager.bundle_loader.close()
        manager.direct_reader.close()


check_memory_guards()
