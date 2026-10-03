"""Use real GLM checkpoint matrices, independently of whole-model download."""

import argparse
import json
import struct
import tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file
from exl3_ssd_stream.runtime import Config, Linear, WeightManager

PREFIXES = [
    "model.layers.0.self_attn.q_a_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.3.mlp.experts.0.up_proj",
]


@torch.inference_mode()
def main(source):
    shard = source / "model-00001-of-00039.safetensors"
    if not shard.exists():
        shard = shard.with_name(shard.name + ".partial")
    with shard.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        tensors = {}
        for key, meta in header.items():
            if not any(key.startswith(prefix + ".") for prefix in PREFIXES):
                continue
            start, end = meta["data_offsets"]
            stream.seek(8 + header_size + start)
            data = bytearray(stream.read(end - start))
            if len(data) != end - start:
                raise RuntimeError("Selected real matrix not downloaded yet")
            dtype = {"F16": torch.float16, "I16": torch.int16, "I32": torch.int32}[meta["dtype"]]
            tensors[key] = torch.frombuffer(data, dtype=dtype).reshape(meta["shape"])
    with tempfile.TemporaryDirectory(prefix="real-glm-exl3-") as directory:
        Path(directory, "config.json").write_bytes((source / "config.json").read_bytes())
        save_file(tensors, str(Path(directory, "model.safetensors")))
        config = Config.from_directory(directory)
        torch.manual_seed(17)
        try:
            for prefix in PREFIXES:
                k = tensors[prefix + ".suh"].numel()
                n = tensors[prefix + ".svh"].numel()
                linear = Linear(config=config, key=prefix, in_features=k, out_features=n)
                x = torch.randn(1, 1, k, device="cuda:0", dtype=torch.float16)
                linear.load(torch.device("cuda:0"))
                resident = linear.forward(x, {}).clone()
                linear.unload()
                for host_budget in [0, 128 * 2**20]:
                    manager = WeightManager(0, host_budget)
                    with manager.patch():
                        linear.load(torch.device("cuda:0"))
                        for _ in range(4):
                            streamed = linear.forward(x, {})
                            torch.testing.assert_close(streamed, resident, rtol=0, atol=0)
                        metrics = manager.metrics()
                        linear.unload()
                    assert manager.bytes == 0 and not manager.cached
                    assert manager.host_bytes == 0 and not manager.host_cached
                    if host_budget and ".experts." not in prefix:
                        assert metrics["host_weight_cache_hits"] == 3
                    if ".experts." in prefix:
                        assert metrics["host_weight_cache_hits"] == 0
                    print(
                        json.dumps(dict(matrix=prefix, shape=[n, k], exact=True, **metrics)),
                        flush=True,
                    )
        finally:
            config.stc.close()
    print("PASS: real GLM EXL3 3/4/5-bit matrices streamed without numerical changes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    main(parser.parse_args().model)
