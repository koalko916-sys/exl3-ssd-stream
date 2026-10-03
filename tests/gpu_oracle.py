"""CUDA oracle: complete GLM dense/MoE/DSA path, resident versus SSD weights."""

import json
import tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file
from exl3_ssd_stream.runtime import Config, Model, Cache, WeightManager, StreamingEngine


def fixture(directory, quantized=False):
    torch.manual_seed(1729)
    h, q, kv, heads, nope, rope, v, experts = 128, 128, 128, 2, 64, 64, 64, 4
    cfg = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        model_type="glm_moe_dsa",
        hidden_size=h,
        vocab_size=256,
        num_hidden_layers=3,
        num_attention_heads=heads,
        q_lora_rank=q,
        kv_lora_rank=kv,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=v,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=2,
        indexer_types=["full", "full", "shared"],
        rope_interleave=True,
        indexer_rope_interleave=True,
        intermediate_size=128,
        moe_intermediate_size=128,
        n_shared_experts=1,
        n_routed_experts=experts,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        tie_word_embeddings=False,
        max_position_embeddings=512,
        rope_theta=10000.0,
        eos_token_id=255,
    )
    tensors = {}

    def rand(key, *shape):
        tensors[key] = (torch.randn(*shape) * 0.085).half()

    def norm(key, width, bias=False):
        tensors[key + ".weight"] = torch.ones(width, dtype=torch.float16)
        if bias:
            tensors[key + ".bias"] = torch.zeros(width, dtype=torch.float16)

    rand("model.embed_tokens.weight", 256, h)
    rand("lm_head.weight", 256, h)
    norm("model.norm", h)
    for layer in range(3):
        root = f"model.layers.{layer}"
        norm(root + ".input_layernorm", h)
        norm(root + ".post_attention_layernorm", h)
        a = root + ".self_attn"
        rand(a + ".q_a_proj.weight", q, h)
        norm(a + ".q_a_layernorm", q)
        rand(a + ".q_b_proj.weight", heads * (nope + rope), q)
        rand(a + ".kv_a_proj_with_mqa.weight", kv + rope, h)
        norm(a + ".kv_a_layernorm", kv)
        rand(a + ".kv_b_proj.weight", heads * (nope + v), kv)
        rand(a + ".o_proj.weight", h, heads * v)
        if layer < 2:
            rand(a + ".indexer.wq_b.weight", 2 * 128, q)
            rand(a + ".indexer.wk.weight", 128, h)
            norm(a + ".indexer.k_norm", 128, True)
            rand(a + ".indexer.weights_proj.weight", 2, h)
        prefixes = (
            [root + ".mlp"]
            if layer == 0
            else [root + ".mlp.shared_experts"]
            + [root + f".mlp.experts.{i}" for i in range(experts)]
        )
        for prefix in prefixes:
            for name in ["up_proj", "gate_proj", "down_proj"]:
                rand(prefix + "." + name + ".weight", h, h)
        if layer:
            rand(root + ".mlp.gate.weight", experts, h)
            tensors[root + ".mlp.gate.e_score_correction_bias"] = torch.randn(experts).half() * 0.05
    if quantized:
        for key, value in list(tensors.items()):
            if not key.endswith(".weight") or value.ndim != 2:
                continue
            if any(part in key for part in ["embed_tokens", "kv_b_proj", ".mlp.gate.weight"]):
                continue
            n, k = value.shape
            if n % 128 or k % 128:
                continue
            prefix = key[:-7]
            bits = 3 if ".experts." in key else 5
            if prefix == "lm_head":
                bits = 6
            del tensors[key]
            tensors[prefix + ".trellis"] = torch.randint(
                0, 65536, (k // 16, n // 16, 16 * bits), dtype=torch.int32
            ).short()
            tensors[prefix + ".suh"] = torch.sign(torch.randn(k)).half()
            tensors[prefix + ".svh"] = (torch.sign(torch.randn(n)) * 0.02).half()
            tensors[prefix + ".mul1"] = torch.ones(1, dtype=torch.int16)
    Path(directory, "config.json").write_text(json.dumps(cfg))
    save_file(tensors, str(Path(directory, "model.safetensors")))


@torch.inference_mode()
def evaluate(directory, budget=None, chunk_size=1, sequence=None, host_budget=0):
    config = Config.from_directory(directory)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=512)
    runner = StreamingEngine.__new__(StreamingEngine)
    runner.model, runner.cache, runner.context = model, cache, 512
    manager = WeightManager(budget, host_budget) if budget is not None else None
    import contextlib

    with manager.patch() if manager else contextlib.nullcontext():
        model.load(device="cuda:0")
        result = []
        try:
            ids = torch.tensor([sequence or [1, 7, 12, 34, 9, 3]])
            for position in range(0, ids.shape[-1], chunk_size):
                result.append(
                    runner.forward_tokens(ids[:, position : position + chunk_size], position).cpu()
                )
            metrics = manager.metrics() if manager else {}
        finally:
            model.unload()
            config.stc.close()
        if manager:
            assert (
                manager.bytes == 0
                and not manager.cached
                and manager.host_bytes == 0
                and not manager.host_cached
            )
    return torch.cat(result, dim=1), metrics


def main():
    for quantized in [False, True]:
        with tempfile.TemporaryDirectory(prefix="glm-ssd-oracle-") as directory:
            fixture(directory, quantized)
            resident, _ = evaluate(directory)
            for budget, chunk_size in [(0, 1), (256 * 1024, 1), (256 * 1024, 3)]:
                streamed, metrics = evaluate(directory, budget, chunk_size)
                # Resident EXL3 fuses MoE GEMMs; the streamed path rounds separate
                # FP16 GEMMs. Bound that rounding and require identical token choice.
                rounded = quantized or chunk_size > 1
                torch.testing.assert_close(
                    streamed, resident, rtol=0.01 if rounded else 0, atol=0.002 if rounded else 0
                )
                assert torch.equal(streamed.argmax(-1), resident.argmax(-1))
                assert torch.isfinite(streamed).all()
                assert metrics["weight_cache_bytes"] <= budget
                assert metrics["weight_loads"] > 0
                if budget:
                    assert metrics["weight_cache_hits"] > 0
                print(
                    json.dumps(
                        dict(
                            test="resident_vs_ssd",
                            quantized=quantized,
                            budget=budget,
                            chunk_size=chunk_size,
                            max_logit_difference=float((streamed - resident).abs().max()),
                            identical_tokens=True,
                            **metrics,
                        )
                    ),
                    flush=True,
                )
            sequence = [1, 7, 12, 34, 9, 3] * 3
            resident_prefill, _ = evaluate(directory, sequence=sequence)
            streamed_prefill, metrics = evaluate(directory, 256 * 1024, 16, sequence)
            torch.testing.assert_close(streamed_prefill, resident_prefill, rtol=0.02, atol=0.005)
            assert torch.isfinite(streamed_prefill).all()
            print(
                json.dumps(
                    dict(
                        test="chunked_prefill",
                        quantized=quantized,
                        chunk_size=16,
                        max_logit_difference=float(
                            (streamed_prefill - resident_prefill).abs().max()
                        ),
                        **metrics,
                    )
                ),
                flush=True,
            )
            if quantized:
                streamed, metrics = evaluate(directory, 0, host_budget=256 * 1024)
                torch.testing.assert_close(streamed, resident, rtol=0.01, atol=0.002)
                assert metrics["host_weight_cache_hits"] > 0
                assert metrics["host_weight_cache_bytes"] <= 256 * 1024
                print(json.dumps(dict(test="RAM_weight_cache", **metrics)), flush=True)
    print(
        "PASS: dense, routed/shared experts, sparse DSA and shared indexer; cache released.",
        flush=True,
    )


if __name__ == "__main__":
    main()
