# EXL3 SSD Stream

[![CPU checks](https://github.com/koalko916-sys/exl3-ssd-stream/actions/workflows/checks.yml/badge.svg)](https://github.com/koalko916-sys/exl3-ssd-stream/actions/workflows/checks.yml)

**Experimental:** run a GLM EXL3 checkpoint larger than RAM and VRAM by loading
linear weights from an SSD on demand, using ExLlamaV3's CUDA kernels.

[Русская инструкция](docs/README.ru.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Measured results](benchmarks/README.md)

Verified with **Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw**, a 273 GiB checkpoint,
on **Windows, one RTX 3080 10 GiB, Ryzen 5 7500F and 16 GiB system RAM**.
The complete model answered a prompt. It was slow: **0.073 tokens/s**.
This is a capacity experiment, not a claim of practical chat speed or optimal performance.

## What this adds

- On-demand linear weight loading without converting the EXL3 checkpoint.
- A bounded permanent GPU cache for non-routed matrices.
- An optional bounded pinned RAM cache for non-routed EXL3 matrices.
- Routed experts loaded as needed and released after computation.
- Published GLM chat template, greedy decoding, finite-logit checks and JSON metrics.
- Resident-versus-streamed CUDA oracles for dense, MoE, MLA and DSA computations.

**ExLlamaV3 provides the architecture, EXL3 quantization and CUDA kernels.** This
project supplies the loading and cache adapter. It began as an experiment beside
a local WARP checkout, but runs independently of WARP and does not modify its C engine.
It does not invent a new quantization or make all 753B parameters resident.

## Quick start: verified Windows environment

Requirements: Python 3.13, an NVIDIA CUDA GPU and a driver compatible with CUDA 12.8.
The reference machine had a 10 GiB GPU. Smaller GPUs and other operating systems
have not been validated. Allow roughly 310 GiB free SSD space for the model,
runtime and headroom; the weights alone occupy 272.66 GiB.

```powershell
git clone https://github.com/koalko916-sys/exl3-ssd-stream.git
cd exl3-ssd-stream
powershell -NoProfile -File scripts/install-windows.ps1
```

The installer creates a local `.venv`, installs PyTorch 2.10.0 CUDA 12.8, the official
ExLlamaV3 1.5.3 CPython 3.13 Windows wheel and this package. It does not download weights.

Download the exact checkpoint to an SSD with sufficient space:

```powershell
.venv\Scripts\hf.exe download Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw --revision d06b4f42db97c8bb7a8f72e819b979f132e4a721 --local-dir E:\models\glm53-exl3
```

Run a short prompt, then use the persistent interactive session:

```powershell
.venv\Scripts\python.exe -m exl3_ssd_stream --model E:\models\glm53-exl3 --prompt "What is 2 + 2? Answer briefly." --tokens 256 --report benchmark-exl3.json
.venv\Scripts\python.exe -m exl3_ssd_stream --model E:\models\glm53-exl3 --interactive --context 2048 --tokens 256 --report last-response.json
```

In chat, `/new` clears history; `exit` closes the engine. Older complete turns are
dropped when context fills. `--context` must be a positive multiple of 256.
Use Ctrl+C to interrupt a slow run. Incomplete reasoning is not counted as a final
answer or added to chat history.

For Linux, first install a compatible CUDA PyTorch/ExLlamaV3 environment using the
[upstream instructions](https://github.com/turboderp-org/exllamav3), then `pip install -e .`.
The adapter is portable Python, but Linux inference is **untested** here.
`python -m exl3_ssd_stream --help` works without importing GPU dependencies.

## Memory controls

| Option | Default behavior |
|---|---|
| `--cache-gib -1` | Choose from free VRAM after initial model load, keeping a 1.5 GiB reserve |
| `--cache-gib 0` | Disable the permanent linear-weight GPU cache |
| `--host-cache-gib -1` | Choose up to 2 GiB from available RAM, leaving 4 GiB at selection time |
| `--host-cache-gib 0` | Disable the pinned RAM weight cache |

These budgets cover cached linear weights, **not total process/GPU memory**.
Embeddings, routers, norms, MLA weights, KV cache and temporary matrices also need
memory. Automatic reserves are heuristics, not an OOM guarantee. Explicit budgets
are useful when other applications occupy memory. Close large GPU applications
before running. The reference run selected 4.90 GiB of GPU cache and zero RAM cache.

Run only one model process on this GPU. During package validation, an already-open
interactive GLM occupied about 8.3 GiB dedicated VRAM. Concurrent full-model reruns
became much slower and were stopped; no throughput claim is made for them.
Reducing the cache did not resolve that contention. Keep the GPU available for
one engine before comparing timings.

## Actual result

Prompt: `What is 2 + 2? Answer briefly.` Answer: `2 + 2 = 4`.

| Metric | One measured run, 2026-10-03 |
|---|---:|
| Generated tokens, including the reasoning-close marker | 8 |
| Decode time, including the final EOS forward | 110.13 s |
| Output tokens / decode time | 0.07264 tokens/s |
| Time to first token after generation started | 155.71 s |
| Full request, excluding environment/model startup | 252.11 s |
| Peak allocated VRAM | 8.46 GiB |
| Peak reserved VRAM | 8.65 GiB |
| Peak process working RAM | 3.64 GiB |

[Raw report](benchmarks/rtx3080-glm53.json) includes checkpoint revision, hardware,
software versions and cache counters. This short arithmetic prompt proves a real
full-model response; it is not an evaluation of coding, long context, tool use or
general model quality. The timing is not an upper bound on this hardware's speed.

## Checks

CPU checks require no model or GPU dependencies:

```console
python -m unittest discover -s tests -p "test_*.py" -v
```

CUDA checks require the installed environment:

```console
python -m tests.gpu_oracle
python -m tests.real_matrix_oracle --model /path/to/downloaded/checkpoint
```

The synthetic oracle covers FP16 and 3/5/6-bit EXL3, dense and routed/shared experts,
shared DSA indexers, token/chunk prefill, GPU/RAM cache bounds and cleanup. Separate
FP16 GEMMs can round differently from resident fused MoE kernels; the oracle bounds
the difference and checks token choices in its short sequence. The real-matrix
oracle checks selected 3/4/5-bit checkpoint matrices exactly against resident kernels.
CPU CI does **not** imply CUDA inference has been tested on GitHub runners.

The portable package passed the local CUDA oracles, CPU checks and wheel build.
Its attempted full-model reruns were interrupted due to another interactive engine
occupying the same GPU; see [package validation](benchmarks/package-validation.json).
The full-model timings above belong to the original adapter before packaging.

## Scope and limitations

- One CUDA GPU and one engine per process; the loader patch is process-global and not thread-safe.
- Only `GlmMoeDsaForCausalLM` is accepted. The verified checkpoint is full GLM-5.3;
  GLM-5.3-Flash uses a different architecture and is not supported by this adapter.
- Greedy text generation only; no sampling, API server, tool-call execution, vision or batching service.
- CUDA graphs and load arenas are disabled to avoid retaining evicted weight pointers.
- No MTP speculative decoding, even though the checkpoint includes an MTP layer.
- Disk read volume can be large, especially during prefill. Expert caching,
  prefetching and asynchronous overlap are future work.
- Uses ExLlamaV3 internal APIs and is pinned to 1.5.3; other versions may break it.
- The public Windows installer is provided for reproduction. It has not been
  independently tested on a second clean PC.

## Credits and license

[ExLlamaV3](https://github.com/turboderp-org/exllamav3) by Turboderp provides the
inference implementation. [WARP](https://github.com/sqliteai/warp) inspired the
larger-than-RAM experiment. GLM comes from [Z.ai](https://github.com/zai-org/GLM-5);
the verified weight edit is by dealignai and its EXL3 checkpoint by Infatoshi.
This project is independent of those authors and projects.

Adapter code: [MIT](LICENSE). No model weights, upstream binaries or credentials
are distributed. Dependencies and model weights retain their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md) and the linked model card.
