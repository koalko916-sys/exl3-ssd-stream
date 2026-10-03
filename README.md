# EXL3 SSD Stream

[![CPU checks](https://github.com/koalko916-sys/exl3-ssd-stream/actions/workflows/checks.yml/badge.svg)](https://github.com/koalko916-sys/exl3-ssd-stream/actions/workflows/checks.yml)

Experimental SSD streaming for a GLM EXL3 checkpoint larger than RAM and VRAM,
using **ExLlamaV3's architecture, quantization and CUDA kernels**.

[Русская инструкция](docs/README.ru.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Measurements](benchmarks/README.md) · [Optimization details](docs/OPTIMIZATION.md)

Verified with **Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw**, a **273 GiB / 753B**
checkpoint, on **Windows, RTX 3080 10 GiB, Ryzen 5 7500F and 16 GiB RAM**.
Weights are unchanged: no expert pruning, new quantization or smaller replacement model.

**v0.2.0:** optional Windows direct SSD reads, expert prefetch/cache, verified MTP8
and prompt lookup. An installed local run reached **1.167 tokens/s on exact copying**
of existing input text (169 output tokens). **This is not a general 1 token/s claim.**
Ordinary test prompts measured about **0.21–0.40 tokens/s**. Long waits remain.

## Quick start on Windows

Requirements: Python 3.13, NVIDIA CUDA GPU and a driver compatible with CUDA 12.8.
Allow roughly 310 GiB free SSD space for weights, runtime and headroom.
The verified machine had a 10 GiB GPU; smaller GPUs have not been validated.

```powershell
git clone https://github.com/koalko916-sys/exl3-ssd-stream.git
cd exl3-ssd-stream
powershell -NoProfile -File scripts/install-windows.ps1
.venv\Scripts\hf.exe download Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw --revision d06b4f42db97c8bb7a8f72e819b979f132e4a721 --local-dir E:\models\glm53-exl3
.venv\Scripts\python.exe -m exl3_ssd_stream --optimized --mtp 8 --model E:\models\glm53-exl3 --context 2048 --tokens 256 --prompt "What is 2 + 2? Answer briefly." --report benchmark-exl3.json
.venv\Scripts\python.exe -m exl3_ssd_stream --optimized --mtp 8 --model E:\models\glm53-exl3 --interactive --context 2048 --tokens 256 --report last-response.json
```

The installer creates `.venv` and installs PyTorch 2.10.0 CUDA 12.8, the official
ExLlamaV3 1.5.3 CPython 3.13 Windows wheel and this adapter. It does not download weights.
To update an existing clone: `git pull`, then `.venv\Scripts\python.exe -m pip install -e .`.

`/new` clears history; `exit` closes the engine. Use only one engine per GPU.
Responses can take several minutes. Copied fragments appear in blocks after full
target verification. Incomplete reasoning is not added to history. `--context` must
be a positive multiple of 256; `--tokens` must be smaller than the context.

Omit `--optimized` to retain the original v0.1 loading path. The optimized path uses
Win32 direct I/O and is **Windows-only**. Linux inference for the original adapter
remains untested; install a compatible upstream CUDA environment first.
`--help` and CPU tests work without importing GPU dependencies. WARP is not required.

## What changed

- Grouped expert reads, two bounded prefetch workers and explicitly freed pinned buffers.
- Bounded GPU/RAM caches, dynamic frequency-based expert replacement and MTP GPU priority.
- Disk-backed embedding row reads and a lighter renderer of the official chat template.
- Exact-prefix KV reuse across complete chat turns.
- Full-model verification of MTP and prompt/history lookup, including hybrid bootstrap.
- Memory-pressure guards; rejected proposals never become unverified output.

The adapter runs independently of WARP and does not modify its C engine. It does
not make the complete model resident. ExLlamaV3 supplies the inference kernels.

## Measured results

Individual runs on the reference PC, not stable throughput guarantees:

| Test | Mode | Output tokens | Decode tokens/s | Full request |
|---|---|---:|---:|---:|
| Short arithmetic, historical v0.1 | Original adapter | 8 | 0.073 | 252 s |
| Exact short copy | Previous local MTP8 | 75 | 0.296 | 350 s |
| Same short copy, identical token IDs | Hybrid MTP + lookup | 75 | 1.008 | 169 s |
| Long exact copy, installed local profile | Hybrid MTP + lookup | 169 | **1.167** | **318 s** |
| Short arithmetic, installed local profile | MTP8 | 8 | 0.306 | 73 s |

Long-copy first output appeared at **265 s**; decoding took **145 s**. Its 168 visible
tokens also exceeded 1 token/s (**1.160**). Rates count non-EOS output tokens, including
the reasoning-close marker, and include EOS computation in decode time. They exclude
prompt processing and model initialization. Large blocks improve completion time
but can delay first output. Total-request throughput for this copy is only 0.531 tokens/s.

All long-copy token IDs matched the development reference. Automatic RAM budgets
and background load varied; these are combined configuration results, not isolated
algorithm speedups. Copying favors lookup because it reuses the prompt. Novel output
usually has less draft acceptance; universal 1 token/s has not been reached.

[Raw evidence](benchmarks/optimization-round2.json) retains successful and unsuccessful
experiments, prompts, outputs, cache budgets, TTFT and timings. It describes the local
adapter before public namespace packaging. [Public package validation](benchmarks/optimized-package-validation.json)
records checks of the packaged code. Historical evidence is retained separately.

## Memory controls

Profile defaults apply only when the corresponding environment variable is absent.
`--cache-gib` and `--host-cache-gib` override automatic cache selection: -1 selects
automatically, 0 disables that weight cache. Budgets cover cached weights, not total
process memory; automatic reserves are heuristics rather than an OOM guarantee.

| Optimized setting | Default |
|---|---:|
| `--mtp` | 8; use 0 for serial generation |
| `GLM_NGRAM_WINDOW` | 128; use 0 to disable lookup |
| `GLM_PREFETCH_WORKERS` | 2 |
| `GLM_GPU_RESERVE_GIB` | 0.75 GiB at cache selection |
| `GLM_RAM_RESERVE_GIB` | 1.25 GiB |
| `GLM_DYNAMIC_GPU_RESERVE_GIB` | 0.125 GiB before verification rounds |

SSD testing measured 2.08 GB/s with two readers; four/eight were slower. MTP16/32
and fully resident MTP did not consistently improve ordinary prompts. Large dynamic
GPU reserves evicted useful weights. One count run had unexplained long pauses;
its result is preserved without claiming a proven cause or cure.

## Checks

```console
python -m unittest discover -s tests -p "test_*.py" -v
python -m tests.gpu_oracle
python -m tests.real_matrix_oracle --model /path/to/checkpoint
python -m tests.optimized_oracle --model E:\models\glm53-exl3
```

The first command needs no model or GPU dependencies. CUDA checks need the installed
environment; the optimized oracle requires Windows. It covers official chat templates,
real expert outputs, prefetch/cache eviction and memory-pressure storage release.
CPU CI is not GPU validation. Generated Python answers were checked syntactically,
**never executed**.

## Scope and license

Only `GlmMoeDsaForCausalLM` is accepted. GLM-5.3-Flash has a different architecture.
One GPU, greedy text decoding; no API server, tool execution, vision or batching service.
Loader patching is process-global; concurrent engines are unsupported. CUDA graphs
and arenas are disabled to avoid retaining streamed pointers. CUDA rounding can vary
with batch size; identical token IDs are not promised for every prompt. Internal APIs
are pinned to ExLlamaV3 1.5.3. No clean second-machine installation is claimed.

Adapter: [MIT](LICENSE). No weights, upstream binaries or credentials are distributed.
[ExLlamaV3](https://github.com/turboderp-org/exllamav3), [WARP](https://github.com/sqliteai/warp),
[Z.ai](https://github.com/zai-org/GLM-5), dealignai and Infatoshi retain their credits
and licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). This project is independent.
