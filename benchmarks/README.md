# Evidence and methodology

## Version 0.2 optimized experiments

`optimization-round2.json` retains measurements from the local optimized adapter,
including final fresh-process math/copy validation. Local paths were removed
recursively; prompts are synthetic. Hashes refer to the measured local source.
Public packaging changes namespace, CLI routing and diagnostic labels; its checks
are in `optimized-package-validation.json`.

The public optimized package passed CPU/CLI, real-expert/template/cache-pressure
CUDA checks and wheel build. Seven numerical core modules are AST-identical to
the measured adapter after normalizing diagnostic labels, and profile defaults match.
Its attempted full-model rerun was interrupted because the user's pre-existing
interactive GLM occupied the same GPU/RAM. Only the benchmark was stopped; no
successful public-package rerun or throughput is claimed. Local successful runs
remain separately identified rather than relabeled as package measurements.

Final local copy: 169 tokens, 1.166839 tokens/s, 144.8358 s decode, 318.2392 s total,
264.9253 s first output. All token IDs matched the reference. Visible output alone
is 1.159934 tokens/s; total-request throughput is 0.531. Arithmetic: 0.305689 tokens/s.
Neither is a general model benchmark. Decode time includes EOS compute, excludes
prompt processing; EOS is excluded from the output count. Background/cache budgets varied.

Code/math/count/followup, lower-performing variants and unexplained count pauses
are retained. `optimized-oracle.txt` records development checks with paths redacted.

## Historical version 0.1 evidence

`rtx3080-glm53.json` is the anonymized original full-model report. Only the local
model-directory path was removed; hardware, package versions, checkpoint revision
and provenance were added. Numeric measurements are unchanged. Prompt and answer
are the actual outputs, not an illustrative example.

The request used `--tokens 256`, `--context 512`, automatic GPU and host cache
selection, greedy argmax decoding and `reasoning_effort="low"`. The model stopped
on a published EOS ID after 8 non-EOS output tokens, including `</think>`.

Timing starts after environment and model initialization. Prefill contributes to
total request time and TTFT. Decode timing starts before the last prompt token's
forward and ends after the EOS forward. Thus `generated_tokens / decode_seconds`
includes EOS compute in its denominator. It is an end-to-end decoder rate rather
than a kernel-only or post-first-token steady-state measurement. CUDA is synchronized
at timing boundaries. One run does not establish stable speed or maximum throughput.

The report's 239.33 billion uploaded bytes and 214.05 s weight-load time cover
both prefill and generation. They are not per-token disk bandwidth measurements;
driver caching and loader behavior also affect actual physical SSD traffic.

`synthetic-oracle.jsonl` and `real-matrix-oracle.jsonl` retain the original JSON
records from correctness tests. Their terminal PASS lines were removed to keep
valid JSONL. Synthetic quantized tensors exercise code paths, not model quality.
The actual checkpoint test covers three selected matrices, not every matrix.

The original full-model run preceded portable packaging. The source adapter was
preserved; packaging split out the CLI, removed local paths and added input checks.
Additional package checks are recorded in `package-validation.json` after execution.

The package passed the synthetic and real-matrix CUDA oracles, four CPU tests,
lint/format/syntax checks, CLI help, wheel build and GitHub CPU CI. Full-model
package reruns were interrupted when a separate interactive GLM was found using
about 8.3 GiB dedicated VRAM on the same GPU. Reducing the test's cache did not
resolve contention. The user's interactive process was kept running, and only
the benchmark processes were stopped. No rerun throughput or successful package
full-model response is claimed. The original full-model response remains the
recorded evidence, with its provenance clearly identified.

To contribute a result, include GPU/driver, CPU, RAM, storage, Python/PyTorch/ExLlama
versions, model revision, flags, number of runs and prompt/output token counts.
Redact local paths and do not submit credentials or private prompts.
