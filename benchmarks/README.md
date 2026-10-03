# Evidence and methodology

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
