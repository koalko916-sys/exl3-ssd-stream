# Contributing

Useful work includes expert caching, prefetch/compute overlap, larger prefill
chunks, better static admission and broader hardware validation. Open an issue
with your environment and a reproducible case before proposing architecture changes.

For numerical changes, compare with resident ExLlamaV3 and retain the existing
GPU oracle. Include cache-bound and teardown checks when changing memory ownership.
For performance changes, report matched prompts, flags and hardware before/after;
include TTFT, decode rate, total request time and peak memory. Do not substitute
matrix-only microbenchmarks for full-model response measurements.

CPU formatting/lint and validation checks run in CI. GPU tests require a local
CUDA device; document which tests actually ran. Model weights are never committed.
