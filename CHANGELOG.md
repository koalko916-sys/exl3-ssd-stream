# Changelog

## 0.2.0

- Optional Windows `--optimized` profile; original adapter remains the default.
- Direct grouped SSD reads, expert prefetch/cache and explicit pinned-buffer lifetime.
- Verified MTP8, prompt lookup, hybrid bootstrap and exact-prefix chat cache reuse.
- Disk embedding rows, official lightweight chat formatter and memory-pressure guards.
- Positive and negative experiments: 1.167 tokens/s on copying; ordinary prompts
  remain around 0.21–0.40 tokens/s, with substantial first-output waits.
- Optimized CUDA oracle, CPU lookup checks and public-package validation.

## 0.1.0

- Standalone package for the experimentally verified GLM EXL3 SSD loading adapter.
- Bounded non-expert GPU/RAM caches and on-demand routed experts.
- Persistent text CLI with checkpoint chat template and measured JSON reports.
- CUDA resident/streaming oracles and CPU input/response checks.
- Public benchmark from one real full-model run on an RTX 3080 10 GiB.
- English/Russian instructions, credits and explicit hardware/feature limitations.
