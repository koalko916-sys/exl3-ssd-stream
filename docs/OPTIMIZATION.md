# Optimized Windows profile

`--optimized` selects a separate Windows adapter; the original remains the default.
ExLlamaV3 1.5.3 supplies architecture, quantization and computation in both paths.

## Loading and caches

Unbuffered Win32 reads use 4096-byte alignment. Up/gate/down projections share a
grouped expert read. Two workers prefetch selected experts into bounded pinned
buffers. Transfers complete before buffer reuse. Original EXL3 bytes are loaded
without conversion. Nonexpert GPU weights use bounded static admission, with
priority for the head and repeatedly used MTP nonexpert matrices.

Pinned RAM stores nonexpert weights and a bounded raw expert cache with replacement
by observed access frequency. The entire expert bank is never loaded. Embedding
rows are read on demand. cudaHostAlloc/cudaFreeHost buffers have explicit storage
ownership so retired allocations can be freed rather than retained in a pinned pool.

The official template is rendered without another complete vocabulary. Twelve cases
matched AutoTokenizer byte for byte. Prefill chunks default to 128; fresh parameter
dictionaries avoid stale DSA metadata. Chat can reuse exact prefixes, KV and main
hidden states across complete turns.

## Full-model verified speculation

MTP proposes from the checkpoint's next-token layer. The target verifies
`[last token, proposals...]` in a batch. Only the matching prefix and target bonus
token are emitted. Both caches roll back to the accepted prefix, and MTP receives
true main-model hidden states. Every proposal is verified before becoming output.

Lookup prefers long repeated suffixes (8–24 matching tokens) in verified context,
and proposes up to 128 continuation tokens, cropped at turn/EOS boundaries.
Inside MTP drafting, a matching suffix can bootstrap a longer input continuation.
That search is restricted to already-verified input, excluding all current draft
tokens; otherwise self-repetition can yield costly incorrect proposals. The whole
hybrid block is checked by the main model.

Without matching text, decoding uses adaptive MTP8 with smaller windows/cooldown
after rejection. Larger MTP windows, confidence drafting and fully resident MTP
remain disabled experiments in the measured default.

## Defaults and evidence

`optimized/profile.py` lists all settings. Explicit environment variables take
precedence. Automatic cache selection reserves GPU 0.75 GiB and RAM 1.25 GiB;
host weight caching is capped at 5 GiB. Raw expert caching is bounded at 1 GiB and
constrained by available RAM. These are weight-cache budgets, not total allocations.
Memory guards shrink caches; the dynamic GPU threshold is 0.125 GiB before a
verification round, not a promise of that much free memory at peak.

`GLM_NGRAM_WINDOW=0` disables lookup. `--mtp 0` bypasses both MTP and lookup.
`GLM_PREFETCH_WORKERS` permits 1–8 readers; two measured best. `GLM_MEMORY_GUARD=0`
disables guards. Arbitrary concurrent GPU loads may still exhaust memory.

[Full data](../benchmarks/optimization-round2.json) retains all prompts, outputs,
acceptance, settings, code/math/count/followup, copies, IO comparisons and negative
experiments. Cache budgets/background load varied. Reader waits overlap between
threads and are not wall-clock IO time. One count run had unexplained pauses and
0.115 tokens/s; no cause or permanent cure is established.

Final local copy: 169 tokens, 1.166839 tokens/s, 144.8358 s decode, 318.2392 s total,
264.9253 s first output, exact reference IDs. One token is `</think>`; visible output
alone is 1.159934 tokens/s. Arithmetic: 0.305689; development code around 0.21 and
math around 0.39. This is not universal 1 token/s. Different CUDA batch shapes may
change rounding and routing; all-prompt bitwise identity is not guaranteed.

No clean second-machine installation, long-context evaluation or general quality
score is claimed. See public package checks separately from local development data.

## Prior work studied

- [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md): context lookup alongside draft proposals.
- [SpecExec](https://arxiv.org/abs/2406.02532): parallel verification with offloaded weights.
- [ExLlamaV3 expert-cache RFC](https://github.com/turboderp-org/exllamav3/issues/254): predictive expert caching.
- [GLM-5.2 MTP experiment](https://gist.github.com/malaiwah/4bbb16bef2e336e94af165076cdba955): reviewed as prior work; no replacement MTP weights installed.

The adapter implementation is independent. Upstream kernels and licenses are preserved.
