"""Measured Windows profile; explicit environment settings take precedence."""

import os

DEFAULTS = {
    "GLM_GROUP_EXPERTS": "1",
    "GLM_PREFETCH": "1",
    "GLM_EXPERT_CACHE": "1",
    "GLM_HEAD_FIRST": "1",
    "GLM_EMPTY_CACHE": "1",
    "GLM_GPU_RESERVE_GIB": "0.75",
    "GLM_PREFILL_CHUNK": "128",
    "GLM_PREFETCH_WORKERS": "2",
    "GLM_DYNAMIC_EXPERT_CACHE": "1",
    "GLM_MTP_GPU_FIRST": "1",
    "GLM_RAM_RESERVE_GIB": "1.25",
    "GLM_NGRAM_WINDOW": "128",
    "GLM_NGRAM_BOOTSTRAP": "1",
    "GLM_GROW_DRAFT": "0",
    "GLM_DRAFT_CONFIDENCE": "0",
    "GLM_MTP_RESIDENT": "0",
    "GLM_MEMORY_GUARD": "1",
    "GLM_DYNAMIC_GPU_RESERVE_GIB": "0.125",
}


def apply_defaults():
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
