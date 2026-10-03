"""Input and response validation, usable without CUDA or model weights."""

import math


def validate_generation(context: int, tokens: int) -> None:
    if context < 256 or context % 256:
        raise ValueError("Context must be a positive multiple of 256")
    if tokens < 1 or tokens >= context:
        raise ValueError("Output tokens must be positive and smaller than context")


def cache_budget(value: str) -> float:
    budget = float(value)
    if not math.isfinite(budget) or budget < 0 and budget != -1:
        raise ValueError("Cache budget must be -1 (automatic) or nonnegative")
    return budget


def final_answer(text: str) -> str:
    # The supported GLM chat template opens <think> in the generation prompt.
    # An unfinished reasoning channel must never count as a successful answer.
    if "</think>" not in text:
        return ""
    return text.split("</think>", 1)[1].strip()
