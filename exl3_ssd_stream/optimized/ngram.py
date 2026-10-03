"""Prompt/history lookup drafts; every proposed token still needs target verification."""


def lookup_draft(tokens, max_tokens, minimum=8, maximum=24, source_limit=None):
    if max_tokens <= 0 or len(tokens) < minimum * 2:
        return []
    source_limit = len(tokens) if source_limit is None else min(source_limit, len(tokens))
    for size in range(min(maximum, len(tokens) // 2), minimum - 1, -1):
        suffix = tokens[-size:]
        for begin in range(min(len(tokens) - size - 1, source_limit - size), -1, -1):
            if tokens[begin : begin + size] == suffix:
                continuation = tokens[begin + size : min(source_limit, begin + size + max_tokens)]
                if continuation:
                    return continuation
    return []
