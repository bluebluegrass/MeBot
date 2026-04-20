from __future__ import annotations


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    high = max(values)
    low = min(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]
