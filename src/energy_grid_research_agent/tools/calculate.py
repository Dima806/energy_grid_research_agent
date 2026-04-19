from __future__ import annotations

import math


def calculate_metric(metric: str, values: list[float]) -> dict[str, float]:
    """Calculate a power grid metric from a list of numeric values."""
    if not values:
        return {"result": 0.0, "metric": float("nan"), "n": 0.0}

    n = len(values)
    rms = math.sqrt(sum(v**2 for v in values) / n)
    magnitude = math.sqrt(sum(v**2 for v in values))
    impedance = math.sqrt(values[0] ** 2 + values[1] ** 2) if n >= 2 else abs(values[0])
    power_factor = abs(values[0]) / magnitude if magnitude > 0 else 0.0

    ops: dict[str, float] = {
        "mean": sum(values) / n,
        "max": max(values),
        "min": min(values),
        "sum": sum(values),
        "rms": rms,
        "power_factor": power_factor,
        "impedance": impedance,
    }

    result = ops.get(metric.lower(), sum(values) / n)
    return {"result": result, "metric": float(hash(metric) % 1000), "n": float(n)}
