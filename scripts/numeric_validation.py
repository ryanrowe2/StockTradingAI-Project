"""Explicit real-number validation for prices, features, and probabilities.

NumPy dtype=float silently discards imaginary components from complex arrays.
Reject all complex scalars before coercion, including values with zero imaginary
part; retain ordinary numeric-string support where the public API allows it.
"""

from __future__ import annotations

import numpy as np


def binary_label_array(values, *, context: str) -> np.ndarray:
    """Accept binary labels without silently converting complex values to int."""
    supplied = np.asarray(values)
    if np.iscomplexobj(supplied) or (
        supplied.dtype.kind == "O"
        and any(isinstance(value, (complex, np.complexfloating)) for value in supplied.flat)
    ):
        raise ValueError(f"{context} must be real binary (0 or 1), not complex")
    if not np.isin(supplied, (0, 1)).all():
        raise ValueError(f"{context} must be binary (0 or 1)")
    return supplied.astype(int)


def real_float_array(
    values, *, context: str, allow_missing: bool = False
) -> np.ndarray:
    """Convert to float64 without ever discarding complex information."""
    original = np.asarray(values)
    if np.iscomplexobj(original) or (
        original.dtype.kind == "O"
        and any(isinstance(v, (complex, np.complexfloating)) for v in original.flat)
    ):
        raise ValueError(f"{context}: complex values are not permitted")
    try:
        converted = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: real numeric values are required") from exc
    valid = ~np.isinf(converted) if allow_missing else np.isfinite(converted)
    if not valid.all():
        raise ValueError(f"{context} must be finite (nonfinite input)")
    return converted


def real_float_or_nan(value: object) -> float:
    """Coerce one raw market scalar to a real number, or mark it invalid."""
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
