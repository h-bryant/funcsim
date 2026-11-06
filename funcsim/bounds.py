"""
Validation of optional support bounds, shared by `fit`, `compare`, and `Kde`.
"""
import numbers
from typing import Optional
import numpy as np


def validate_bound(value, name: str) -> Optional[float]:
    """
    Return None (no bound) or the bound as a finite float.

    Bools are rejected because they are the legacy "which group" flags of
    `compare`.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a float or None, not a bool")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a float or None, "
                        f"not {type(value).__name__}")
    bound = float(value)
    if not np.isfinite(bound):
        raise ValueError(f"{name} must be finite, not {value}")
    return bound


def check_bounds_against_data(dataA: np.ndarray,
                              lower: Optional[float],
                              upper: Optional[float]) -> None:
    """
    Raise ValueError unless lower < upper and every observation lies in
    [lower, upper] (observations exactly on a bound are allowed).
    """
    if lower is not None and upper is not None and not lower < upper:
        raise ValueError(f"lowerBound ({lower}) must be strictly less than "
                         f"upperBound ({upper})")
    if lower is not None and dataA.min() < lower:
        raise ValueError(f"lowerBound ({lower}) exceeds the smallest "
                         f"observation ({dataA.min()})")
    if upper is not None and dataA.max() > upper:
        raise ValueError(f"upperBound ({upper}) is below the largest "
                         f"observation ({dataA.max()})")
