"""Tiny value type so callers can request 4-bit without importing the config module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantSpec:
    load_in_4bit: bool = False
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True
