"""Lightweight records for layered equation and fill exports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquationLayer:
    """A named set of equations that share the same drawing role."""

    name: str
    kind: str
    equations: list[str]


@dataclass(frozen=True)
class FillRegion:
    """A filled grayscale region described by its boundary segment IDs."""

    region_id: int
    gray: int
    boundary_segment_ids: list[int]
    label: str
