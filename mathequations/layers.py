from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquationLayer:
    name: str
    kind: str
    equations: list[str]


@dataclass(frozen=True)
class FillRegion:
    region_id: int
    gray: int
    boundary_segment_ids: list[int]
    label: str
