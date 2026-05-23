"""Endpoint gap scoring and bridge selection for Centerline V2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .centerline_graph import (
    Endpoint,
    RawBranch,
    SkeletonGraph,
    estimate_endpoint_tangent,
    trace_raw_branches,
)

Point = tuple[float, float]


@dataclass
class BridgeCandidate:
    """A possible connection between two skeleton endpoints."""

    candidate_id: int
    endpoint_a_id: int
    endpoint_b_id: int
    branch_a_id: int
    branch_b_id: int
    start_point: Point
    end_point: Point
    tangent_a: Point
    tangent_b: Point
    distance: float
    angle_a_degrees: float
    angle_b_degrees: float
    smoothness_degrees: float
    score: float = 0.0
    diagnostics: dict[str, float | int] = field(default_factory=dict)


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _unit_between(start: Point, end: Point) -> tuple[Point, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return (0.0, 0.0), 0.0
    return (dx / distance, dy / distance), distance


def _angle_degrees(a: Point, b: Point) -> float:
    return math.degrees(math.acos(_clamp(_dot(a, b))))


def _endpoint_branch_map(branches: list[RawBranch]) -> dict[int, RawBranch]:
    result: dict[int, RawBranch] = {}
    for branch in branches:
        if branch.start_endpoint_id is not None:
            result[branch.start_endpoint_id] = branch
        if branch.end_endpoint_id is not None:
            result[branch.end_endpoint_id] = branch
    return result


def _as_gray(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _sample_gap_values(image: np.ndarray | None, start: Point, end: Point, *, radius: int = 1) -> list[int]:
    if image is None:
        return []
    height, width = image.shape[:2]
    _, distance = _unit_between(start, end)
    sample_count = max(2, int(round(distance)))
    values: list[int] = []
    for index in range(1, sample_count):
        t = index / sample_count
        x = int(round(start[0] + (end[0] - start[0]) * t))
        y = int(round(start[1] + (end[1] - start[1]) * t))
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        if x0 < x1 and y0 < y1:
            values.append(int(np.max(image[y0:y1, x0:x1])))
    return values


def _mask_evidence(mask: np.ndarray | None, start: Point, end: Point) -> float:
    values = _sample_gap_values(mask, start, end, radius=1)
    if not values:
        return 0.0
    return float(sum(value > 0 for value in values) / len(values))


def _gray_evidence(gray: np.ndarray | None, start: Point, end: Point) -> float:
    if gray is None:
        return 0.0
    height, width = gray.shape[:2]
    _, distance = _unit_between(start, end)
    sample_count = max(2, int(round(distance)))
    dark = 0
    total = 0
    for index in range(1, sample_count):
        t = index / sample_count
        x = int(round(start[0] + (end[0] - start[0]) * t))
        y = int(round(start[1] + (end[1] - start[1]) * t))
        if 0 <= x < width and 0 <= y < height:
            total += 1
            if int(gray[y, x]) < 245:
                dark += 1
    if total == 0:
        return 0.0
    return dark / total


def score_bridge_candidate(
    candidate: BridgeCandidate,
    *,
    mask: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> float:
    """Score a bridge candidate; lower is better."""
    gray_image = _as_gray(gray)
    mask_ratio = _mask_evidence(mask, candidate.start_point, candidate.end_point)
    gray_ratio = _gray_evidence(gray_image, candidate.start_point, candidate.end_point)
    distance_cost = candidate.distance
    tangent_cost = (candidate.angle_a_degrees + candidate.angle_b_degrees) / 18.0
    smoothness_cost = candidate.smoothness_degrees / 18.0
    mask_evidence_bonus = mask_ratio * min(candidate.distance, 8.0) * 0.8
    gray_evidence_bonus = gray_ratio * min(candidate.distance, 8.0) * 0.4
    score = distance_cost + tangent_cost + smoothness_cost - mask_evidence_bonus - gray_evidence_bonus
    candidate.score = float(score)
    candidate.diagnostics = {
        "distance_cost": float(distance_cost),
        "tangent_cost": float(tangent_cost),
        "smoothness_cost": float(smoothness_cost),
        "mask_evidence_bonus": float(mask_evidence_bonus),
        "gray_evidence_bonus": float(gray_evidence_bonus),
        "score": float(score),
    }
    return candidate.score


def find_bridge_candidates(
    graph: SkeletonGraph,
    *,
    max_gap: float,
    angle_threshold_degrees: float,
    mask: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> list[BridgeCandidate]:
    """Find endpoint bridge candidates that pass distance and tangent filters."""
    branches = trace_raw_branches(graph)
    branch_by_endpoint = _endpoint_branch_map(branches)
    candidates: list[BridgeCandidate] = []
    mask_image = mask if mask is not None else graph.mask
    for left_index, endpoint_a in enumerate(graph.endpoints):
        branch_a = branch_by_endpoint.get(endpoint_a.endpoint_id)
        if branch_a is None:
            continue
        for endpoint_b in graph.endpoints[left_index + 1 :]:
            branch_b = branch_by_endpoint.get(endpoint_b.endpoint_id)
            if branch_b is None or branch_a.branch_id == branch_b.branch_id:
                continue
            gap_unit, distance = _unit_between(endpoint_a.point, endpoint_b.point)
            if distance <= 1e-9 or distance > max_gap:
                continue
            tangent_a = estimate_endpoint_tangent(branch_a, endpoint_a)
            tangent_b = estimate_endpoint_tangent(branch_b, endpoint_b)
            reverse_gap = (-gap_unit[0], -gap_unit[1])
            angle_a = _angle_degrees(tangent_a, gap_unit)
            angle_b = _angle_degrees(tangent_b, reverse_gap)
            smoothness = math.degrees(math.acos(_clamp(-_dot(tangent_a, tangent_b))))
            if max(angle_a, angle_b, smoothness) > angle_threshold_degrees:
                continue
            candidate = BridgeCandidate(
                candidate_id=len(candidates) + 1,
                endpoint_a_id=endpoint_a.endpoint_id,
                endpoint_b_id=endpoint_b.endpoint_id,
                branch_a_id=branch_a.branch_id,
                branch_b_id=branch_b.branch_id,
                start_point=endpoint_a.point,
                end_point=endpoint_b.point,
                tangent_a=tangent_a,
                tangent_b=tangent_b,
                distance=distance,
                angle_a_degrees=angle_a,
                angle_b_degrees=angle_b,
                smoothness_degrees=smoothness,
            )
            score_bridge_candidate(candidate, mask=mask_image, gray=gray)
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item.score, item.distance, item.endpoint_a_id, item.endpoint_b_id))
    return candidates


def select_bridges(candidates: list[BridgeCandidate]) -> list[BridgeCandidate]:
    """Greedily accept non-conflicting bridges by score."""
    selected: list[BridgeCandidate] = []
    used_endpoints: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: (item.score, item.distance, item.endpoint_a_id, item.endpoint_b_id)):
        if candidate.endpoint_a_id in used_endpoints or candidate.endpoint_b_id in used_endpoints:
            continue
        selected.append(candidate)
        used_endpoints.add(candidate.endpoint_a_id)
        used_endpoints.add(candidate.endpoint_b_id)
    return selected
