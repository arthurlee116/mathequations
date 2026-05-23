"""Geometry-aware centerline graph records for Lineart Centerline V2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]
Pixel = tuple[int, int]


@dataclass(frozen=True)
class Endpoint:
    """A skeleton endpoint in public ``(x, y)`` pixel coordinates."""

    endpoint_id: int
    point: Point
    pixel: Pixel
    degree: int


@dataclass(frozen=True)
class Junction:
    """A skeleton junction in public ``(x, y)`` pixel coordinates."""

    junction_id: int
    point: Point
    pixel: Pixel
    degree: int


@dataclass(frozen=True)
class RawBranch:
    """One raw skeleton branch between graph nodes, or one closed loop."""

    branch_id: int
    points: list[Point]
    start_endpoint_id: int | None = None
    end_endpoint_id: int | None = None
    start_junction_id: int | None = None
    end_junction_id: int | None = None
    closed: bool = False

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def length(self) -> float:
        total = 0.0
        for start, end in zip(self.points, self.points[1:]):
            total += math.hypot(end[0] - start[0], end[1] - start[1])
        if self.closed and len(self.points) > 1:
            total += math.hypot(self.points[0][0] - self.points[-1][0], self.points[0][1] - self.points[-1][1])
        return total


@dataclass(frozen=True)
class StrokeChain:
    """A reconstructed centerline stroke candidate."""

    stroke_id: int
    points: list[Point]
    branch_ids: tuple[int, ...] = ()
    closed: bool = False
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def length(self) -> float:
        total = 0.0
        for start, end in zip(self.points, self.points[1:]):
            total += math.hypot(end[0] - start[0], end[1] - start[1])
        if self.closed and len(self.points) > 1:
            total += math.hypot(self.points[0][0] - self.points[-1][0], self.points[0][1] - self.points[-1][1])
        return total


@dataclass
class SkeletonGraph:
    """Skeleton pixels plus node classifications used by Centerline V2."""

    skeleton: np.ndarray
    pixels: set[Pixel]
    degrees: dict[Pixel, int]
    endpoints: list[Endpoint]
    junctions: list[Junction]
    mask: np.ndarray | None = None
    raw_branches: list[RawBranch] = field(default_factory=list)
    diagnostics: dict[str, int | float] = field(default_factory=dict)


def _sort_pixel(pixel: Pixel) -> tuple[int, int]:
    x, y = pixel
    return (y, x)


def _to_point(pixel: Pixel) -> Point:
    x, y = pixel
    return (float(x), float(y))


def _neighbors(pixel: Pixel, pixels: set[Pixel]) -> list[Pixel]:
    x, y = pixel
    result: list[Pixel] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = (x + dx, y + dy)
            if candidate in pixels:
                if dx != 0 and dy != 0 and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                    continue
                result.append(candidate)
    result.sort(key=_sort_pixel)
    return result


def _edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
    return (a, b) if _sort_pixel(a) <= _sort_pixel(b) else (b, a)


def build_skeleton_graph(skeleton: np.ndarray, *, mask: np.ndarray | None = None) -> SkeletonGraph:
    """Build a geometry-aware skeleton graph from a one-pixel mask."""
    ys, xs = np.where(skeleton > 0)
    pixels = {(int(x), int(y)) for y, x in zip(ys, xs)}
    degrees = {pixel: len(_neighbors(pixel, pixels)) for pixel in pixels}

    endpoints: list[Endpoint] = []
    junctions: list[Junction] = []
    for pixel in sorted(pixels, key=_sort_pixel):
        degree = degrees[pixel]
        if degree == 1:
            endpoints.append(Endpoint(len(endpoints) + 1, _to_point(pixel), pixel, degree))
        elif degree >= 3:
            junctions.append(Junction(len(junctions) + 1, _to_point(pixel), pixel, degree))

    return SkeletonGraph(
        skeleton=skeleton.copy(),
        pixels=pixels,
        degrees=degrees,
        endpoints=endpoints,
        junctions=junctions,
        mask=mask.copy() if mask is not None else None,
        diagnostics={
            "pixel_count": len(pixels),
            "endpoint_count": len(endpoints),
            "junction_count": len(junctions),
        },
    )


def _node_maps(graph: SkeletonGraph) -> tuple[dict[Pixel, Endpoint], dict[Pixel, Junction]]:
    endpoint_by_pixel = {endpoint.pixel: endpoint for endpoint in graph.endpoints}
    junction_by_pixel = {junction.pixel: junction for junction in graph.junctions}
    return endpoint_by_pixel, junction_by_pixel


def _trace_from_node(
    start: Pixel,
    neighbor: Pixel,
    graph: SkeletonGraph,
    visited_edges: set[tuple[Pixel, Pixel]],
) -> list[Pixel]:
    path = [start, neighbor]
    visited_edges.add(_edge_key(start, neighbor))
    previous = start
    current = neighbor
    while graph.degrees[current] == 2:
        candidates = [pixel for pixel in _neighbors(current, graph.pixels) if pixel != previous]
        if not candidates:
            break
        next_pixel = candidates[0]
        key = _edge_key(current, next_pixel)
        if key in visited_edges:
            break
        path.append(next_pixel)
        visited_edges.add(key)
        previous, current = current, next_pixel
    return path


def _trace_loop(
    start: Pixel,
    graph: SkeletonGraph,
    visited_edges: set[tuple[Pixel, Pixel]],
) -> list[Pixel]:
    path = [start]
    previous: Pixel | None = None
    current = start
    while True:
        candidates = [pixel for pixel in _neighbors(current, graph.pixels) if pixel != previous]
        candidates = [pixel for pixel in candidates if _edge_key(current, pixel) not in visited_edges]
        if not candidates:
            break
        next_pixel = candidates[0]
        visited_edges.add(_edge_key(current, next_pixel))
        if next_pixel == start:
            break
        path.append(next_pixel)
        previous, current = current, next_pixel
    return path


def trace_raw_branches(graph: SkeletonGraph) -> list[RawBranch]:
    """Trace raw graph branches without deciding final stroke reconstruction."""
    if graph.raw_branches:
        return graph.raw_branches
    endpoint_by_pixel, junction_by_pixel = _node_maps(graph)
    nodes = sorted(
        [pixel for pixel, degree in graph.degrees.items() if degree != 2],
        key=_sort_pixel,
    )
    visited_edges: set[tuple[Pixel, Pixel]] = set()
    branches: list[RawBranch] = []

    for node in nodes:
        for neighbor in _neighbors(node, graph.pixels):
            key = _edge_key(node, neighbor)
            if key in visited_edges:
                continue
            path = _trace_from_node(node, neighbor, graph, visited_edges)
            start = path[0]
            end = path[-1]
            start_endpoint = endpoint_by_pixel.get(start)
            end_endpoint = endpoint_by_pixel.get(end)
            start_junction = junction_by_pixel.get(start)
            end_junction = junction_by_pixel.get(end)
            branches.append(
                RawBranch(
                    branch_id=len(branches) + 1,
                    points=[_to_point(pixel) for pixel in path],
                    start_endpoint_id=start_endpoint.endpoint_id if start_endpoint else None,
                    end_endpoint_id=end_endpoint.endpoint_id if end_endpoint else None,
                    start_junction_id=start_junction.junction_id if start_junction else None,
                    end_junction_id=end_junction.junction_id if end_junction else None,
                )
            )

    for pixel in sorted(graph.pixels, key=_sort_pixel):
        if graph.degrees[pixel] != 2:
            continue
        edges = [_edge_key(pixel, neighbor) for neighbor in _neighbors(pixel, graph.pixels)]
        if any(edge in visited_edges for edge in edges):
            continue
        loop = _trace_loop(pixel, graph, visited_edges)
        if len(loop) >= 2:
            branches.append(
                RawBranch(
                    branch_id=len(branches) + 1,
                    points=[_to_point(item) for item in loop],
                    closed=True,
                )
            )

    graph.raw_branches = branches
    graph.diagnostics["raw_branch_count"] = len(branches)
    return branches


def _unit_vector(dx: float, dy: float) -> tuple[float, float]:
    magnitude = math.hypot(dx, dy)
    if magnitude <= 1e-9:
        return (0.0, 0.0)
    return (dx / magnitude, dy / magnitude)


def estimate_endpoint_tangent(branch: RawBranch, endpoint: Endpoint) -> tuple[float, float]:
    """Estimate the outward tangent at ``endpoint`` using nearby branch pixels."""
    if len(branch.points) < 2:
        return (0.0, 0.0)
    lookahead = min(4, len(branch.points) - 1)
    if branch.points[0] == endpoint.point:
        inner = branch.points[lookahead]
        outer = branch.points[0]
        return _unit_vector(outer[0] - inner[0], outer[1] - inner[1])
    if branch.points[-1] == endpoint.point:
        inner = branch.points[-1 - lookahead]
        outer = branch.points[-1]
        return _unit_vector(outer[0] - inner[0], outer[1] - inner[1])
    raise ValueError("endpoint is not attached to branch")
