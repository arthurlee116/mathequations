"""Trace one-pixel skeleton masks into drawable stroke paths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]
Pixel = tuple[int, int]


@dataclass(frozen=True)
class StrokePath:
    """One traced line-art stroke in pixel coordinates."""

    stroke_id: int
    points: list[Point]
    closed: bool

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for start, end in zip(self.points, self.points[1:]):
            total += math.hypot(end[0] - start[0], end[1] - start[1])
        if self.closed:
            first = self.points[0]
            last = self.points[-1]
            total += math.hypot(first[0] - last[0], first[1] - last[1])
        return total


def _neighbors(pixel: Pixel, pixels: set[Pixel]) -> list[Pixel]:
    y, x = pixel
    result: list[Pixel] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            candidate = (y + dy, x + dx)
            if candidate in pixels:
                result.append(candidate)
    result.sort()
    return result


def _to_points(path: list[Pixel]) -> list[Point]:
    return [(float(x), float(y)) for y, x in path]


def _edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
    return (a, b) if a <= b else (b, a)


def _trace_from_node(
    start: Pixel,
    neighbor: Pixel,
    pixels: set[Pixel],
    degrees: dict[Pixel, int],
    visited_edges: set[tuple[Pixel, Pixel]],
) -> list[Pixel]:
    path = [start, neighbor]
    visited_edges.add(_edge_key(start, neighbor))
    previous = start
    current = neighbor
    while degrees[current] == 2:
        candidates = [pixel for pixel in _neighbors(current, pixels) if pixel != previous]
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


def _trace_loop(start: Pixel, pixels: set[Pixel], visited_pixels: set[Pixel]) -> list[Pixel]:
    path = [start]
    visited_pixels.add(start)
    previous: Pixel | None = None
    current = start
    while True:
        candidates = [pixel for pixel in _neighbors(current, pixels) if pixel != previous]
        candidates = [pixel for pixel in candidates if pixel not in visited_pixels or pixel == start]
        if not candidates:
            break
        next_pixel = candidates[0]
        if next_pixel == start:
            break
        path.append(next_pixel)
        visited_pixels.add(next_pixel)
        previous, current = current, next_pixel
    return path


def trace_skeleton_paths(skeleton: np.ndarray, *, min_length: int = 8) -> list[StrokePath]:
    """Trace skeleton pixels into open paths and simple closed loops."""
    ys, xs = np.where(skeleton > 0)
    pixels = {(int(y), int(x)) for y, x in zip(ys, xs)}
    if not pixels:
        return []
    degrees = {pixel: len(_neighbors(pixel, pixels)) for pixel in pixels}
    nodes = [pixel for pixel, degree in degrees.items() if degree != 2]
    visited_edges: set[tuple[Pixel, Pixel]] = set()
    strokes: list[StrokePath] = []

    for node in sorted(nodes):
        for neighbor in _neighbors(node, pixels):
            key = _edge_key(node, neighbor)
            if key in visited_edges:
                continue
            path = _trace_from_node(node, neighbor, pixels, degrees, visited_edges)
            if len(path) >= min_length:
                strokes.append(StrokePath(len(strokes) + 1, _to_points(path), closed=False))

    visited_loop_pixels: set[Pixel] = set()
    for pixel in sorted(pixels):
        if pixel in visited_loop_pixels or degrees[pixel] != 2:
            continue
        edge_seen = any(pixel in edge for edge in visited_edges)
        if edge_seen:
            continue
        loop = _trace_loop(pixel, pixels, visited_loop_pixels)
        if len(loop) >= min_length:
            strokes.append(StrokePath(len(strokes) + 1, _to_points(loop), closed=True))

    return strokes
