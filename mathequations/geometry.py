from __future__ import annotations

import math

import cv2
import numpy as np


Point = tuple[float, float]


def contour_to_points(contour: np.ndarray) -> list[Point]:
    return [(float(point[0][0]), float(point[0][1])) for point in contour]


def pixel_to_cartesian(
    point: Point,
    *,
    width: int,
    height: int,
    scale: float,
) -> Point:
    u, v = point
    x = (u - width / 2) * scale
    y = (height / 2 - v) * scale
    return (round(x, 6), round(y, 6))


def simplify_contour(contour: np.ndarray, epsilon_px: float = 0.5) -> list[Point]:
    approx = cv2.approxPolyDP(contour, epsilon_px, True)
    return contour_to_points(approx)


def polyline_length(points: list[Point], *, closed: bool = True) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    end_index = len(points) if closed else len(points) - 1
    for index in range(end_index):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def resample_closed_points(points: list[Point], target_count: int) -> list[Point]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(points) <= 1:
        return points[:]

    total_length = polyline_length(points, closed=True)
    if total_length == 0:
        return [points[0]]

    target_count = max(3, target_count)
    step = total_length / target_count
    sampled: list[Point] = []
    distance_to_next = 0.0

    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        x1, y1 = start
        x2, y2 = end
        segment_length = math.hypot(x2 - x1, y2 - y1)
        if segment_length == 0:
            continue

        travelled = 0.0
        while distance_to_next <= segment_length - travelled + 1e-9:
            ratio = (travelled + distance_to_next) / segment_length
            x = x1 + (x2 - x1) * ratio
            y = y1 + (y2 - y1) * ratio
            sampled.append((round(x, 6), round(y, 6)))
            travelled += distance_to_next
            distance_to_next = step
        distance_to_next -= segment_length - travelled

    if len(sampled) > target_count + 1:
        sampled = sampled[:target_count]
    return sampled


def allocate_targets(lengths: list[float], total_target: int) -> list[int]:
    if not lengths:
        return []
    total_length = sum(lengths)
    if total_length <= 0:
        return [0 for _ in lengths]
    allocations = [max(3, round(total_target * length / total_length)) for length in lengths]
    difference = total_target - sum(allocations)
    order = sorted(range(len(lengths)), key=lambda index: lengths[index], reverse=True)
    while difference != 0 and order:
        for index in order:
            if difference == 0:
                break
            if difference > 0:
                allocations[index] += 1
                difference -= 1
            elif allocations[index] > 3:
                allocations[index] -= 1
                difference += 1
    return allocations


def map_contours_to_cartesian(
    contours: list[list[Point]],
    *,
    width: int,
    height: int,
    scale: float,
) -> list[list[Point]]:
    return [
        [pixel_to_cartesian(point, width=width, height=height, scale=scale) for point in contour]
        for contour in contours
    ]
