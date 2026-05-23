from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    with Image.open(path) as pil_image:
        rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def normalize_white_background(image: np.ndarray, *, white_threshold: int = 245) -> np.ndarray:
    result = image.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result[gray >= white_threshold] = [255, 255, 255]
    return result


def foreground_mask(
    image: np.ndarray,
    *,
    saturation_threshold: int = 40,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_mask = (hsv[:, :, 1] > saturation_threshold).astype(np.uint8) * 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_white_mask = (gray < 245).astype(np.uint8) * 255
    mask = cv2.bitwise_or(saturation_mask, non_white_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_foreground_contours(mask: np.ndarray, *, min_area: float = 50.0) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    useful = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    useful.sort(key=cv2.contourArea, reverse=True)
    return useful


def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No foreground pixels found in mask")
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return (x_min, y_min, x_max, y_max)
