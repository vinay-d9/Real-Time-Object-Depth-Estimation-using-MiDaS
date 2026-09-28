"""Preprocessing, inference, and post-processing for MiDaS predictions."""

import time

import cv2
import numpy as np
import torch

from .model import MidasBundle


def prepare_rgb(image: np.ndarray) -> np.ndarray:
    """Return an image in the RGB uint8 format expected by the transform."""

    if image is None or image.size == 0:
        raise ValueError("The input image is empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The input must be a three-channel color image.")
    return np.ascontiguousarray(image, dtype=np.uint8)


def predict_depth(image_rgb: np.ndarray, bundle: MidasBundle) -> tuple[np.ndarray, float]:
    """Run MiDaS and resize its dense prediction to the input image size."""

    image_rgb = prepare_rgb(image_rgb)
    input_batch = bundle.transform(image_rgb).to(bundle.device)
    start = time.perf_counter()

    with torch.inference_mode():
        prediction = bundle.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return prediction.detach().cpu().numpy().astype(np.float32), elapsed_ms


def normalize_depth(raw_depth: np.ndarray) -> np.ndarray:
    """Normalize relative depth to uint8 for display, without claiming meters."""

    if raw_depth.size == 0:
        raise ValueError("The depth prediction is empty.")
    normalized = cv2.normalize(raw_depth, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def create_depth_map(raw_depth: np.ndarray) -> np.ndarray:
    """Create an RGB visualization where brighter values indicate nearer regions."""

    depth_gray = normalize_depth(raw_depth)
    depth_bgr = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)


def create_overlay(image_rgb: np.ndarray, depth_map_rgb: np.ndarray, opacity: float = 0.45) -> np.ndarray:
    """Blend the original RGB image with its depth visualization."""

    image_rgb = prepare_rgb(image_rgb)
    return cv2.addWeighted(image_rgb, 1 - opacity, depth_map_rgb, opacity, 0)