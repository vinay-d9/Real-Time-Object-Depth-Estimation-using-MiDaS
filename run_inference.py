"""Minimal MiDaS smoke test and command-line image demonstration."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.inference import create_depth_map, create_overlay, predict_depth
from src.model import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-trained MiDaS on one image.")
    parser.add_argument("image", type=Path, help="Path to an input image")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    image_rgb = np.array(image)
    bundle = load_model(args.device)
    raw_depth, elapsed_ms = predict_depth(image_rgb, bundle)
    depth_map = create_depth_map(raw_depth)
    overlay = create_overlay(image_rgb, depth_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "depth_map.png"), cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output_dir / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Model: {bundle.model_type}")
    print(f"Device: {bundle.device}")
    print(f"Input size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")
    print(f"Inference time: {elapsed_ms:.1f} ms")
    print(f"Saved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()