"""Streamlit application for practical MiDaS relative-depth inference."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.inference import create_depth_map, create_overlay, predict_depth
from src.model import available_devices, load_model


st.set_page_config(page_title="MiDaS Depth Estimation", page_icon="D", layout="wide")


@st.cache_resource(show_spinner="Loading the pre-trained MiDaS model...")
def get_model(device_preference: str):
    return load_model(device_preference)


def show_prediction(image_rgb: np.ndarray, bundle) -> None:
    """Run one image prediction and display all requested visualizations."""

    raw_depth, elapsed_ms = predict_depth(image_rgb, bundle)
    depth_map = create_depth_map(raw_depth)
    overlay = create_overlay(image_rgb, depth_map)

    first, second, third = st.columns(3)
    first.image(image_rgb, caption="Original RGB input", use_container_width=True)
    second.image(depth_map, caption="Relative depth map", use_container_width=True)
    third.image(overlay, caption="Depth overlay", use_container_width=True)
    st.caption(
        f"Model: {bundle.model_type} | Device: {bundle.device} | "
        f"Inference time: {elapsed_ms:.1f} ms"
    )


def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Decode an uploaded image and convert it from PIL RGB to NumPy RGB."""

    return np.array(Image.open(uploaded_file).convert("RGB"))


def process_video(uploaded_file, bundle, max_frames: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Sample a video, process frames, and return the last displayed result."""

    video_bytes = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix or ".mp4") as temporary:
        temporary.write(video_bytes)
        temporary.flush()
        capture = cv2.VideoCapture(temporary.name)
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video file.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // max_frames) if total_frames else 1
        original = depth_map = overlay = None
        processed = 0
        total_ms = 0.0
        frame_index = 0
        while processed < max_frames:
            success, frame_bgr = capture.read()
            if not success:
                break
            if frame_index % frame_step == 0:
                original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                raw_depth, elapsed_ms = predict_depth(original, bundle)
                depth_map = create_depth_map(raw_depth)
                overlay = create_overlay(original, depth_map)
                total_ms += elapsed_ms
                processed += 1
            frame_index += 1
        capture.release()

    if original is None:
        raise ValueError("No readable frames were found in the video.")
    return original, depth_map, overlay, total_ms, processed


def main() -> None:
    st.title("Real-Time Object Depth Estimation")
    st.subheader("Using a pre-trained MiDaS model")
    st.write(
        "This application estimates relative depth from RGB input. "
        "It does not measure physical distance in meters."
    )

    with st.sidebar:
        st.header("Controls")
        mode = st.radio("Input mode", ("Image upload", "Camera snapshot", "Video file"))
        device_choice = st.selectbox("Device", ("auto", *available_devices()))
        st.divider()
        st.markdown("**Color convention**")
        st.caption("The Inferno colormap uses brighter regions for relatively closer predictions.")

    uploaded_image = None
    if mode == "Image upload":
        uploaded_image = st.file_uploader("Upload an RGB image", type=("jpg", "jpeg", "png"))
    elif mode == "Camera snapshot":
        uploaded_image = st.camera_input("Capture an image")
    else:
        uploaded_image = st.file_uploader("Upload a video", type=("mp4", "mov", "avi", "m4v"))

    if uploaded_image is None:
        st.info("Choose an input mode and provide an image, camera snapshot, or video.")
        return

    try:
        bundle = get_model(device_choice)
        if mode != "Video file":
            show_prediction(read_uploaded_image(uploaded_image), bundle)
        else:
            original, depth_map, overlay, total_ms, processed = process_video(uploaded_image, bundle)
            first, second, third = st.columns(3)
            first.image(original, caption="Last sampled RGB frame", use_container_width=True)
            second.image(depth_map, caption="Relative depth map", use_container_width=True)
            third.image(overlay, caption="Depth overlay", use_container_width=True)
            average_ms = total_ms / processed
            st.caption(
                f"Processed {processed} sampled frames | Average inference: {average_ms:.1f} ms/frame | "
                f"Approximate processing rate: {1000 / average_ms:.2f} FPS"
            )
    except Exception as error:
        st.error(f"Unable to process this input: {error}")

    with st.expander("Model and workflow information"):
        st.markdown(
            "**MiDaS DPT-Hybrid** is a pre-trained transformer-based monocular depth model. "
            "The input is converted to RGB, resized and normalized with the official MiDaS "
            "transform, passed through the model in inference mode, resized back to the source "
            "resolution, normalized for display, and color-mapped. The result is relative depth, "
            "so colors show ordering of depth rather than guaranteed meter values."
        )


if __name__ == "__main__":
    main()