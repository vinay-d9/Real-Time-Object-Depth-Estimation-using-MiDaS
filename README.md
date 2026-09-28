# Real-Time Object Depth Estimation using MiDaS

## Overview

This individual mini-project demonstrates practical inference with a genuine pre-trained MiDaS model. Given one RGB image or a video frame, the system predicts a dense relative-depth map and creates an optional overlay.

Student: DUBA VINAY  

## Problem Statement

Estimate the relative depth ordering of objects and regions in a scene using a single RGB image or video frame. This is useful for scene understanding, robotics, navigation assistance, augmented reality, and computer vision demonstrations.

MiDaS estimates relative depth. It does not provide calibrated physical distance in meters.

## Features

- Image upload
- Camera snapshot through Streamlit
- Sampled video-file processing
- Original image, depth map, and overlay
- Automatic CPU/MPS/CUDA device selection
- Explicit preprocessing, inference, resizing, normalization, and color mapping
- Inference time and approximate video processing rate

## Model Used

The project uses **MiDaS DPT-Hybrid**, loaded through the official PyTorch Hub entry point. MiDaS is a pre-trained monocular relative-depth model. Monocular means that it estimates depth from one RGB image without stereo cameras or a depth sensor.

DPT-Hybrid uses a transformer-style DPT encoder and dense-prediction decoder. The encoder extracts visual features, multi-scale representations are combined by the decoder, and a prediction head produces one depth value per spatial location. The values are relative inverse-depth-like predictions, not metric measurements.

## Working Pipeline

```text
Image or video frame
        -> RGB conversion
        -> Official MiDaS transform
        -> Tensor with batch dimension
        -> Pre-trained DPT-Hybrid model
        -> Inference mode prediction
        -> Resize to original dimensions
        -> Min-max normalization
        -> Inferno color mapping
        -> Relative depth map and overlay
```

## Project Architecture

- `src/model.py`: device selection and official model/transform loading.
- `src/inference.py`: input preparation, model inference, output resizing, normalization, and visualization.
- `app.py`: Streamlit image, camera, and video interface.
- `run_inference.py`: minimal command-line demonstration independent of Streamlit.
- `docs/project_notes.md`: report-ready academic explanation.
- `docs/viva_questions.md`: concise viva preparation.

## Installation

Python 3.11 or newer is recommended. Python 3.14 was tested on Apple Silicon for this project.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

The first model run downloads the official MiDaS repository cache and DPT-Hybrid weights. Internet access is required for that first run.

For Streamlit Community Cloud, push this project to a GitHub repository, select `app.py` as the main file, and use Python 3.12. The first cloud request downloads the MiDaS repository and approximately 470 MB of DPT-Hybrid weights, so the initial request can take several minutes. Later requests reuse Streamlit's cached model.

## Running the Application

```bash
source .venv/bin/activate
streamlit run app.py
```

### Streamlit Community Cloud

1. Push the project to GitHub.
2. Open https://share.streamlit.io/ and choose **Deploy an app**.
3. Select the repository, branch, and `app.py` as the main file.
4. Deploy with Python 3.12. The included `runtime.txt` requests this version.
5. Wait for the first model download before uploading an image.

For the minimal command-line test:

```bash
python run_inference.py path/to/image.jpg
```

Outputs are written to `outputs/depth_map.png` and `outputs/overlay.png`.

## Demonstration Results

Run the command-line test or upload an image in Streamlit. The application displays the original input, a colorized relative-depth map, and an overlay. The generated result files are demonstrations produced by actual inference and are not fabricated accuracy evidence.

## Evaluation

This is an inference demonstration, so no new model accuracy is claimed. Evaluation is qualitative: inspect whether nearer and farther scene regions are ordered sensibly, and record inference time or approximate FPS on the available hardware.

## Limitations

- Relative depth is not guaranteed to be distance in meters.
- Monocular depth has inherent scale ambiguity.
- Lighting, reflections, transparency, occlusion, and small objects can reduce quality.
- CPU inference may be too slow for continuous real-time use.
- Video mode samples frames and shows the last processed frame rather than producing a synchronized live stream.

## Future Improvements

Possible extensions include metric-depth models such as ZoeDepth, calibration, object detection combined with depth, stereo or depth-camera fusion, GPU/edge deployment, and a dedicated live-stream component.

## References

- MiDaS repository: https://github.com/isl-org/MiDaS
- PyTorch Hub MiDaS documentation: https://pytorch.org/hub/intelisl_midas_v2/
- MiDaS v3.1 technical report: https://arxiv.org/abs/2307.14460
- Vision Transformers for Dense Prediction: https://arxiv.org/abs/2103.13413
- Streamlit documentation: https://docs.streamlit.io/
- OpenCV documentation: https://docs.opencv.org/