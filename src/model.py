"""Loading and device selection for the official pre-trained MiDaS model."""

from dataclasses import dataclass
import os
from pathlib import Path

import certifi
import torch


MODEL_TYPE = "DPT_Hybrid"


@dataclass
class MidasBundle:
    """The model, its official input transform, and the selected device."""

    model: torch.nn.Module
    transform: object
    device: torch.device
    model_type: str


def available_devices() -> list[str]:
    """Return devices that can be selected in the application."""

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    return devices


def select_device(preference: str = "auto") -> torch.device:
    """Choose the requested accelerator, falling back to a working device."""

    preference = preference.lower()
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device_preference: str = "auto") -> MidasBundle:
    """Load MiDaS from the local Hub cache, downloading it only if needed."""

    # Some macOS Python installations do not expose the system CA bundle to
    # urllib, which Torch Hub uses for its GitHub download.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    device = select_device(device_preference)
    cached_repo = Path(torch.hub.get_dir()) / "intel-isl_MiDaS_master"
    if cached_repo.exists():
        model = torch.hub.load(
            str(cached_repo), MODEL_TYPE, source="local", trust_repo=True
        )
        transforms = torch.hub.load(
            str(cached_repo), "transforms", source="local", trust_repo=True
        )
    else:
        model = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE, trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)

    # DPT-based MiDaS variants use the DPT preprocessing pipeline.
    transform = transforms.dpt_transform
    model.to(device)
    model.eval()
    return MidasBundle(model, transform, device, MODEL_TYPE)