"""Depth Anything 3 on the local GPU: one relative depth map per photo.

The model is loaded once per process and shared, so the API server pays the
load only on its first request. Depth grows with distance from the camera.
"""

import os
import threading
import time
from functools import lru_cache

import cv2
import numpy as np
import torch

DEFAULT_MODEL = os.getenv("DA3_MODEL_NAME", "depth-anything/da3-small")
DEFAULT_RESOLUTION = int(os.getenv("DA3_INPUT_SIZE", "768"))

# One GPU: run one inference at a time even when requests overlap.
_gpu = threading.Lock()


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load_model(name: str = DEFAULT_MODEL, on: str | None = None):
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(name).to(on or device()).eval()


def estimate_depth(
    image_bgr: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    resolution: int = DEFAULT_RESOLUTION,
    size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Depth map (float32) at `size` (width, height; default: the image's) and the inference seconds."""
    with _gpu:
        model = load_model(model_name)
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model.inference([cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)], process_res=resolution)
        seconds = time.perf_counter() - started
    depth = prediction.depth
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32).reshape(depth.shape[-2:])
    if size is None:
        size = (image_bgr.shape[1], image_bgr.shape[0])
    return cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR), seconds
