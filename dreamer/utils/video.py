"""Video export utilities."""

import logging
import os
from typing import Any, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio
except ImportError:
    imageio = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


def _to_numpy_uint8(frames: Any) -> np.ndarray:
    """Normalize frames to (T, H, W, 3) uint8 numpy array."""
    # Handle list of PIL Images
    if isinstance(frames, list):
        if Image is not None and isinstance(frames[0], Image.Image):
            frames = [np.array(f.convert("RGB")) for f in frames]
        frames = np.stack(frames, axis=0)

    if isinstance(frames, np.ndarray):
        arr = frames
    else:
        # Assume torch tensor
        import torch

        if isinstance(frames, torch.Tensor):
            arr = frames.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported video type: {type(frames)}")

    # Normalize shape
    if arr.ndim == 3:
        # (H, W, C) single frame -> (1, H, W, C)
        arr = np.expand_dims(arr, axis=0)
    elif arr.ndim == 4:
        pass  # (T, H, W, C)
    else:
        raise ValueError(f"Unexpected video array shape: {arr.shape}")

    # Normalize dtype
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        arr = (arr * 255).clip(0, 255)
    arr = arr.astype(np.uint8)
    return arr


def save_video(frames: Any, path: str, fps: int = 16):
    """Save video frames to disk using OpenCV or imageio.

    Args:
        frames: List of PIL Images, numpy array (T,H,W,C), or torch tensor.
        path: Output file path (e.g., .mp4).
        fps: Frames per second.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = _to_numpy_uint8(frames)
    t, h, w, c = arr.shape

    if cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {path}")
        for i in range(t):
            # OpenCV expects BGR
            frame = cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()
        logger.info(f"Video saved to {path} ({t} frames @ {fps} fps)")
        return

    if imageio is not None:
        imageio.mimsave(path, arr, fps=fps)
        logger.info(f"Video saved to {path} ({t} frames @ {fps} fps)")
        return

    raise RuntimeError(
        "No video backend available. Install opencv-python or imageio."
    )


def export_to_video(frames: Any, path: str, fps: int = 16) -> str:
    """Convenience wrapper around save_video returning the path."""
    save_video(frames, path, fps=fps)
    return path
