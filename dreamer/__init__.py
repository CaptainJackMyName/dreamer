"""Dreamer: A high-performance, extensible inference engine for video generation models."""

__version__ = "0.0.1"

# Trigger adapter registrations (side-effect import)
from . import adapters  # noqa: F401

from .entrypoints.offline import VideoGenLLM, GenerationConfig
from .core.config import ModelConfig

__all__ = [
    "VideoGenLLM",
    "GenerationConfig",
    "ModelConfig",
]
