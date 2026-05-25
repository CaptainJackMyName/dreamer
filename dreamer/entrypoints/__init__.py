"""User-facing entrypoints: offline LLM-style API and online REST server."""

from .offline import VideoGenLLM, GenerationConfig

__all__ = ["VideoGenLLM", "GenerationConfig"]
