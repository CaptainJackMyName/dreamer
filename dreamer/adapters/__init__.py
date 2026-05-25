"""Concrete model adapters."""

# Import to trigger adapter registration
from .wan import WanAdapter

__all__ = ["WanAdapter"]
