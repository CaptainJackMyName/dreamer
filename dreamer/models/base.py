"""Abstract interfaces for model components (TextEncoder, DiT, VAE, etc.).

These interfaces define the contracts that any model component must satisfy,
allowing the engine to treat heterogeneous models uniformly.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch


class TextEncoderInterface(ABC):
    """Abstract text encoder (e.g., T5, UMT5)."""

    @abstractmethod
    def encode(self, prompt: str, **kwargs) -> torch.Tensor:
        """Encode text prompt into embeddings."""
        pass

    @abstractmethod
    def load(self, model_path: str, **kwargs):
        pass


class ImageEncoderInterface(ABC):
    """Abstract image encoder (e.g., CLIP, ViT)."""

    @abstractmethod
    def encode(self, image: Any, **kwargs) -> torch.Tensor:
        """Encode image into embeddings or latents."""
        pass

    @abstractmethod
    def load(self, model_path: str, **kwargs):
        pass


class DiTInterface(ABC):
    """Abstract Diffusion Transformer core (e.g., STDiT, WanModel)."""

    @abstractmethod
    def denoise_step(
        self,
        latents: torch.Tensor,
        text_embeds: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predict noise residual for one denoising step."""
        pass

    @abstractmethod
    def load(self, model_path: str, **kwargs):
        pass


class VAEInterface(ABC):
    """Abstract VAE encoder/decoder."""

    @abstractmethod
    def encode(self, video: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode video frames into latents."""
        pass

    @abstractmethod
    def decode(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latents into video frames."""
        pass

    @abstractmethod
    def load(self, model_path: str, **kwargs):
        pass
