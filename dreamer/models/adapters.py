"""Base model adapter with registration hooks."""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
import logging

from ..core.config import ModelConfig, GenerationConfig
from ..core.pipeline import PipelineUnit, BasePipeline

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """Base class for all model adapters.

    To add a new model:
      1. Subclass BaseModelAdapter.
      2. Implement load(), generate(), generate_batch(), build_pipeline().
      3. Decorate the class with @register_model("your-model-id").
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._loaded = False

    @abstractmethod
    def load(self):
        """Load model weights and initialize runtime."""
        pass

    @abstractmethod
    def generate(self, config: GenerationConfig) -> Any:
        """Run single-sample generation and return video frames."""
        pass

    @abstractmethod
    def generate_batch(self, configs: List[GenerationConfig]) -> List[Any]:
        """Run batch generation."""
        pass

    @abstractmethod
    def build_pipeline(self) -> BasePipeline:
        """Return the execution pipeline for this model."""
        pass

    def unload(self):
        """Release model resources. Adapters may override."""
        self._loaded = False
        logger.info(f"Adapter for {self.model_config.model_id} unloaded.")

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
