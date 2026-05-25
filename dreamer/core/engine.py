"""Video generation engine core shared by offline and online modes."""

import logging
from typing import List, Any, Optional

from .config import ModelConfig, GenerationConfig
from .registry import get_adapter
from .pipeline import PipelineContext

logger = logging.getLogger(__name__)


class VideoGenEngineCore:
    """Core inference engine that orchestrates model adapters."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        adapter_cls = get_adapter(model_config.model_id)
        self.adapter = adapter_cls(model_config)
        self._loaded = False

    def load(self):
        """Lazy-load the underlying model components."""
        if not self._loaded:
            logger.info(f"Loading adapter for {self.model_config.model_id} ...")
            self.adapter.load()
            self._loaded = True
            logger.info("Adapter loaded successfully.")

    def generate(self, config: GenerationConfig) -> Any:
        """Synchronous single-sample generation (offline mode)."""
        self.load()
        return self.adapter.generate(config)

    def generate_batch(self, configs: List[GenerationConfig], batch_size: Optional[int] = None) -> List[Any]:
        """Synchronous batch generation (offline mode)."""
        self.load()
        if batch_size is None or batch_size >= len(configs):
            return self.adapter.generate_batch(configs)

        results: List[Any] = []
        for i in range(0, len(configs), batch_size):
            chunk = configs[i : i + batch_size]
            results.extend(self.adapter.generate_batch(chunk))
        return results

    async def generate_async(self, config: GenerationConfig) -> Any:
        """Asynchronous generation (online mode)."""
        self.load()
        # Adapters may override with true async; default delegates to sync.
        return self.adapter.generate(config)

    def build_pipeline(self):
        """Expose the adapter's pipeline for introspection or custom execution."""
        self.load()
        return self.adapter.build_pipeline()
