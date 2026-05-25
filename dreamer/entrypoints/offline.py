"""Offline inference entrypoint: VideoGenLLM class.

Mirrors the ergonomics of vLLM's LLM class for quick local scripting.
"""

import logging
from typing import List, Any, Optional

from ..core.config import ModelConfig, GenerationConfig
from ..core.engine import VideoGenEngineCore
from ..utils.video import save_video

logger = logging.getLogger(__name__)


class VideoGenLLM:
    """High-level offline video generation interface.

    Example::

        from dreamer import VideoGenLLM, GenerationConfig

        engine = VideoGenLLM(
            model="wan-2.2-t2v-a14b",
            precision="bf16",
            device_map="auto",
        )

        cfg = GenerationConfig(
            prompt="A cat walking on the beach at sunset",
            num_frames=81,
            width=1280,
            height=720,
            num_inference_steps=50,
            seed=42,
        )

        video = engine.generate(cfg)
        engine.save_video(video, "output.mp4", fps=16)
    """

    def __init__(
        self,
        model: str,
        components: Optional[dict] = None,
        precision: str = "bf16",
        device_map: str = "auto",
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):
        self.model_config = ModelConfig(
            model_id=model,
            components=components,
            precision=precision,
            device_map=device_map,
            gpu_memory_utilization=gpu_memory_utilization,
            extra_args=kwargs,
        )
        self.engine = VideoGenEngineCore(self.model_config)
        # eager-load so that first generate() is not blocked by IO
        self.engine.load()

    def generate(self, config: GenerationConfig) -> Any:
        """Generate a single video."""
        return self.engine.generate(config)

    def generate_batch(
        self,
        configs: List[GenerationConfig],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        """Generate multiple videos."""
        return self.engine.generate_batch(configs, batch_size=batch_size)

    def save_video(self, video: Any, path: str, fps: int = 16) -> str:
        """Save generated frames to a video file.

        Returns the absolute path of the saved file.
        """
        save_video(video, path, fps=fps)
        return path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.adapter.unload()
        return False
