"""Configuration classes for Dreamer engine."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ModelConfig:
    """Model loading configuration."""

    model_id: str
    components: Optional[Dict[str, str]] = None
    precision: str = "bf16"
    device_map: str = "auto"
    gpu_memory_utilization: float = 0.9
    offload_folder: Optional[str] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Single generation request configuration."""

    prompt: str = ""
    negative_prompt: str = ""
    num_frames: int = 81
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 50
    seed: Optional[int] = None
    guidance_scale: float = 5.0
    fps: int = 16
    response_format: str = "url"
    image: Optional[Any] = None
    # Model-specific overrides
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelConfig:
    """Distributed parallel configuration."""

    tensor_parallel_size: int = 1
    sequence_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
