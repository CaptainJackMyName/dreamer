"""Wan model adapter for Dreamer engine.

Supports Wan-AI/Wan2.2-T2V-A14B-Diffusers and compatible checkpoints.
Uses diffusers under the hood for reliable, production-ready inference.
"""

import logging
from typing import List, Any, Optional

import torch

from ...core.config import ModelConfig, GenerationConfig
from ...core.registry import register_model
from ...core.pipeline import PipelineContext, BasePipeline, PipelineUnit
from ...models.adapters import BaseModelAdapter

logger = logging.getLogger(__name__)


class PromptEmbedUnit(PipelineUnit):
    name = "prompt_embed"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.set("stage", "prompt_embed")
        return context


class NoiseInitUnit(PipelineUnit):
    name = "noise_init"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.set("stage", "noise_init")
        return context


class DiTDenoiseUnit(PipelineUnit):
    name = "dit_denoise"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.set("stage", "dit_denoise")
        return context


class VAEDecodeUnit(PipelineUnit):
    name = "vae_decode"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.set("stage", "vae_decode")
        return context


@register_model("Wan2.2-T2V-A14B-Diffusers")
@register_model("Wan2.2-TI2V-5B-Diffusers")
class WanAdapter(BaseModelAdapter):
    """Adapter for Wan text-to-video diffusion models.

    Registered IDs:
      - Wan2.2-T2V-A14B-Diffusers (default repo: Wan-AI/Wan2.2-T2V-A14B-Diffusers)
      - Wan2.2-TI2V-5B-Diffusers (default repo: Wan-AI/Wan2.2-TI2V-5B-Diffusers)
    """

    # Mapping of registered model_id -> default Hugging Face repo_id
    DEFAULT_REPOS = {
        "Wan2.2-T2V-A14B-Diffusers": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan2.2-TI2V-5B-Diffusers": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    }

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.pipe: Optional[Any] = None
        self.repo_id: str = self._resolve_repo_id()
        self.torch_dtype = self._resolve_dtype()
        self.device = self._resolve_device()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_repo_id(self) -> str:
        # 1) explicit override via extra_args
        override = self.model_config.extra_args.get("repo_id")
        if override:
            return override
        # 2) lookup default mapping
        default = self.DEFAULT_REPOS.get(self.model_config.model_id)
        if default:
            return default
        # 3) treat model_id itself as a huggingface repo_id
        return self.model_config.model_id

    def _resolve_dtype(self) -> torch.dtype:
        precision = self.model_config.precision.lower()
        if precision in ("fp16", "float16"):
            return torch.float16
        if precision in ("bf16", "bfloat16"):
            return torch.bfloat16
        return torch.float32

    def _resolve_device(self) -> str:
        dev = self.model_config.extra_args.get("device")
        if dev:
            return dev
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # BaseModelAdapter interface
    # ------------------------------------------------------------------
    def load(self):
        if self._loaded:
            return

        logger.info(
            f"Loading Wan adapter: model_id={self.model_config.model_id}, "
            f"repo_id={self.repo_id}, dtype={self.torch_dtype}, device={self.device}"
        )

        # Lazy import to avoid hard dependency at import time
        from diffusers.pipelines.wan.pipeline_wan import WanPipeline
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

        extra = dict(self.model_config.extra_args)
        for key in ("repo_id", "device"):
            extra.pop(key, None)

        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            **extra,
        }

        if self.model_config.device_map:
            load_kwargs.setdefault("device_map", self.model_config.device_map)

        try:
            vae = AutoencoderKLWan.from_pretrained(self.repo_id, subfolder="vae", torch_dtype=self.torch_dtype)
            self.pipe = WanPipeline.from_pretrained(self.repo_id, vae=vae, **load_kwargs)
        except Exception as exc:
            logger.error(f"WanPipeline load failed: {exc}")
            raise exc

        if self.model_config.device_map != "auto":
            self.pipe.to(self.device)

        self._loaded = True
        logger.info("Wan pipeline loaded successfully.")

    def generate(self, config: GenerationConfig) -> Any:
        if not self._loaded:
            raise RuntimeError("Adapter not loaded. Call load() first.")

        kwargs = {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "num_frames": config.num_frames,
            "width": config.width,
            "height": config.height,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            **config.extra_args,
        }
        if config.seed is not None:
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(config.seed)

        logger.info(
            f"Generating video: {config.width}x{config.height}, "
            f"frames={config.num_frames}, steps={config.num_inference_steps}"
        )

        result = self.pipe(**kwargs)
        # diffusers returns an object with .frames[0]
        frames = result.frames[0] if hasattr(result, "frames") else result
        return frames

    def generate_batch(self, configs: List[GenerationConfig]) -> List[Any]:
        # Wan pipeline does not natively support batched prompts with different
        # resolutions / frame counts. Fall back to sequential generation.
        results = []
        for cfg in configs:
            results.append(self.generate(cfg))
        return results

    def build_pipeline(self) -> BasePipeline:
        return BasePipeline(
            units=[
                PromptEmbedUnit(),
                NoiseInitUnit(),
                DiTDenoiseUnit(),
                VAEDecodeUnit(),
            ]
        )

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info("Wan pipeline unloaded and CUDA cache cleared.")
