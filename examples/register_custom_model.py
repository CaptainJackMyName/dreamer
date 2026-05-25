"""Example: How to register a new model adapter (extensibility demo).

Suppose you want to add support for a hypothetical "MyVideoModel".
You only need to:
  1. Subclass BaseModelAdapter.
  2. Implement load(), generate(), generate_batch(), build_pipeline().
  3. Decorate with @register_model("your-model-id").
"""

from typing import List, Any

from dreamer.core.config import ModelConfig, GenerationConfig
from dreamer.core.registry import register_model
from dreamer.core.pipeline import BasePipeline, PipelineContext, PipelineUnit
from dreamer.models.adapters import BaseModelAdapter


class MyCustomDenoiseUnit(PipelineUnit):
    name = "my_denoise"

    def run(self, context: PipelineContext) -> PipelineContext:
        print("Running custom denoise step...")
        context.set("latents", "fake_latents")
        return context


@register_model("my-custom-model")
class MyCustomAdapter(BaseModelAdapter):
    """Template adapter showing the minimal contract required."""

    def load(self):
        print("MyCustomAdapter.load() called")
        self._loaded = True

    def generate(self, config: GenerationConfig) -> Any:
        return f"fake_video_for_prompt_{config.prompt}"

    def generate_batch(self, configs: List[GenerationConfig]) -> List[Any]:
        return [self.generate(c) for c in configs]

    def build_pipeline(self) -> BasePipeline:
        return BasePipeline(units=[MyCustomDenoiseUnit()])


def main():
    from dreamer import VideoGenLLM
    from dreamer.core.registry import list_models

    print("Registered models:", list_models())

    engine = VideoGenLLM(model="my-custom-model")
    cfg = GenerationConfig(prompt="hello world")
    result = engine.generate(cfg)
    print("Result:", result)


if __name__ == "__main__":
    main()
