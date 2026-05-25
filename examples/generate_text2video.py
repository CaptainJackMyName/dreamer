"""Quick-start example: Text-to-video generation with Wan2.2."""
from datetime import datetime
import os

from dreamer import VideoGenLLM, GenerationConfig


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    # 1. Initialize engine (downloads weights on first run)
    engine = VideoGenLLM(
        model="/data1/workspace/panfei/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",   # Registered model ID
        precision="bf16",           # Use bf16 for A100/H100; use fp16 for older GPUs
        device_map="cuda",          # Auto-dispatch to available GPU(s)
    )

    # 2. Configure generation
    config = GenerationConfig(
        prompt="A majestic eagle soaring over snow-capped mountains at sunrise",
        negative_prompt="blurry, low quality, distorted",
        num_frames=81,
        # width=1280,
        # height=704,
        width=960,
        height=512,
        num_inference_steps=50,
        guidance_scale=5.0,
        seed=42,
        fps=16,
    )

    # 3. Generate video
    video = engine.generate(config)

    # 4. Save to disk
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = f"wan_output_{timestamp}.mp4"
    engine.save_video(video, save_path, fps=16)
    print(f"Done! Saved to {save_path}")


if __name__ == "__main__":
    main()
