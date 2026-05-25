"""Example: Batch generation for efficient multi-video production."""

from dreamer import VideoGenLLM, GenerationConfig


def main():
    engine = VideoGenLLM(
        model="wan-2.2-t2v-a14b",
        precision="bf16",
        device_map="auto",
    )

    prompts = [
        "A serene lake reflecting cherry blossoms in spring",
        "A futuristic cityscape with flying cars at night",
        "A cozy cabin in a snowy forest with smoke from the chimney",
    ]

    configs = [
        GenerationConfig(
            prompt=p,
            num_frames=81,
            width=1280,
            height=720,
            num_inference_steps=50,
            seed=100 + i,
        )
        for i, p in enumerate(prompts)
    ]

    videos = engine.generate_batch(configs, batch_size=1)

    for i, video in enumerate(videos):
        path = f"batch_output_{i}.mp4"
        engine.save_video(video, path, fps=16)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
