"""Command-line interface for Dreamer."""

import argparse
import logging
import sys


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_generate(args):
    from dreamer import VideoGenLLM, GenerationConfig

    engine = VideoGenLLM(
        model=args.model,
        precision=args.precision,
        device_map=args.device_map,
    )

    config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
    )

    video = engine.generate(config)
    engine.save_video(video, args.output, fps=args.fps)
    print(f"Video saved to: {args.output}")


def cmd_serve(args):
    import uvicorn

    from dreamer.entrypoints.api import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


def cmd_list_models(args):
    from dreamer.core.registry import list_models

    models = list_models()
    print("Registered models:")
    for m in models:
        print(f"  - {m}")


def main():
    parser = argparse.ArgumentParser(
        prog="dreamer",
        description="Dreamer: High-performance video generation engine",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    gen_parser = subparsers.add_parser("generate", help="Generate a video from text")
    gen_parser.add_argument("--model", default="wan-2.2-t2v-a14b", help="Model identifier")
    gen_parser.add_argument("--prompt", required=True, help="Text prompt")
    gen_parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    gen_parser.add_argument("--num-frames", type=int, default=81, help="Number of frames")
    gen_parser.add_argument("--width", type=int, default=1280, help="Video width")
    gen_parser.add_argument("--height", type=int, default=720, help="Video height")
    gen_parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    gen_parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale")
    gen_parser.add_argument("--fps", type=int, default=16, help="Output FPS")
    gen_parser.add_argument("--precision", default="bf16", help="Precision (fp16/bf16/fp32)")
    gen_parser.add_argument("--device-map", default="auto", help="Device map strategy")
    gen_parser.add_argument("--output", default="output.mp4", help="Output file path")

    # ------------------------------------------------------------------
    # serve
    # ------------------------------------------------------------------
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")

    # ------------------------------------------------------------------
    # list-models
    # ------------------------------------------------------------------
    subparsers.add_parser("list-models", help="List registered models")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
