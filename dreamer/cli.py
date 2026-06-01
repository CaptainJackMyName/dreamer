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


def _resolve_model_arg(model_arg: str):
    """Resolve a user-provided model argument to a registered model_id and optional repo_id/path.

    Supports:
      - Local directory path  -> (basename, path)
      - Registered model_id   -> (model_id, None)
      - Known repo_id         -> (model_id, repo_id)
      - HuggingFace repo_id   -> heuristic extraction of model_id from last path segment
    """
    import os

    from dreamer.core.registry import _MODEL_REGISTRY

    # 1) Local directory
    if os.path.isdir(model_arg):
        model_id = os.path.basename(model_arg.rstrip(os.sep))
        return model_id, model_arg

    # 2) Exact registered model_id
    if model_arg in _MODEL_REGISTRY:
        return model_arg, None

    # 3) Reverse lookup via adapter DEFAULT_REPOS
    for mid, adapter_cls in _MODEL_REGISTRY.items():
        default_repos = getattr(adapter_cls, "DEFAULT_REPOS", {})
        if model_arg in default_repos.values():
            return mid, model_arg

    # 4) Heuristic: repo_id like "org/name" -> try "name" as model_id
    if "/" in model_arg:
        guessed = model_arg.split("/")[-1]
        if guessed in _MODEL_REGISTRY:
            return guessed, model_arg

    registered = ", ".join(_MODEL_REGISTRY.keys())
    raise ValueError(
        f"Cannot resolve model argument '{model_arg}'. "
        + f"It is not a local path, a registered model_id, or a known repo_id. "
        + f"Registered models: {registered}"
    )


def cmd_serve(args):
    import uvicorn

    from dreamer.core.config import ModelConfig
    from dreamer.core.engine import VideoGenEngineCore
    from dreamer.entrypoints.api import app, _engines

    logger = logging.getLogger(__name__)

    model_id, repo_id = _resolve_model_arg(args.model)

    extra = {}
    if repo_id:
        extra["repo_id"] = repo_id

    cfg = ModelConfig(
        model_id=model_id,
        precision=args.precision,
        device_map="cuda",
        gpu_memory_utilization=args.gpu_memory_utilization,
        extra_args=extra,
    )

    logger.info("Pre-loading model '%s' before starting server ...", model_id)
    engine = VideoGenEngineCore(cfg)
    engine.load()
    _engines[model_id] = engine
    logger.info("Model '%s' loaded. Starting API server on %s:%d", model_id, args.host, args.port)

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
    serve_parser.add_argument(
        "--model",
        required=True,
        help="Model identifier, HuggingFace repo_id, or local directory path",
    )
    serve_parser.add_argument(
        "--precision", default="bf16", help="Precision (fp16/bf16/fp32)"
    )
    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )

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
