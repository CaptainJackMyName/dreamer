"""OpenAI-compatible FastAPI server for online video generation.

Endpoints:
  POST /v1/generation/text2video
  POST /v1/generation/image2video
  GET  /v1/query/video/{task_id}
  GET  /v1/models
  GET  /health
"""

import base64
import io
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..core.config import GenerationConfig, ModelConfig
from ..core.engine import VideoGenEngineCore
from ..core.registry import list_models
from ..utils.video import save_video

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global engine store (simple in-memory; can be replaced with Redis later)
# ---------------------------------------------------------------------------
_engines: Dict[str, VideoGenEngineCore] = {}
_tasks: Dict[str, Dict[str, Any]] = {}


def _get_engine(model_id: str) -> VideoGenEngineCore:
    if model_id not in _engines:
        cfg = ModelConfig(model_id=model_id)
        _engines[model_id] = VideoGenEngineCore(cfg)
        _engines[model_id].load()
    return _engines[model_id]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class Text2VideoRequest(BaseModel):
    model: str = "wan-2.2-t2v-a14b"
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 81
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 50
    seed: Optional[int] = None
    guidance_scale: float = 5.0
    fps: int = 16
    response_format: str = "url"  # url | base64


class Image2VideoRequest(BaseModel):
    model: str = "wan-2.2-t2v-a14b"
    image: str  # base64 encoded image
    prompt: str = ""
    num_frames: int = 81
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 50
    seed: Optional[int] = None
    guidance_scale: float = 5.0
    fps: int = 16
    response_format: str = "url"


class GenerationResponse(BaseModel):
    task_id: str
    status: str  # pending | running | completed | failed
    video_url: Optional[str] = None
    video_base64: Optional[str] = None
    message: str = ""


class ModelsResponse(BaseModel):
    models: List[str]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dreamer API server starting up...")
    yield
    logger.info("Dreamer API server shutting down...")
    for eng in _engines.values():
        eng.adapter.unload()


app = FastAPI(
    title="Dreamer Video Generation API",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _frames_to_base64(frames: Any, fps: int = 16) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        save_video(frames, tmp.name, fps=fps)
        with open(tmp.name, "rb") as f:
            data = f.read()
        os.unlink(tmp.name)
    return base64.b64encode(data).decode("utf-8")


def _save_to_temp(frames: Any, fps: int = 16) -> str:
    tmp_dir = tempfile.gettempdir()
    fname = f"dreamer_{uuid.uuid4().hex}.mp4"
    path = os.path.join(tmp_dir, fname)
    save_video(frames, path, fps=fps)
    return path


def _request_to_config(req: Text2VideoRequest) -> GenerationConfig:
    return GenerationConfig(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_frames=req.num_frames,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        fps=req.fps,
        response_format=req.response_format,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_available_models():
    return ModelsResponse(models=list_models())


@app.post("/v1/generation/text2video", response_model=GenerationResponse)
async def text2video(req: Text2VideoRequest):
    task_id = uuid.uuid4().hex
    _tasks[task_id] = {"status": "running", "start_time": time.time()}

    try:
        engine = _get_engine(req.model)
        config = _request_to_config(req)
        frames = engine.generate(config)

        if req.response_format == "base64":
            b64 = _frames_to_base64(frames, fps=req.fps)
            _tasks[task_id].update({"status": "completed", "result": b64})
            return GenerationResponse(
                task_id=task_id,
                status="completed",
                video_base64=b64,
                message="Generation succeeded.",
            )
        else:
            path = _save_to_temp(frames, fps=req.fps)
            _tasks[task_id].update({"status": "completed", "path": path})
            return GenerationResponse(
                task_id=task_id,
                status="completed",
                video_url=f"/v1/download/video/{task_id}",
                message="Generation succeeded.",
            )
    except Exception as exc:
        logger.exception("text2video generation failed")
        _tasks[task_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/generation/image2video", response_model=GenerationResponse)
async def image2video(req: Image2VideoRequest):
    task_id = uuid.uuid4().hex
    _tasks[task_id] = {"status": "running", "start_time": time.time()}

    try:
        import base64
        from PIL import Image

        image_bytes = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        engine = _get_engine(req.model)
        config = GenerationConfig(
            prompt=req.prompt,
            num_frames=req.num_frames,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
            guidance_scale=req.guidance_scale,
            fps=req.fps,
            image=image,
        )
        frames = engine.generate(config)

        if req.response_format == "base64":
            b64 = _frames_to_base64(frames, fps=req.fps)
            _tasks[task_id].update({"status": "completed", "result": b64})
            return GenerationResponse(
                task_id=task_id,
                status="completed",
                video_base64=b64,
                message="Generation succeeded.",
            )
        else:
            path = _save_to_temp(frames, fps=req.fps)
            _tasks[task_id].update({"status": "completed", "path": path})
            return GenerationResponse(
                task_id=task_id,
                status="completed",
                video_url=f"/v1/download/video/{task_id}",
                message="Generation succeeded.",
            )
    except Exception as exc:
        logger.exception("image2video generation failed")
        _tasks[task_id]["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/query/video/{task_id}")
async def query_task(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "status": task["status"],
        "elapsed": time.time() - task.get("start_time", time.time()),
    }


@app.get("/v1/download/video/{task_id}")
async def download_video(task_id: str):
    task = _tasks.get(task_id)
    if not task or task.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Video not ready")
    path = task.get("path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video file missing")
    return FileResponse(path, media_type="video/mp4")
