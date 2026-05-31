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
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..core.config import GenerationConfig, ModelConfig
from ..core.engine import VideoGenEngineCore
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
    model: str = "Wan2.2-TI2V-5B-Diffusers"
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 81
    width: int = 1280
    height: int = 704
    num_inference_steps: int = 50
    seed: Optional[int] = None
    guidance_scale: float = 5.0
    fps: int = 16


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
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    cur_time = datetime.now()
    timestamp = cur_time.strftime("%Y-%m-%d %H:%M:%S")
    return {"status": "ok", "timestamp": timestamp}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_available_models():
    """Return only models currently loaded in GPU memory."""
    return ModelsResponse(models=list(_engines.keys()))


@app.post("/v1/generation/text2video", response_model=GenerationResponse)
async def text2video(req: Text2VideoRequest):
    task_id = uuid.uuid4().hex
    _tasks[task_id] = {"status": "running", "start_time": time.time()}

    try:
        engine = _get_engine(req.model)
        config = _request_to_config(req)
        frames = engine.generate(config)
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
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": task.get("error_msg", "视频生成失败")}
        )
    path = task.get("path")
    if not path or not os.path.exists(path):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": task.get("error_msg", "视频路径不存在")}
        )
    file_name = f"dreamer_{task_id}.mp4"

    return FileResponse(
        path, 
        media_type="video/mp4",
        filename=file_name,
        headers={"Content-Disposition": f"attachment; filename={file_name}"}
    )
