"""OpenAI-compatible FastAPI server for online video generation.

Endpoints:
  POST /v1/generation/text2video
  POST /v1/generation/image2video
  GET  /v1/query/video/{task_id}
  GET  /v1/models
  GET  /health
"""

import io
import logging
import os
import tempfile
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..beans import (
    Text2VideoRequest,
    Image2VideoRequest,
    GeneralResponse,
    GenerationData,
    GenerationResponse,
    ModelsData,
    ModelsResponse,
)
from ..core.config import GenerationConfig, ModelConfig
from ..core.engine import VideoGenEngineCore
from ..constants import (
    VideoGenerationStatus,
    HttpResponseCodes,
)
from ..utils import (
    save_video,
    TimeUtil,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global engine store (simple in-memory; can be replaced with Redis later)
# ---------------------------------------------------------------------------
_engines: dict[str, VideoGenEngineCore] = {}
_tasks: dict[str, dict[str, any]] = {}


def _get_engine(model_id: str) -> VideoGenEngineCore:
    if model_id not in _engines:
        cfg = ModelConfig(model_id=model_id)
        _engines[model_id] = VideoGenEngineCore(cfg)
        _engines[model_id].load()
    return _engines[model_id]


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

def _save_to_temp(frames: any, fps: int = 16) -> str:
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
@app.get("/health", response_model=GeneralResponse, response_model_exclude_none=True)
async def health():
    timestamp = TimeUtil.get_current_time_in_YMDHMS()

    return GeneralResponse(
        code=HttpResponseCodes.SUCCESS,
        message="success",
        timestamp=timestamp,
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_available_models():
    """Return only models currently loaded in GPU memory."""
    return ModelsResponse(
        code=HttpResponseCodes.SUCCESS,
        message="success",
        data=ModelsData(models=list(_engines.keys())),
    )


@app.post("/v1/generation/text2video", response_model=GenerationResponse, response_model_exclude_none=True)
async def text2video(req: Text2VideoRequest):
    task_id = uuid.uuid4().hex
    timestamp = TimeUtil.get_current_time_in_YMDHMS()
    _tasks[task_id] = {"status": VideoGenerationStatus.GENERATING.value, "start_time": timestamp}

    try:
        engine = _get_engine(req.model)
        config = _request_to_config(req)
        frames = engine.generate(config)
        path = _save_to_temp(frames, fps=req.fps)
        _tasks[task_id].update({"status": VideoGenerationStatus.COMPLETED.value, "path": path})
        
        gd = GenerationData(
            task_id=task_id,
            status=VideoGenerationStatus.COMPLETED.value,
            video_url=f"/v1/download/video/{task_id}",
            message="Generation succeeded.",
        )

        return GenerationResponse(code=HttpResponseCodes.SUCCESS, message="success", data=gd)
    except Exception as exc:
        logger.exception("text2video generation failed")
        _tasks[task_id]["status"] = VideoGenerationStatus.FAILED.value
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/generation/image2video", response_model=GenerationResponse)
async def image2video(req: Image2VideoRequest):
    task_id = uuid.uuid4().hex
    timestamp = TimeUtil.get_current_time_in_YMDHMS()
    _tasks[task_id] = {"status": VideoGenerationStatus.GENERATING.value, "start_time": timestamp}

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
        
        gd = GenerationData(
            task_id=task_id,
            status=VideoGenerationStatus.COMPLETED.value,
            video_url=f"/v1/download/video/{task_id}",
            message="Generation succeeded.",
        )

        return GenerationResponse(
            code=HttpResponseCodes.SUCCESS,
            message="success",
            data=gd,
        )
    except Exception as exc:
        logger.exception("image2video generation failed")
        _tasks[task_id]["status"] = VideoGenerationStatus.FAILED.value
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


@app.get("/v1/download/video/{task_id}", response_model_exclude_none=True)
async def download_video(task_id: str):
    task = _tasks.get(task_id)

    if task is None:
        return GeneralResponse(
            code=HttpResponseCodes.FAILURE,
            message=f"Task {task_id} not found",
        )
    
    task_status = task.get("status")
    
    if task_status == VideoGenerationStatus.QUEUING.value:
        return GeneralResponse(
            code=HttpResponseCodes.SUCCESS,
            message=f"Task {task_id} is queuing",
        )
    elif task_status == VideoGenerationStatus.GENERATING.value:
        return GeneralResponse(
            code=HttpResponseCodes.SUCCESS,
            message=f"Task {task_id} is in progress",
        )
    elif task_status == VideoGenerationStatus.FAILED.value:
        return GeneralResponse(
            code=HttpResponseCodes.FAILURE,
            message=f"Task {task_id} failed",
        )
    
    video_path = task.get("path")
    
    if (not video_path) or (not os.path.exists(video_path)):
        return GeneralResponse(
            code=HttpResponseCodes.FAILURE,
            message=task.get("error_msg", "视频路径不存在"), 
        )
    
    file_name = f"dreamer_{task_id}.mp4"

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=file_name,
        headers={"Content-Disposition": f"attachment; filename={file_name}"}
    )
