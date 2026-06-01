from pydantic import BaseModel


class Text2VideoRequest(BaseModel):
    model: str = "Wan2.2-TI2V-5B-Diffusers"
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 81
    width: int = 1280
    height: int = 704
    num_inference_steps: int = 50
    seed: int | None = None
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
    seed: int | None = None
    guidance_scale: float = 5.0
    fps: int = 16
