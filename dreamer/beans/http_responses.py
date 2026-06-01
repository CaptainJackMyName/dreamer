from pydantic import BaseModel


class GeneralResponse(BaseModel):
    code: int
    message: str
    data: dict | None = None
    timestamp: str | None = None

class GenerationData(BaseModel):
    task_id: str
    status: str  # queuing | generating | completed | failed
    video_url: str | None = None
    message: str | None = None

class GenerationResponse(BaseModel):
    code: int
    message: str
    data: GenerationData

class ModelsData(BaseModel):
    models: list[str]

class ModelsResponse(BaseModel):
    code: int
    message: str
    data: ModelsData