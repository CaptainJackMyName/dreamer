from enum import Enum

class VideoGenerationStatus(Enum):
    QUEUING = "queuing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"