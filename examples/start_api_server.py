"""Example: Start the OpenAI-compatible API server programmatically."""

import uvicorn

from dreamer.entrypoints.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
