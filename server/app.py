"""
server/app.py — OpenEnv required server entry point.
Satisfies: openenv validate check for server/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: F401 — re-export the FastAPI app


def start():
    """
    Entry point for [project.scripts] in pyproject.toml.
    Called when running: uv run server (or python -m server.app)
    """
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    start()