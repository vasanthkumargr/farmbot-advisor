"""
FastAPI application for the Farmbot Advisor Environment.

This module creates an HTTP server that exposes the FarmbotEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Or run directly:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: uv sync"
    ) from e

try:
    from models import FarmbotAction, FarmbotObservation
except ImportError:
    from ..models import FarmbotAction, FarmbotObservation

from .farmbot_environment import FarmbotEnvironment


# Create the app with web interface and README integration
app = create_app(
    FarmbotEnvironment,
    FarmbotAction,
    FarmbotObservation,
    env_name="farmbot_advisor",
    max_concurrent_envs=4,
)


# Root endpoint — required so validator ping to GET / returns 200 (not 404)
@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "farmbot-advisor",
        "tasks": ["irrigation_decision", "fertilizer_recommendation", "harvest_timing"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks"]
    }


# Tasks listing endpoint
@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "irrigation_decision", "difficulty": "easy"},
            {"id": "fertilizer_recommendation", "difficulty": "medium"},
            {"id": "harvest_timing", "difficulty": "hard"},
        ]
    }


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)