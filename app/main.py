"""
Customer Support Triage — OpenEnv Environment
FastAPI server implementing the full OpenEnv spec.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import Action, Observation, Reward, AgentAction
from tasks.task1_classification import ClassificationTask
from tasks.task2_drafting import ResponseDraftingTask
from tasks.task3_queue import QueueManagementTask


# ─── State ────────────────────────────────────────────────────────────────────

TASKS = {
    "ticket_classification": ClassificationTask,
    "response_drafting": ResponseDraftingTask,
    "queue_management": QueueManagementTask,
}

# Active task instances (per-session; single-process demo)
_active: Dict[str, Any] = {}
_current_task_id: str = "ticket_classification"


def get_task(task_id: str = None):
    tid = task_id or _current_task_id
    if tid not in _active:
        raise HTTPException(status_code=400, detail=f"No active episode for task '{tid}'. Call /reset first.")
    return _active[tid]


# ─── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm all tasks
    for tid, TaskClass in TASKS.items():
        instance = TaskClass()
        instance.reset()
        _active[tid] = instance
    yield


app = FastAPI(
    title="Customer Support Triage — OpenEnv",
    description=(
        "A real-world OpenEnv environment where AI agents learn to triage, "
        "categorize, prioritize, and resolve customer support tickets."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the dashboard UI
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ─── OpenEnv Core Endpoints ───────────────────────────────────────────────────

@app.post("/reset")
async def reset(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Reset the environment and return the initial observation."""
    global _current_task_id
    tid = task_id or _current_task_id
    if tid not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{tid}'. Valid: {list(TASKS.keys())}")

    _current_task_id = tid
    instance = TASKS[tid]()
    _active[tid] = instance
    obs = instance.reset()
    return {"task_id": tid, "observation": obs.model_dump()}


@app.post("/step")
async def step(action: Action, task_id: Optional[str] = None) -> Dict[str, Any]:
    """Take an action and return (observation, reward, done, info)."""
    task = get_task(task_id)
    try:
        obs, reward, done, info = task.step(action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Step failed: {str(e)}")

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the current environment state without advancing."""
    task = get_task(task_id)
    return task.state()


# ─── Extended Endpoints ───────────────────────────────────────────────────────

@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List all tasks with their action schemas."""
    return {
        "tasks": [
            {
                "id": "ticket_classification",
                "name": "Ticket Classification & Routing",
                "difficulty": "easy",
                "description": "Classify incoming tickets by category and priority.",
                "max_steps": 10,
                "action_schema": {
                    "action_type": "classify",
                    "ticket_id": "string (from observation.current_ticket.ticket_id)",
                    "category": "one of: billing | technical | account | feature_request | abuse | unknown",
                    "priority": "one of: P1 | P2 | P3 | P4",
                    "reasoning": "string (optional)",
                },
            },
            {
                "id": "response_drafting",
                "name": "Response Drafting & Quality",
                "difficulty": "medium",
                "description": "Draft professional customer-facing responses using KB context.",
                "max_steps": 6,
                "action_schema": {
                    "action_type": "draft_response",
                    "ticket_id": "string",
                    "response_text": "string (full customer-facing response, 50-400 words)",
                    "reasoning": "string (optional)",
                },
            },
            {
                "id": "queue_management",
                "name": "SLA Queue Management",
                "difficulty": "hard",
                "description": "Manage 20 tickets across 3 agents. Maximize SLA compliance and FCR.",
                "max_steps": 40,
                "action_schema": {
                    "action_type": "assign_ticket | escalate | resolve | close | no_op",
                    "ticket_id": "string (required for assign/escalate/resolve/close)",
                    "target_agent_id": "agent_billing | agent_tech | agent_general (required for assign/escalate)",
                    "resolution_summary": "string (optional for resolve)",
                    "reasoning": "string (optional)",
                },
            },
        ]
    }


@app.post("/grader")
async def grader(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Return grader score for the current or completed episode."""
    task = get_task(task_id)
    return task.grader_score()


@app.post("/baseline")
async def baseline() -> Dict[str, Any]:
    """
    Run the heuristic baseline agent against all 3 tasks and return scores.
    This endpoint replicates what baseline_inference.py does server-side.
    """
    from graders.baseline_agent import run_baseline_all_tasks
    results = await run_baseline_all_tasks()
    return results


@app.get("/health")
async def health():
    return {"status": "ok", "tasks": list(TASKS.keys())}


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Customer Support Triage — OpenEnv</h1><p>See /docs for API.</p>")
