"""
OpenEnv — Multi-Domain AI Testing Environment
FastAPI server implementing the full OpenEnv spec across 4 domains.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import os, sys, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import Action, Observation, Reward, AgentAction
from tasks.task1_classification import ClassificationTask
from tasks.task2_drafting import ResponseDraftingTask
from tasks.task3_queue import QueueManagementTask
from tasks.task4_multiturn import MultiTurnTask
from tasks.legal_tasks import LegalIdentifyClauseTask, LegalRiskFlagTask, LegalRedlineTask
from tasks.clinical_tasks import ClinicalTriageTask, ClinicalESITask, ClinicalNoteTask
from tasks.pr_tasks import PRTypeTask, PRBugIdentifyTask, PRReviewTask

# ─── Task Registry ────────────────────────────────────────────────────────────

TASKS = {
    "ticket_classification":        ClassificationTask,
    "response_drafting":            ResponseDraftingTask,
    "queue_management":             QueueManagementTask,
    "multi_turn_conversation":      MultiTurnTask,
    "legal_clause_identification":  LegalIdentifyClauseTask,
    "legal_risk_flagging":          LegalRiskFlagTask,
    "legal_clause_redlining":       LegalRedlineTask,
    "clinical_triage_classification": ClinicalTriageTask,
    "clinical_esi_assignment":      ClinicalESITask,
    "clinical_triage_note":         ClinicalNoteTask,
    "pr_type_classification":       PRTypeTask,
    "pr_bug_identification":        PRBugIdentifyTask,
    "pr_review_comment":            PRReviewTask,
}

_active: Dict[str, Any] = {}
_current_task_id: str = "ticket_classification"


def get_task(task_id: str = None):
    tid = task_id or _current_task_id
    if tid not in _active:
        raise HTTPException(
            status_code=400,
            detail=f"No active episode for task '{tid}'. Call /reset first."
        )
    return _active[tid]


# ─── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    for tid, TaskClass in TASKS.items():
        instance = TaskClass()
        instance.reset()
        _active[tid] = instance
    yield


app = FastAPI(
    title="OpenEnv — Multi-Domain AI Testing Environment",
    description=(
        "Real-world OpenEnv environment testing AI agents across "
        "Customer Support, Legal Review, Clinical Triage, and Software Engineering."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Core OpenEnv Endpoints ───────────────────────────────────────────────────

@app.post("/reset")
async def reset(task_id: Optional[str] = None) -> Dict[str, Any]:
    global _current_task_id
    tid = task_id or _current_task_id
    if tid not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{tid}'. Valid: {list(TASKS.keys())}"
        )
    _current_task_id = tid
    instance = TASKS[tid]()
    _active[tid] = instance
    obs = instance.reset()
    return {"task_id": tid, "observation": obs.model_dump()}


@app.post("/step")
async def step(action: Action, task_id: Optional[str] = None) -> Dict[str, Any]:
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
    task = get_task(task_id)
    return task.state()


# ─── Extended Endpoints ───────────────────────────────────────────────────────

@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            # ── Customer Support ──
            {
                "id": "ticket_classification",
                "name": "Ticket Classification & Routing",
                "difficulty": "easy",
                "description": "Classify incoming support tickets by category and priority.",
                "action_schema": {
                    "action_type": "classify",
                    "ticket_id": "string",
                    "category": "billing|technical|account|feature_request|abuse|unknown",
                    "priority": "P1|P2|P3|P4",
                },
            },
            {
                "id": "response_drafting",
                "name": "Response Drafting & Quality",
                "difficulty": "medium",
                "description": "Draft professional customer-facing responses using KB context.",
                "action_schema": {
                    "action_type": "draft_response",
                    "ticket_id": "string",
                    "response_text": "string (100-300 words)",
                },
            },
            {
                "id": "queue_management",
                "name": "SLA Queue Management",
                "difficulty": "hard",
                "description": "Manage 20 tickets across 3 agents. Maximize SLA compliance and FCR.",
                "action_schema": {
                    "action_type": "assign_ticket|resolve|escalate|close|no_op",
                    "ticket_id": "string",
                    "target_agent_id": "agent_billing|agent_tech|agent_general",
                },
            },
            {
                "id": "multi_turn_conversation",
                "name": "Multi-Turn De-escalation",
                "difficulty": "very_hard",
                "description": "Handle a multi-turn frustrated customer conversation requiring context retention.",
                "action_schema": {
                    "action_type": "draft_response|escalate|resolve|close",
                    "response_text": "string",
                },
            },
            # ── Legal ──
            {
                "id": "legal_clause_identification",
                "name": "Legal Clause Identification",
                "difficulty": "easy",
                "description": "Identify the type of each contractual clause.",
                "action_schema": {
                    "action_type": "identify_clause",
                    "clause_type": "indemnity|liability|ip|termination|unknown",
                },
            },
            {
                "id": "legal_risk_flagging",
                "name": "Legal Risk Flagging",
                "difficulty": "medium",
                "description": "Assess and flag risk levels for contractual clauses.",
                "action_schema": {
                    "action_type": "flag_risk",
                    "risk_level": "low|medium|high|critical",
                    "reasoning": "string",
                },
            },
            {
                "id": "legal_clause_redlining",
                "name": "Legal Clause Redlining",
                "difficulty": "hard",
                "description": "Rewrite risky contract clauses with safer alternative language.",
                "action_schema": {
                    "action_type": "redline",
                    "redline_text": "string (improved clause text)",
                },
            },
            # ── Clinical ──
            {
                "id": "clinical_triage_classification",
                "name": "Clinical Triage Body System",
                "difficulty": "easy",
                "description": "Classify patient chief complaints into affected body systems.",
                "action_schema": {
                    "action_type": "classify_triage",
                    "body_system": "cardiac|respiratory|neurologic|gi|musculoskeletal|other",
                },
            },
            {
                "id": "clinical_esi_assignment",
                "name": "Clinical ESI Assignment",
                "difficulty": "medium",
                "description": "Assign Emergency Severity Index (ESI) 1-5 triage levels.",
                "action_schema": {
                    "action_type": "assign_esi",
                    "esi_level": "integer 1-5",
                    "reasoning": "string",
                },
            },
            {
                "id": "clinical_triage_note",
                "name": "Clinical Triage Note Generation",
                "difficulty": "hard",
                "description": "Write a complete clinical triage note for ED patients.",
                "action_schema": {
                    "action_type": "write_triage_note",
                    "triage_note": "string (clinical note)",
                },
            },
            # ── Engineering ──
            {
                "id": "pr_type_classification",
                "name": "PR Type Classification",
                "difficulty": "easy",
                "description": "Classify pull request diffs by type.",
                "action_schema": {
                    "action_type": "classify_pr",
                    "pr_type": "bug_fix|feature|refactor|security",
                },
            },
            {
                "id": "pr_bug_identification",
                "name": "PR Bug Identification",
                "difficulty": "medium",
                "description": "Identify the specific bug or security issue in a code diff.",
                "action_schema": {
                    "action_type": "identify_bug",
                    "bug_description": "string",
                },
            },
            {
                "id": "pr_review_comment",
                "name": "PR Code Review",
                "difficulty": "hard",
                "description": "Write expert code review comments approving or blocking PRs.",
                "action_schema": {
                    "action_type": "review_pr",
                    "review_comment": "string",
                },
            },
        ]
    }


@app.post("/grader")
async def grader(task_id: Optional[str] = None) -> Dict[str, Any]:
    task = get_task(task_id)
    result = task.grader_score()

    # Global reward shaping bonuses
    reward_history = getattr(task, "reward_history", [])
    if reward_history and len(reward_history) >= 3:
        first_half = sum(reward_history[:len(reward_history) // 2])
        second_half = sum(reward_history[len(reward_history) // 2:])
        traj_bonus = 0.05 if second_half > first_half else 0.0
        try:
            std = statistics.stdev(reward_history)
            cons_penalty = -0.05 if std > 0.4 else 0.0
        except Exception:
            cons_penalty = 0.0
        result["final_score"] = max(0.0, min(1.0, result["final_score"] + traj_bonus + cons_penalty))
        result.setdefault("metrics", {})
        result["metrics"]["trajectory_bonus"] = traj_bonus
        result["metrics"]["consistency_penalty"] = cons_penalty

    return result


@app.post("/baseline")
async def baseline() -> Dict[str, Any]:
    """Run heuristic baseline agent across all tasks. No API key needed."""
    from graders.baseline_agent import run_baseline_all_tasks
    return await run_baseline_all_tasks()


@app.get("/health")
async def health():
    return {"status": "ok", "tasks": list(TASKS.keys())}


@app.get("/", response_class=FileResponse)
async def root():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_path = os.path.join(root_dir, "static", "index.html")
    return FileResponse(static_path)
