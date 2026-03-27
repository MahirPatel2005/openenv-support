---
title: Customer Support Triage — OpenEnv
emoji: 🎫
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
  - agent-environment
short_description: OpenEnv environment for customer support triage agents
---

# Customer Support Triage — OpenEnv Environment

> A real-world OpenEnv environment where AI agents learn to **triage, classify, prioritize, and resolve** customer support tickets across a SaaS platform.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-6c63ff)](https://openenv.dev)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Spaces-openenv--support-yellow)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)

---

## 🎯 Environment Description

Customer support triage is a high-stakes, real-world task that every software company faces. Agents must:

- **Classify** incoming tickets accurately by category and urgency
- **Draft** professional, factually grounded customer responses using a knowledge base
- **Manage** a live queue of tickets across specialized agents under SLA constraints

This environment models realistic support scenarios including billing disputes, API outages, SSO failures, account transfers, feature requests, and abuse reports — with varying customer tiers, sentiment signals, and time pressure.

---

## 📋 Tasks

| Task | ID | Difficulty | Max Steps | Pass Threshold |
|------|----|-----------|-----------|----------------|
| Ticket Classification | `ticket_classification` | 🟢 Easy | 10 | 0.70 |
| Response Drafting | `response_drafting` | 🟡 Medium | 6 | 0.60 |
| SLA Queue Management | `queue_management` | 🔴 Hard | 40 | 0.50 |

### Task 1: Ticket Classification (Easy)
Classify 10 incoming tickets by **category** (billing/technical/account/feature_request/abuse/unknown) and **priority** (P1-P4). Scores 0.6 for correct category, 0.4 for correct priority (with partial credit for adjacent priority levels).

### Task 2: Response Drafting (Medium)
Draft complete customer-facing responses to pre-classified tickets. Graded on 5 dimensions using a deterministic rubric:
- KB reference (25%) — uses relevant knowledge base information
- Issue addressed (30%) — directly addresses the customer's specific problem
- Actionable steps (20%) — provides concrete next steps
- Tone & empathy (15%) — professional, empathetic language
- No hallucination (10%) — no false promises or invented information

### Task 3: SLA Queue Management (Hard)
Manage a queue of 20 mixed tickets across 3 specialized agents. Actions: `assign_ticket`, `escalate`, `resolve`, `close`, `no_op`. Score based on:
- Resolution rate (35%)
- SLA compliance (35%)
- First-contact resolution rate (20%)
- Escalation penalty (-20% if excessive)

---

## 🔌 Action & Observation Spaces

### Observation
```python
class Observation(BaseModel):
    task_id: str
    step: int
    current_ticket: Optional[Ticket]       # For tasks 1 & 2
    ticket_queue: List[Ticket]             # For task 3
    agents: List[AgentInfo]                # For task 3
    knowledge_base: List[KBArticle]        # For task 2
    sla_status: Dict[str, str]             # ticket_id -> ok/warning/breached
    valid_actions: List[str]
    episode_done: bool
    info: Dict[str, Any]
```

### Action
```python
class Action(BaseModel):
    action_type: AgentAction               # classify | draft_response | assign_ticket | escalate | resolve | close | no_op
    ticket_id: Optional[str]
    category: Optional[TicketCategory]     # For classify
    priority: Optional[TicketPriority]     # For classify
    response_text: Optional[str]           # For draft_response
    target_agent_id: Optional[str]         # For assign_ticket / escalate
    resolution_summary: Optional[str]      # For resolve
    reasoning: Optional[str]              # Optional transparency field
```

### Reward
```python
class Reward(BaseModel):
    total: float                           # -1.0 to 1.0
    classification_accuracy: float
    response_quality: float
    sla_compliance: float
    first_contact_resolution: float
    customer_satisfaction: float
    penalty: float                         # Negative component
    breakdown: Dict[str, float]            # Per-dimension scores
```

---

## 🚀 Setup & Usage

### Local Development

```bash
git clone https://huggingface.co/spaces/your-username/openenv-support
cd openenv-support

pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

Visit `http://localhost:7860` for the interactive dashboard.
Visit `http://localhost:7860/docs` for the Swagger API.

### Docker

```bash
docker build -t openenv-support .
docker run -p 7860:7860 openenv-support
```

### Running the Baseline

```bash
export OPENAI_API_KEY="sk-..."
export OPENENV_BASE_URL="http://localhost:7860"

python baseline_inference.py               # All tasks
python baseline_inference.py --task ticket_classification
python baseline_inference.py --model gpt-4o
```

### OpenEnv Validate

```bash
openenv validate --url http://localhost:7860
```

---

## 📊 Baseline Scores

| Agent | Classification | Drafting | Queue Mgmt | Overall |
|-------|---------------|---------|------------|---------|
| Heuristic (keyword-based) | 0.72 | 0.61 | 0.54 | 0.62 |
| GPT-4o-mini | ~0.85 | ~0.74 | ~0.61 | ~0.73 |

> Scores are deterministic given the same random seed. Run `/baseline` endpoint to reproduce.

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset episode (query param: `task_id`) |
| `/step` | POST | Take action, get observation + reward |
| `/state` | GET | Current state without advancing |
| `/tasks` | GET | List tasks with action schemas |
| `/grader` | POST | Get episode grader score |
| `/baseline` | POST | Run heuristic baseline on all tasks |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

---

## 📁 Project Structure

```
openenv-support/
├── app/
│   ├── main.py              # FastAPI application
│   └── models.py            # Typed Pydantic models
├── tasks/
│   ├── task1_classification.py
│   ├── task2_drafting.py
│   └── task3_queue.py
├── graders/
│   └── baseline_agent.py    # Heuristic baseline
├── data/
│   ├── tickets.py           # Synthetic ticket dataset
│   └── knowledge_base.py    # KB articles
├── static/
│   └── index.html           # Interactive dashboard
├── openenv.yaml             # OpenEnv metadata
├── baseline_inference.py    # LLM baseline script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🏗️ Design Decisions

**Why customer support?** It's a universal, high-value real-world task. Every SaaS company needs it. The task has natural difficulty progression (classification → response quality → queue optimization under constraints) and rich partial reward signals at every step.

**Why deterministic graders?** The rubric uses keyword matching, heuristic NLP signals, and exact comparisons — no LLM-in-the-loop for grading, ensuring reproducible scores across runs.

**Reward shaping:** Every step returns a non-zero reward signal. Even incorrect actions return small penalties rather than zero, giving RL agents a gradient to follow rather than a sparse reward landscape.
