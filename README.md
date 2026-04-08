---
title: OpenEnv — Multi-Domain
emoji: 🌐
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - agents
  - nlp
  - multi-domain
short_description: OpenEnv multi-domain evaluation environment for AI agents
---

# OpenEnv — Multi-Domain AI Agent Environment

> A comprehensive OpenEnv environment where AI agents are evaluated across 13 complex real-world tasks spanning 4 professional domains: **Customer Support, Legal Review, Clinical Triage, and Software Engineering (PR Review)**.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-6c63ff)](https://openenv.dev)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Spaces-openenv--support-yellow)](https://huggingface.co/spaces)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)

---

## 🎯 Environment Description

To build general-purpose AI agents, we must evaluate them outside of traditional multiple-choice benchmarks. This environment provides 13 interactive, multi-step tasks across diverse verticals to test an agent's reasoning, tool use, and decision-making capabilities. 

Agents must classify data, identify risks, draft domain-specific responses, and manage queues under dynamic constraints.

---

## 📋 Tasks & Domains

This environment features 13 tasks across 4 distinct professional domains:

### 🎫 Customer Support
| Task | ID | Difficulty | Max Steps |
|------|----|-----------|-----------|
| Ticket Classification | `ticket_classification` | 🟢 Easy | 10 |
| Response Drafting | `response_drafting` | 🟡 Medium | 6 |
| SLA Queue Management | `queue_management` | 🔴 Hard | 40 |
| Multi-Turn De-escalation | `multi_turn_conversation` | 🟣 Very Hard | 8 |

### ⚖️ Legal Review
| Task | ID | Difficulty | Max Steps |
|------|----|-----------|-----------|
| Clause Identification | `legal_clause_identification` | 🟢 Easy | 10 |
| Risk Flagging | `legal_risk_flagging` | 🟡 Medium | 10 |
| Clause Redlining | `legal_clause_redlining` | 🔴 Hard | 5 |

### 🏥 Clinical Triage
| Task | ID | Difficulty | Max Steps |
|------|----|-----------|-----------|
| Body System Classification | `clinical_triage_classification` | 🟢 Easy | 10 |
| ESI Level Assignment | `clinical_esi_assignment` | 🟡 Medium | 10 |
| Triage Note Generation | `clinical_triage_note` | 🔴 Hard | 5 |

### 💻 PR Review (Software Engineering)
| Task | ID | Difficulty | Max Steps |
|------|----|-----------|-----------|
| PR Type Classification | `pr_type_classification` | 🟢 Easy | 10 |
| Bug Identification | `pr_bug_identification` | 🟡 Medium | 5 |
| Code Review Comment | `pr_review_comment` | 🔴 Hard | 5 |

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
export OPENENV_BASE_URL="http://localhost:7860"

# Evaluate heuristic baseline on all tasks
python inference.py              
```

### OpenEnv Validate

```bash
openenv validate --url http://localhost:7860
```

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
│   ├── task3_queue.py
│   ├── task4_multiturn.py
│   ├── legal_tasks.py
│   ├── clinical_tasks.py
│   └── pr_tasks.py
├── graders/
│   └── baseline_agent.py    # Auto-grader heuristics
├── data/
│   ├── tickets.py
│   ├── knowledge_base.py
│   ├── legal_data.py
│   ├── clinical_data.py
│   └── pr_data.py
├── static/
│   └── index.html           # Interactive multi-domain dashboard
├── openenv.yaml             # OpenEnv metadata
├── inference.py    # Automated test runner for baseline
├── validate.py              # Validation script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🏗️ Design Decisions

**Why multi-domain?** Real-world agents will need to operate across various specific verticals. Providing environments in Support, Legal, Clinical, and Engineering proves that an agent architecture is truly generalizable and robust.

**Why deterministic graders?** The rubric uses keyword matching, heuristic NLP signals (via \`sentence-transformers\`), semantic similarities, and exact comparisons — minimizing LLM-in-the-loop dependencies for grading and ensuring reproducible scores across runs for all 13 tasks.

**Dynamic Reward Shaping:** Every step returns a non-zero reward signal. Even incorrect actions return small penalties rather than zero, giving RL agents a gradient to follow rather than a sparse reward landscape.
