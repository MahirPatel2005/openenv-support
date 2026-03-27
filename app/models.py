"""
Typed Pydantic models for the Customer Support Triage OpenEnv environment.
All models follow the OpenEnv specification for Observation, Action, and Reward.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import enum


# ─────────────────────────── Enumerations ────────────────────────────

class TicketCategory(str, enum.Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    ABUSE = "abuse"
    UNKNOWN = "unknown"


class TicketPriority(str, enum.Enum):
    P1_CRITICAL = "P1"
    P2_HIGH = "P2"
    P3_MEDIUM = "P3"
    P4_LOW = "P4"


class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class AgentAction(str, enum.Enum):
    CLASSIFY = "classify"
    DRAFT_RESPONSE = "draft_response"
    ASSIGN_TICKET = "assign_ticket"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    CLOSE = "close"
    NO_OP = "no_op"


# ─────────────────────────── Sub-models ───────────────────────────────

class Ticket(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_id: str
    customer_tier: Literal["free", "pro", "enterprise"]
    created_at: str
    sla_deadline: str
    category: Optional[TicketCategory] = None
    priority: Optional[TicketPriority] = None
    status: TicketStatus = TicketStatus.OPEN
    assigned_agent: Optional[str] = None
    previous_interactions: List[Dict[str, str]] = Field(default_factory=list)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    tags: List[str] = Field(default_factory=list)


class AgentInfo(BaseModel):
    agent_id: str
    name: str
    specialization: List[TicketCategory]
    current_load: int = 0
    max_load: int = 5
    availability: bool = True


class KnowledgeBaseArticle(BaseModel):
    article_id: str
    title: str
    content: str
    applicable_categories: List[TicketCategory]
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ─────────────────────────── Core Models ──────────────────────────────

class Observation(BaseModel):
    """
    What the agent sees at each step. Contains the current ticket,
    agent roster, knowledge base snippets, and queue snapshot.
    """
    task_id: str = Field(description="Active task identifier")
    step: int = Field(description="Current step number within episode")
    current_ticket: Optional[Ticket] = Field(
        default=None,
        description="The ticket currently requiring action"
    )
    ticket_queue: List[Ticket] = Field(
        default_factory=list,
        description="Snapshot of open tickets in the queue (hard task only)"
    )
    agents: List[AgentInfo] = Field(
        default_factory=list,
        description="Available support agents (hard task only)"
    )
    knowledge_base: List[KnowledgeBaseArticle] = Field(
        default_factory=list,
        description="Relevant KB articles for the current ticket"
    )
    sla_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of ticket_id -> SLA status (ok/warning/breached)"
    )
    valid_actions: List[str] = Field(
        default_factory=list,
        description="List of valid action types for the current state"
    )
    episode_done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """
    An action the agent takes. The action_type determines which fields
    are required. See /tasks for the full schema per task.
    """
    action_type: AgentAction = Field(
        description="Type of action to perform"
    )
    ticket_id: Optional[str] = Field(
        default=None,
        description="Target ticket ID (required for most actions)"
    )
    # For CLASSIFY
    category: Optional[TicketCategory] = Field(
        default=None,
        description="Ticket category (required for classify action)"
    )
    priority: Optional[TicketPriority] = Field(
        default=None,
        description="Ticket priority (required for classify action)"
    )
    # For DRAFT_RESPONSE
    response_text: Optional[str] = Field(
        default=None,
        description="Full response text to send to customer (required for draft_response)"
    )
    # For ASSIGN_TICKET / ESCALATE
    target_agent_id: Optional[str] = Field(
        default=None,
        description="Agent to assign to (required for assign_ticket/escalate)"
    )
    # For RESOLVE
    resolution_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of how ticket was resolved"
    )
    # Metadata
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning (optional, used for transparency)"
    )


class Reward(BaseModel):
    """
    Step-level reward signal with breakdown for interpretability.
    """
    total: float = Field(description="Total reward for this step", ge=-1.0, le=1.0)
    classification_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    response_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    sla_compliance: float = Field(default=0.0, ge=-1.0, le=1.0)
    first_contact_resolution: float = Field(default=0.0, ge=0.0, le=1.0)
    customer_satisfaction: float = Field(default=0.0, ge=-1.0, le=1.0)
    penalty: float = Field(default=0.0, le=0.0, description="Negative penalty applied")
    breakdown: Dict[str, float] = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    """Final episode result returned from /grader."""
    task_id: str
    episode_id: str
    total_steps: int
    final_score: float = Field(ge=0.0, le=1.0)
    reward_history: List[float]
    metrics: Dict[str, Any]
    passed: bool
