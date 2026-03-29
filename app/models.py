"""
Typed Pydantic models for the Customer Support Triage, Legal, Clinical, and Engineering OpenEnv environments.
All models follow the OpenEnv specification for Observation, Action, and Reward.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import enum


# ─────────────────────────── Enumerations ────────────────────────────

# Support Enums
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

# Legal Enums
class ClauseType(str, enum.Enum):
    INDEMNITY = "indemnity"
    LIABILITY = "liability"
    IP = "ip"
    TERMINATION = "termination"
    UNKNOWN = "unknown"

class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Clinical Enums
class BodySystem(str, enum.Enum):
    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    NEUROLOGIC = "neurologic"
    GI = "gi"
    MUSCULOSKELETAL = "musculoskeletal"
    OTHER = "other"

# PR Enums
class PRType(str, enum.Enum):
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    SECURITY = "security"

class AgentAction(str, enum.Enum):
    # Support
    CLASSIFY = "classify"
    DRAFT_RESPONSE = "draft_response"
    ASSIGN_TICKET = "assign_ticket"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    CLOSE = "close"
    
    # Legal
    IDENTIFY_CLAUSE = "identify_clause"
    FLAG_RISK = "flag_risk"
    REDLINE = "redline"
    
    # Clinical
    CLASSIFY_TRIAGE = "classify_triage"
    ASSIGN_ESI = "assign_esi"
    WRITE_TRIAGE_NOTE = "write_triage_note"
    
    # Engineering
    CLASSIFY_PR = "classify_pr"
    IDENTIFY_BUG = "identify_bug"
    REVIEW_PR = "review_pr"
    
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

class LegalClause(BaseModel):
    clause_id: str
    text: str
    contract_type: str
    counterparty: str
    true_clause_type: Optional[ClauseType] = None
    true_risk_level: Optional[RiskLevel] = None
    true_risk_justification: Optional[str] = None

class ClinicalPatient(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    vitals: Dict[str, str]
    history: str
    true_body_system: Optional[BodySystem] = None
    true_esi_level: Optional[int] = None
    true_triage_note: Optional[str] = None

class PullRequest(BaseModel):
    pr_id: str
    title: str
    description: str
    diff: str
    author: str
    true_pr_type: Optional[PRType] = None
    true_bug_description: Optional[str] = None
    true_review_comment: Optional[str] = None

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
    What the agent sees at each step. Includes the current domain object
    (ticket, clause, patient, or PR).
    """
    task_id: str = Field(description="Active task identifier")
    step: int = Field(description="Current step number within episode")
    
    current_ticket: Optional[Ticket] = Field(default=None)
    current_clause: Optional[LegalClause] = Field(default=None)
    current_patient: Optional[ClinicalPatient] = Field(default=None)
    current_pr: Optional[PullRequest] = Field(default=None)
    
    ticket_queue: List[Ticket] = Field(default_factory=list)
    agents: List[AgentInfo] = Field(default_factory=list)
    knowledge_base: List[KnowledgeBaseArticle] = Field(default_factory=list)
    sla_status: Dict[str, str] = Field(default_factory=dict)
    valid_actions: List[str] = Field(default_factory=list)
    episode_done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """
    An action the agent takes. Fields requested depend on action_type.
    """
    action_type: AgentAction

    # Support
    ticket_id: Optional[str] = None
    category: Optional[TicketCategory] = None
    priority: Optional[TicketPriority] = None
    response_text: Optional[str] = None
    target_agent_id: Optional[str] = None
    resolution_summary: Optional[str] = None

    # Legal
    clause_id: Optional[str] = None
    clause_type: Optional[ClauseType] = None
    risk_level: Optional[RiskLevel] = None
    redline_text: Optional[str] = None
    
    # Clinical
    patient_id: Optional[str] = None
    body_system: Optional[BodySystem] = None
    esi_level: Optional[int] = Field(None, ge=1, le=5)
    triage_note: Optional[str] = None

    # Engineering
    pr_id: Optional[str] = None
    pr_type: Optional[PRType] = None
    bug_description: Optional[str] = None
    review_comment: Optional[str] = None

    reasoning: Optional[str] = None


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
    penalty: float = Field(default=0.0, le=0.0)
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
