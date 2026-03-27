"""
Task 3: SLA Queue Management (HARD)
Agent manages a queue of 20 tickets across 3 specialized agents.
Must maximize: SLA compliance, first-contact resolution, CSAT.
Must minimize: escalations, SLA breaches, agent overload.
"""

from app.models import (
    Action, Observation, Reward, Ticket, AgentInfo,
    TicketCategory, TicketPriority, TicketStatus, AgentAction
)
from data.tickets import generate_queue
from data.knowledge_base import get_relevant_articles
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import uuid


AGENTS = [
    AgentInfo(
        agent_id="agent_billing",
        name="Jordan (Billing Specialist)",
        specialization=[TicketCategory.BILLING],
        max_load=6,
    ),
    AgentInfo(
        agent_id="agent_tech",
        name="Sam (Technical Support)",
        specialization=[TicketCategory.TECHNICAL, TicketCategory.ACCOUNT],
        max_load=5,
    ),
    AgentInfo(
        agent_id="agent_general",
        name="Alex (General Support)",
        specialization=[TicketCategory.FEATURE_REQUEST, TicketCategory.ABUSE, TicketCategory.ACCOUNT],
        max_load=7,
    ),
]

PRIORITY_SLA_MINUTES = {"P1": 240, "P2": 480, "P3": 1440, "P4": 4320}
ESCALATION_THRESHOLD = 3  # Escalations beyond this count as excessive


class QueueManagementTask:
    TASK_ID = "queue_management"
    MAX_STEPS = 40  # Each step is one management action

    def __init__(self):
        self.episode_id: str = ""
        self.step_count: int = 0
        self.queue: List[Ticket] = []
        self.agents: List[AgentInfo] = []
        self.reward_history: list = []
        self.escalation_count: int = 0
        self.resolved_count: int = 0
        self.sla_breaches: int = 0
        self.fcr_count: int = 0  # First-contact resolutions
        self.closed_tickets: list = []
        self.episode_start: datetime = datetime.utcnow()

    def reset(self) -> Observation:
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.reward_history = []
        self.escalation_count = 0
        self.resolved_count = 0
        self.sla_breaches = 0
        self.fcr_count = 0
        self.closed_tickets = []
        self.episode_start = datetime.utcnow()

        self.queue = generate_queue(size=20)
        self.agents = [a.model_copy() for a in AGENTS]

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = self._execute_action(action)
        self.reward_history.append(reward.total)

        # Check for new SLA breaches each step
        self._update_sla_status()

        done = (
            self.step_count >= self.MAX_STEPS
            or len(self._open_tickets()) == 0
        )

        obs = self._make_observation(done=done)
        return obs, reward, done, {
            "episode_id": self.episode_id,
            "step": self.step_count,
            "open_tickets": len(self._open_tickets()),
            "resolved": self.resolved_count,
        }

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.TASK_ID,
            "episode_id": self.episode_id,
            "step": self.step_count,
            "open_tickets": len(self._open_tickets()),
            "resolved_count": self.resolved_count,
            "sla_breaches": self.sla_breaches,
            "escalation_count": self.escalation_count,
            "fcr_rate": self.fcr_count / max(self.resolved_count, 1),
        }

    def grader_score(self) -> Dict[str, Any]:
        total_tickets = len(self.queue)
        open_remaining = len(self._open_tickets())
        resolution_rate = self.resolved_count / max(total_tickets, 1)
        sla_compliance = 1.0 - (self.sla_breaches / max(total_tickets, 1))
        fcr_rate = self.fcr_count / max(self.resolved_count, 1)
        escalation_penalty = min(self.escalation_count / ESCALATION_THRESHOLD, 1.0) * 0.2

        final_score = (
            resolution_rate * 0.35
            + sla_compliance * 0.35
            + fcr_rate * 0.20
            - escalation_penalty
        )
        final_score = max(0.0, min(1.0, final_score))

        return {
            "task_id": self.TASK_ID,
            "episode_id": self.episode_id,
            "final_score": round(final_score, 4),
            "passed": final_score >= 0.5,
            "metrics": {
                "total_tickets": total_tickets,
                "resolved": self.resolved_count,
                "resolution_rate": round(resolution_rate, 4),
                "sla_compliance": round(sla_compliance, 4),
                "sla_breaches": self.sla_breaches,
                "fcr_rate": round(fcr_rate, 4),
                "escalations": self.escalation_count,
                "open_remaining": open_remaining,
                "steps_used": self.step_count,
            },
        }

    # ─── Private ─────────────────────────────────────────────────────

    def _open_tickets(self) -> List[Ticket]:
        return [t for t in self.queue if t.status not in (TicketStatus.RESOLVED, TicketStatus.CLOSED)]

    def _find_ticket(self, ticket_id: str) -> Optional[Ticket]:
        return next((t for t in self.queue if t.ticket_id == ticket_id), None)

    def _find_agent(self, agent_id: str) -> Optional[AgentInfo]:
        return next((a for a in self.agents if a.agent_id == agent_id), None)

    def _update_sla_status(self):
        now = datetime.utcnow().isoformat() + "Z"
        for ticket in self._open_tickets():
            if ticket.sla_deadline and ticket.sla_deadline < now:
                self.sla_breaches += 1
                ticket.status = TicketStatus.CLOSED  # Breached

    def _make_observation(self, done: bool = False) -> Observation:
        now = datetime.utcnow().isoformat() + "Z"
        sla_map = {}
        for t in self.queue:
            if t.sla_deadline:
                if t.sla_deadline < now:
                    sla_map[t.ticket_id] = "breached"
                elif t.sla_deadline < (datetime.utcnow().isoformat() + "Z"):
                    sla_map[t.ticket_id] = "warning"
                else:
                    sla_map[t.ticket_id] = "ok"

        open_q = self._open_tickets()[:10]  # Show top 10

        return Observation(
            task_id=self.TASK_ID,
            step=self.step_count,
            ticket_queue=open_q,
            agents=self.agents,
            sla_status=sla_map,
            valid_actions=[
                AgentAction.ASSIGN_TICKET,
                AgentAction.ESCALATE,
                AgentAction.RESOLVE,
                AgentAction.CLOSE,
                AgentAction.NO_OP,
            ],
            episode_done=done,
            info={
                "open_count": len(self._open_tickets()),
                "resolved": self.resolved_count,
                "sla_breaches": self.sla_breaches,
                "step": self.step_count,
                "max_steps": self.MAX_STEPS,
            },
        )

    def _execute_action(self, action: Action) -> Reward:
        breakdown = {}
        penalty = 0.0
        reward_val = 0.0

        if action.action_type == AgentAction.NO_OP:
            # Penalize no-op if there are unassigned high-priority tickets
            unassigned_p1 = [
                t for t in self._open_tickets()
                if t.priority == TicketPriority.P1_CRITICAL and not t.assigned_agent
            ]
            penalty = -0.1 * len(unassigned_p1)
            return Reward(total=max(penalty, -1.0), penalty=penalty, breakdown={"no_op_penalty": penalty})

        ticket = self._find_ticket(action.ticket_id) if action.ticket_id else None

        if not ticket:
            return Reward(total=-0.05, penalty=-0.05, breakdown={"invalid_ticket": -0.05})

        if action.action_type == AgentAction.ASSIGN_TICKET:
            agent = self._find_agent(action.target_agent_id) if action.target_agent_id else None
            if not agent:
                return Reward(total=-0.05, penalty=-0.05, breakdown={"invalid_agent": -0.05})

            # Bonus for specialization match
            spec_match = ticket.category in agent.specialization if ticket.category else False
            load_ok = agent.current_load < agent.max_load

            if not load_ok:
                penalty -= 0.1
                breakdown["overload_penalty"] = -0.1
            else:
                agent.current_load += 1

            ticket.assigned_agent = agent.agent_id
            ticket.status = TicketStatus.IN_PROGRESS
            reward_val = 0.15 + (0.1 if spec_match else 0.0)
            breakdown["assignment"] = reward_val

        elif action.action_type == AgentAction.ESCALATE:
            self.escalation_count += 1
            # Only reward escalation for P1/P2 tickets
            is_appropriate = ticket.priority in (TicketPriority.P1_CRITICAL, TicketPriority.P2_HIGH)
            reward_val = 0.1 if is_appropriate else -0.1
            penalty = -0.05 if self.escalation_count > ESCALATION_THRESHOLD else 0.0
            ticket.status = TicketStatus.ESCALATED
            breakdown["escalation"] = reward_val

        elif action.action_type == AgentAction.RESOLVE:
            if ticket.status not in (TicketStatus.IN_PROGRESS, TicketStatus.ESCALATED):
                penalty -= 0.05
                breakdown["resolve_penalty"] = "ticket not in progress"
            else:
                now = datetime.utcnow().isoformat() + "Z"
                sla_ok = ticket.sla_deadline and ticket.sla_deadline > now
                reward_val = 0.3 + (0.2 if sla_ok else -0.1)

                # FCR bonus: resolved without escalation
                if ticket.status != TicketStatus.ESCALATED:
                    self.fcr_count += 1
                    reward_val += 0.1

                # Customer tier bonus
                if ticket.customer_tier == "enterprise":
                    reward_val += 0.05

                ticket.status = TicketStatus.RESOLVED
                self.resolved_count += 1

                # Free up agent
                if ticket.assigned_agent:
                    agent = self._find_agent(ticket.assigned_agent)
                    if agent and agent.current_load > 0:
                        agent.current_load -= 1

                self.closed_tickets.append(ticket.ticket_id)
                breakdown["resolution"] = reward_val

        elif action.action_type == AgentAction.CLOSE:
            ticket.status = TicketStatus.CLOSED
            reward_val = 0.05
            breakdown["close"] = reward_val

        total = max(-1.0, min(1.0, reward_val + penalty))
        return Reward(
            total=round(total, 4),
            sla_compliance=max(0, reward_val),
            penalty=round(penalty, 4),
            first_contact_resolution=self.fcr_count / max(self.resolved_count, 1),
            breakdown=breakdown,
        )
