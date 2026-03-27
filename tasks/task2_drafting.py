"""
Task 2: Response Drafting & Quality (MEDIUM)
Agent drafts a customer-facing response to a pre-classified ticket.
Score: 0.0-1.0 based on factual accuracy, completeness, tone, and KB usage.
"""

from app.models import (
    Action, Observation, Reward, Ticket,
    TicketCategory, AgentAction, AgentInfo
)
from data.tickets import generate_ticket, TICKET_POOL
from data.knowledge_base import get_relevant_articles, KNOWLEDGE_BASE
from typing import Dict, Any, Tuple, List
import uuid
import re


# Grading rubric weights
WEIGHTS = {
    "kb_reference": 0.25,       # Uses information from KB
    "addresses_issue": 0.30,    # Directly addresses the customer's specific issue
    "actionable_steps": 0.20,   # Provides concrete next steps
    "tone_empathy": 0.15,       # Professional, empathetic tone
    "no_hallucination": 0.10,   # No false promises or invented info
}

# Keywords that indicate KB usage per category
KB_SIGNALS = {
    TicketCategory.BILLING: ["refund", "business day", "stripe", "payment", "invoice"],
    TicketCategory.TECHNICAL: ["status.company.com", "request_id", "engineering", "on-call", "workaround"],
    TicketCategory.ACCOUNT: ["settings", "team", "admin", "invite", "escalat"],
    TicketCategory.FEATURE_REQUEST: ["roadmap.company.com", "upvote", "roadmap", "csm"],
    TicketCategory.ABUSE: ["trust", "safety", "24 hour", "investigate", "disable"],
}

# Forbidden phrases (hallucination signals)
FORBIDDEN_PATTERNS = [
    r"will be fixed (today|tonight|tomorrow|this week)",
    r"guarantee(d)? (resolution|fix)",
    r"your (data|account) (is|will be) deleted",
    r"we (will|can) refund (everything|all)",
    r"free (forever|for life)",
]

EMPATHY_SIGNALS = [
    "apologize", "sorry", "understand", "frustrat", "inconvenien",
    "appreciate", "thank you", "we hear you", "important to us",
]

CLOSING_SIGNALS = [
    "let me know", "please reach out", "feel free", "happy to help",
    "any questions", "here for you",
]


class ResponseDraftingTask:
    TASK_ID = "response_drafting"
    MAX_STEPS = 6  # 6 tickets to respond to

    def __init__(self):
        self.episode_id: str = ""
        self.step_count: int = 0
        self.tickets: list = []
        self.current_idx: int = 0
        self.reward_history: list = []
        self.results: list = []

    def reset(self) -> Observation:
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.current_idx = 0
        self.reward_history = []
        self.results = []

        import random
        sampled = random.sample(TICKET_POOL, min(self.MAX_STEPS, len(TICKET_POOL)))
        self.tickets = [generate_ticket(p.copy()) for p in sampled]

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        current = self.tickets[self.current_idx]

        if action.action_type != AgentAction.DRAFT_RESPONSE:
            reward = Reward(total=-0.15, penalty=-0.15, breakdown={"wrong_action": -0.15})
        else:
            reward = self._grade_response(action, current)

        self.reward_history.append(reward.total)
        self.results.append({
            "ticket_id": current.ticket_id,
            "category": current.category.value if current.category else None,
            "response_length": len(action.response_text or ""),
            "score": reward.total,
            "breakdown": reward.breakdown,
        })

        self.current_idx += 1
        done = self.current_idx >= len(self.tickets)
        obs = self._make_observation(done=done)
        return obs, reward, done, {"episode_id": self.episode_id}

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.TASK_ID,
            "episode_id": self.episode_id,
            "step": self.step_count,
            "current_idx": self.current_idx,
            "reward_history": self.reward_history,
        }

    def grader_score(self) -> Dict[str, Any]:
        if not self.results:
            return {"final_score": 0.0, "metrics": {}}
        avg = sum(r["score"] for r in self.results) / len(self.results)
        return {
            "task_id": self.TASK_ID,
            "episode_id": self.episode_id,
            "final_score": round(avg, 4),
            "passed": avg >= 0.6,
            "metrics": {
                "tickets_responded": len(self.results),
                "per_ticket": self.results,
                "avg_response_length": sum(r["response_length"] for r in self.results) / max(len(self.results), 1),
            },
        }

    # ─── Private ─────────────────────────────────────────────────────

    def _make_observation(self, done: bool = False) -> Observation:
        if self.current_idx >= len(self.tickets):
            return Observation(
                task_id=self.TASK_ID,
                step=self.step_count,
                episode_done=True,
                valid_actions=[],
            )

        current = self.tickets[self.current_idx]
        kb = get_relevant_articles(current.category, top_k=2) if current.category else []

        return Observation(
            task_id=self.TASK_ID,
            step=self.step_count,
            current_ticket=current,  # Category/priority revealed for drafting
            knowledge_base=kb,
            valid_actions=[AgentAction.DRAFT_RESPONSE],
            episode_done=done,
            info={
                "remaining": len(self.tickets) - self.current_idx,
                "instruction": (
                    "Draft a complete, professional customer-facing response. "
                    "Reference the knowledge base articles where relevant. "
                    "Address the customer's specific issue with concrete next steps."
                ),
            },
        )

    def _grade_response(self, action: Action, ticket: Ticket) -> Reward:
        text = (action.response_text or "").lower()
        breakdown = {}
        penalty = 0.0

        # 1. KB reference — does response use KB signals?
        signals = KB_SIGNALS.get(ticket.category, [])
        kb_hits = sum(1 for s in signals if s in text)
        kb_score = min(kb_hits / max(len(signals), 1), 1.0)
        breakdown["kb_reference"] = round(kb_score, 3)

        # 2. Addresses the issue — does it mention key terms from the ticket?
        ticket_keywords = set(
            w for w in re.findall(r'\b\w{4,}\b', ticket.subject.lower() + " " + ticket.body.lower())
            if w not in {"this", "that", "with", "have", "your", "from", "when", "they", "their"}
        )
        response_keywords = set(re.findall(r'\b\w{4,}\b', text))
        overlap = len(ticket_keywords & response_keywords)
        address_score = min(overlap / max(len(ticket_keywords) * 0.3, 1), 1.0)
        breakdown["addresses_issue"] = round(address_score, 3)

        # 3. Actionable steps — numbered list or action verbs
        has_numbered = bool(re.search(r'\b(step \d|first|second|third|\d\.|please )', text))
        has_action_verbs = bool(re.search(r'\b(go to|click|navigate|contact|email|send|open|check)\b', text))
        action_score = 0.5 * has_numbered + 0.5 * has_action_verbs
        breakdown["actionable_steps"] = round(action_score, 3)

        # 4. Tone & empathy
        empathy_hits = sum(1 for s in EMPATHY_SIGNALS if s in text)
        closing_hits = sum(1 for s in CLOSING_SIGNALS if s in text)
        tone_score = min((empathy_hits * 0.6 + closing_hits * 0.4) / 2.0, 1.0)
        breakdown["tone_empathy"] = round(tone_score, 3)

        # 5. No hallucination — penalize forbidden patterns
        hallucination_count = sum(1 for p in FORBIDDEN_PATTERNS if re.search(p, text))
        hallucination_score = max(1.0 - hallucination_count * 0.5, 0.0)
        if hallucination_count > 0:
            penalty -= hallucination_count * 0.1
        breakdown["no_hallucination"] = round(hallucination_score, 3)

        # Length penalty (too short = incomplete, too long = padding)
        word_count = len(text.split())
        if word_count < 30:
            penalty -= 0.2
        elif word_count > 500:
            penalty -= 0.05

        total = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS) + penalty
        total = max(0.0, min(1.0, total))

        return Reward(
            total=round(total, 4),
            response_quality=total,
            penalty=round(penalty, 4),
            breakdown=breakdown,
        )
