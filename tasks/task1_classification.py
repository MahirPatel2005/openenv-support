"""
Task 1: Ticket Classification & Routing (EASY)
Agent must classify ticket by category and assign correct priority.
Score: 0.0-1.0 based on accuracy of both fields.
"""

from app.models import (
    Action, Observation, Reward, Ticket,
    TicketCategory, TicketPriority, AgentAction,
    KnowledgeBaseArticle
)
from data.tickets import generate_ticket
from data.knowledge_base import get_relevant_articles
from typing import Dict, Any, Tuple
import uuid


CORRECT_CATEGORY_WEIGHT = 0.6
CORRECT_PRIORITY_WEIGHT = 0.4

# Priority adjacency — partial credit for being one level off
PRIORITY_PARTIAL_CREDIT = {
    ("P1", "P2"): 0.5, ("P2", "P1"): 0.5,
    ("P2", "P3"): 0.5, ("P3", "P2"): 0.5,
    ("P3", "P4"): 0.5, ("P4", "P3"): 0.5,
}


class ClassificationTask:
    TASK_ID = "ticket_classification"
    MAX_STEPS = 10  # 10 tickets to classify per episode

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

        # Generate tickets with ground-truth labels
        self.tickets = []
        from data.tickets import TICKET_POOL
        import random

        for pool_item in random.sample(TICKET_POOL, min(self.MAX_STEPS, len(TICKET_POOL))):
            t = generate_ticket(pool_item.copy())
            # Store ground truth but hide category/priority from observation
            self.tickets.append(t)

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1

        current = self.tickets[self.current_idx]
        reward = self._grade_action(action, current)
        self.reward_history.append(reward.total)
        self.results.append({
            "ticket_id": current.ticket_id,
            "predicted_category": action.category,
            "true_category": current.category,
            "predicted_priority": action.priority,
            "true_priority": current.priority,
            "score": reward.total,
        })

        self.current_idx += 1
        done = self.current_idx >= len(self.tickets)

        obs = self._make_observation(done=done)
        return obs, reward, done, {"episode_id": self.episode_id, "step": self.step_count}

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.TASK_ID,
            "episode_id": self.episode_id,
            "step": self.step_count,
            "current_ticket_idx": self.current_idx,
            "total_tickets": len(self.tickets),
            "reward_history": self.reward_history,
        }

    def grader_score(self) -> Dict[str, Any]:
        if not self.results:
            return {"final_score": 0.0, "metrics": {}}

        total = sum(r["score"] for r in self.results) / len(self.results)
        cat_acc = sum(
            1 for r in self.results
            if r["predicted_category"] == r["true_category"]
        ) / len(self.results)
        pri_acc = sum(
            1 for r in self.results
            if r["predicted_priority"] == r["true_priority"]
        ) / len(self.results)

        return {
            "task_id": self.TASK_ID,
            "episode_id": self.episode_id,
            "final_score": round(total, 4),
            "passed": total >= 0.7,
            "metrics": {
                "category_accuracy": round(cat_acc, 4),
                "priority_accuracy": round(pri_acc, 4),
                "tickets_classified": len(self.results),
                "per_ticket_scores": self.results,
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
                info={"message": "Episode complete"},
            )

        current = self.tickets[self.current_idx]
        # Present ticket without ground-truth category/priority
        visible_ticket = Ticket(
            ticket_id=current.ticket_id,
            subject=current.subject,
            body=current.body,
            customer_id=current.customer_id,
            customer_tier=current.customer_tier,
            created_at=current.created_at,
            sla_deadline=current.sla_deadline,
            sentiment_score=current.sentiment_score,
            tags=[],  # Tags hidden — agent must infer
        )

        return Observation(
            task_id=self.TASK_ID,
            step=self.step_count,
            current_ticket=visible_ticket,
            valid_actions=[AgentAction.CLASSIFY],
            episode_done=done,
            info={
                "remaining": len(self.tickets) - self.current_idx,
                "progress": self.current_idx / len(self.tickets),
            },
        )

    def _grade_action(self, action: Action, ticket: Ticket) -> Reward:
        if action.action_type != AgentAction.CLASSIFY:
            return Reward(total=-0.1, penalty=-0.1, breakdown={"wrong_action_type": -0.1})

        cat_score = 0.0
        pri_score = 0.0
        breakdown = {}

        # Category score
        if action.category == ticket.category:
            cat_score = 1.0
        breakdown["category"] = cat_score

        # Priority score (with partial credit)
        if action.priority == ticket.priority:
            pri_score = 1.0
        elif action.priority and ticket.priority:
            partial_key = (action.priority.value, ticket.priority.value)
            pri_score = PRIORITY_PARTIAL_CREDIT.get(partial_key, 0.0)
        breakdown["priority"] = pri_score

        total = (cat_score * CORRECT_CATEGORY_WEIGHT) + (pri_score * CORRECT_PRIORITY_WEIGHT)

        return Reward(
            total=round(total, 4),
            classification_accuracy=total,
            breakdown=breakdown,
        )
