"""
Task 4: Multi-Turn Conversation Resolution (VERY HARD)
Agent must maintain context across up to 8 back-and-forth dialogue turns with an increasingly agitated customer.
Tests context-window retention, empathetic de-escalation, and appropriate routing/escalation thresholds.
"""

from app.models import (
    Action, Observation, Reward, Ticket, AgentAction, TicketCategory, TicketPriority
)
from typing import Dict, Any, Tuple, List
import uuid
import random

CONVERSATION_SCENARIOS = [
    {
        "category": TicketCategory.TECHNICAL,
        "initial": "My API is returning 500 errors",
        "followups": [
            "I already tried restarting the server, still broken.",
            "This is taking too long to resolve! I need this fixed NOW or we are moving providers.",
            "Can I just speak to a senior engineer or your manager?",
            "Hello??? Are you still there diagnosing this?"
        ],
        "resolution_criteria": {
            "target_turn": 3,  # Should escalate by turn 3 (when asking for manager)
            "must_escalate": True,
        }
    },
    {
        "category": TicketCategory.BILLING,
        "initial": "I was charged $49.99 but my subscription should be on the free tier.",
        "followups": [
            "No, I didn't upgrade by accident. The UI just charged me when I clicked 'view plans'.",
            "Are you calling me a liar? I demand a refund right now.",
            "I will file a chargeback with my bank if you don't process this refund.",
            "Fine. Thank you for issuing the refund."
        ],
        "resolution_criteria": {
            "target_turn": 4, 
            "must_resolve": True, # Should resolve it, not escalate
            "must_escalate": False,
        }
    },
    {
        "category": TicketCategory.FEATURE_REQUEST,
        "initial": "How do I connect this to Zapier?",
        "followups": [
            "What do you mean you don't have Zapier? Your pricing page says '100+ integrations'.",
            "This is false advertising. I want to talk to sales about a custom integration.",
            "If sales can't do it, give me a webhook workaround.",
            "Okay, the webhook workaround works. You can close this."
        ],
        "resolution_criteria": {
            "target_turn": 4,
            "must_resolve": True,
            "must_escalate": False,
        }
    },
    {
        "category": TicketCategory.UNKNOWN,
        "initial": "It's not working.",
        "followups": [
            "The app.",
            "When I open it, it just gives me a white screen.",
            "I am on iOS 17. Yes, I tried deleting and reinstalling.",
            "Wait, it just started working after I turned off my VPN. Nevermind!"
        ],
        "resolution_criteria": {
            "target_turn": 4,
            "must_resolve": True,
            "must_escalate": False,
        }
    },
    {
        "category": TicketCategory.ABUSE,
        "initial": "I'm being harassed by a user on your platform.",
        "followups": [
            "Their username is @badactor123.",
            "They are sending me hundreds of spam messages a minute. Block them!",
            "Thank you. Is my account safe now?",
            "Great, thanks for the quick help."
        ],
        "resolution_criteria": {
            "target_turn": 4,
            "must_resolve": True,
            "must_escalate": False,
        }
    }
]

class MultiTurnTask:
    TASK_ID = "multi_turn_conversation"
    MAX_TURNS = 8

    def __init__(self):
        self.episode_id: str = ""
        self.step_count: int = 0
        self.scenario: dict = {}
        self.history: List[Dict[str, str]] = []
        self.conversation_turn: int = 0
        self.ticket: Ticket = None
        self.status = "ongoing"  # ongoing, escalated, resolved, failed
        self.score = 0.0

    def reset(self) -> Observation:
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.conversation_turn = 0
        self.history = []
        self.status = "ongoing"
        self.score = 0.0
        
        self.scenario = random.choice(CONVERSATION_SCENARIOS)
        
        self.ticket = Ticket(
            ticket_id=f"TKT-MT-{uuid.uuid4().hex[:6].upper()}",
            subject="Assistance Required",
            body=self.scenario["initial"],
            customer_id="CUST-MULTI",
            customer_tier="pro",
            created_at="2026-01-01T12:00:00Z",
            sla_deadline="2026-01-02T12:00:00Z",
            category=self.scenario["category"],
            priority=TicketPriority.P2_HIGH
        )
        self.history.append({"role": "customer", "content": self.scenario["initial"]})

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = 0.0
        
        if action.action_type == AgentAction.ESCALATE:
            if self.scenario["resolution_criteria"].get("must_escalate", False):
                # Correct escalation
                if self.conversation_turn == self.scenario["resolution_criteria"]["target_turn"] - 1:
                    reward = 1.0 # Perfect timing
                else:
                    reward = 0.5 # Right action, wrong timing
                self.status = "escalated_correctly"
            else:
                reward = -0.5 # Unnecessary escalation
                self.status = "failed"
            done = True
            
        elif action.action_type == AgentAction.RESOLVE or action.action_type == AgentAction.CLOSE:
            if self.scenario["resolution_criteria"].get("must_resolve", True):
                if self.conversation_turn >= self.scenario["resolution_criteria"]["target_turn"] - 1:
                    reward = 1.0
                else:
                    reward = 0.0 # Closed too early
            else:
                reward = -0.5 # Resolved instead of escalating
            self.status = "resolved"
            done = True
            
        elif action.action_type == AgentAction.DRAFT_RESPONSE:
            response = action.response_text or ""
            self.history.append({"role": "agent", "content": response})
            
            if self.conversation_turn < len(self.scenario["followups"]):
                followup = self.scenario["followups"][self.conversation_turn]
                self.history.append({"role": "customer", "content": followup})
                self.ticket.body = followup # Update current message
                self.conversation_turn += 1
                
                # Small step reward for keeping the conversation going without failing
                reward = 0.1 
                done = False
            else:
                # Customer stopped responding, resolution achieved implicitly
                if self.scenario["resolution_criteria"].get("must_resolve", True):
                    reward = 1.0
                else:
                    reward = -0.5 # Should have escalated
                self.status = "resolved"
                done = True
        else:
            reward = -0.1
            done = False

        self.ticket.previous_interactions = self.history.copy()
        
        if done:
            self.score = reward

        obs = self._make_observation(done=done)
        return obs, Reward(total=reward), done, {"status": self.status, "turn": self.conversation_turn}

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.TASK_ID,
            "step": self.step_count,
            "turn": self.conversation_turn,
            "history_length": len(self.history)
        }

    def grader_score(self) -> Dict[str, Any]:
        return {
            "task_id": self.TASK_ID,
            "episode_id": self.episode_id,
            "final_score": max(0.0001, min(0.9999, self.score)),
            "passed": self.score >= 0.7,
            "metrics": {
                "turns_survived": self.conversation_turn,
                "end_status": self.status,
            }
        }

    def _make_observation(self, done: bool = False) -> Observation:
        return Observation(
            task_id=self.TASK_ID,
            step=self.step_count,
            current_ticket=self.ticket,
            valid_actions=[AgentAction.DRAFT_RESPONSE, AgentAction.ESCALATE, AgentAction.RESOLVE, AgentAction.CLOSE],
            episode_done=done,
            info={"current_turn": self.conversation_turn, "max_turns": self.MAX_TURNS}
        )
