"""
Legal Document Review Tasks
"""

from app.models import (
    Action, Observation, Reward, LegalClause, AgentAction, ClauseType, RiskLevel
)
from data.legal_data import LEGAL_CLAUSES
import uuid
import random

class LegalIdentifyClauseTask:
    TASK_ID = "legal_clause_identification"

    def __init__(self):
        self.episode_id = ""
        self.step_count = 0
        self.clauses = []
        self.current_idx = 0
        self.results = []

    def reset(self) -> Observation:
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.current_idx = 0
        self.results = []
        self.clauses = random.sample(LEGAL_CLAUSES, min(10, len(LEGAL_CLAUSES)))
        return self._make_obs()

    def step(self, action: Action):
        self.step_count += 1
        if self.current_idx >= len(self.clauses):
            obs = Observation(task_id=self.TASK_ID, step=self.step_count, episode_done=True, info={})
            return obs, Reward(total=0.0), True, {}
        current = self.clauses[self.current_idx]
        
        score = 1.0 if action.clause_type == current["true_clause_type"] else 0.0
        
        self.results.append({
            "score": score,
            "predicted": action.clause_type,
            "true": current["true_clause_type"]
        })
        self.current_idx += 1
        done = self.current_idx >= len(self.clauses)
        
        reward = Reward(total=score, classification_accuracy=score)
        return self._make_obs(done), reward, done, {"step": self.step_count}

    def state(self): return {"task_id": self.TASK_ID, "step": self.step_count, "results": self.results}
    
    def grader_score(self):
        if not self.results: return {"final_score": 0.001}
        total = sum(r["score"] for r in self.results) / len(self.results)
        return {"task_id": self.TASK_ID, "final_score": max(0.001, min(0.999, total)), "passed": total >= 0.7}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.clauses):
            return Observation(task_id=self.TASK_ID, step=self.step_count, episode_done=True, info={})
        c = self.clauses[self.current_idx]
        clause_obj = LegalClause(clause_id=c["clause_id"], text=c["text"], contract_type=c["contract_type"], counterparty=c["counterparty"])
        return Observation(task_id=self.TASK_ID, step=self.step_count, current_clause=clause_obj, valid_actions=[AgentAction.IDENTIFY_CLAUSE])


class LegalRiskFlagTask:
    TASK_ID = "legal_risk_flagging"

    def __init__(self):
        self.current_idx = 0
        self.clauses = []
        self.results = []

    def reset(self) -> Observation:
        self.current_idx = 0
        self.results = []
        self.clauses = random.sample(LEGAL_CLAUSES, min(10, len(LEGAL_CLAUSES)))
        return self._make_obs()

    def step(self, action: Action):
        if self.current_idx >= len(self.clauses):
            obs = Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
            return obs, Reward(total=0.0), True, {}
        current = self.clauses[self.current_idx]

        # Partial credit for adjacent risk levels
        RISK_ORDER = ["low", "medium", "high", "critical"]
        true_risk = current["true_risk_level"].value if hasattr(current["true_risk_level"], "value") else str(current["true_risk_level"])
        pred_risk = action.risk_level.value if action.risk_level and hasattr(action.risk_level, "value") else (str(action.risk_level) if action.risk_level else "medium")
        try:
            diff = abs(RISK_ORDER.index(pred_risk) - RISK_ORDER.index(true_risk))
            score = 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)
        except ValueError:
            score = 0.0

        self.results.append({"score": score})
        self.current_idx += 1
        done = self.current_idx >= len(self.clauses)

        return self._make_obs(done), Reward(total=score), done, {}

    def state(self): return {"task_id": self.TASK_ID, "step": self.current_idx}

    def grader_score(self):
        total = sum(r["score"] for r in self.results) / len(self.results) if self.results else 0.0
        return {"task_id": self.TASK_ID, "final_score": max(0.001, min(0.999, round(total, 4))), "passed": total >= 0.6}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.clauses):
            return Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
        c = self.clauses[self.current_idx]
        clause_obj = LegalClause(clause_id=c["clause_id"], text=c["text"], contract_type=c["contract_type"], counterparty=c["counterparty"])
        return Observation(task_id=self.TASK_ID, step=self.current_idx, current_clause=clause_obj, valid_actions=[AgentAction.FLAG_RISK])


class LegalRedlineTask:
    TASK_ID = "legal_clause_redlining"

    def __init__(self):
        self.current_idx = 0
        self.clauses = []
        self.results = []

    def reset(self) -> Observation:
        self.current_idx = 0
        self.results = []
        self.clauses = random.sample(LEGAL_CLAUSES, min(5, len(LEGAL_CLAUSES)))  # 5 hard tasks
        return self._make_obs()

    def step(self, action: Action):
        current = self.clauses[self.current_idx]
        
        # Simple heuristic grader for redlining
        score = 0.0
        if action.redline_text and len(action.redline_text) > 20:
            score += 0.5
            if current["true_risk_level"] in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                if any(w in action.redline_text.lower() for w in ["cap", "liability", "breach", "mutual", "exception"]):
                    score += 0.5
            else:
                score += 0.5
                
        self.results.append({"score": score})
        self.current_idx += 1
        done = self.current_idx >= len(self.clauses)
        
        return self._make_obs(done), Reward(total=score), done, {}

    def state(self): return {"task_id": self.TASK_ID, "step": self.current_idx}
    
    def grader_score(self):
        total = sum(r["score"] for r in self.results) / len(self.results) if self.results else 0.0
        return {"task_id": self.TASK_ID, "final_score": max(0.001, min(0.999, total)), "passed": total >= 0.6}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.clauses):
            return Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
        c = self.clauses[self.current_idx]
        # Include the true risk so the agent knows what to redline against
        clause_obj = LegalClause(clause_id=c["clause_id"], text=c["text"], contract_type=c["contract_type"], counterparty=c["counterparty"], true_risk_level=c["true_risk_level"])
        return Observation(task_id=self.TASK_ID, step=self.current_idx, current_clause=clause_obj, valid_actions=[AgentAction.REDLINE])
