"""
Senior Engineer Software PR Review Tasks
"""

from app.models import (
    Action, Observation, Reward, PullRequest, AgentAction, PRType
)
from data.pr_data import PULL_REQUESTS
import uuid
import random

class PRTypeTask:
    TASK_ID = "pr_type_classification"

    def __init__(self):
        self.current_idx = 0
        self.prs = []
        self.results = []

    def reset(self) -> Observation:
        self.current_idx = 0
        self.results = []
        self.prs = random.sample(PULL_REQUESTS, min(10, len(PULL_REQUESTS)))
        return self._make_obs()

    def step(self, action: Action):
        if self.current_idx >= len(self.prs):
            obs = Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
            return obs, Reward(total=0.0), True, {}
        current = self.prs[self.current_idx]
        score = 1.0 if action.pr_type == current["true_pr_type"] else 0.0
        self.results.append({"score": score})
        self.current_idx += 1
        done = self.current_idx >= len(self.prs)
        return self._make_obs(done), Reward(total=score), done, {}

    def state(self): return {"task_id": self.TASK_ID, "step": self.current_idx}
    
    def grader_score(self):
        total = sum(r["score"] for r in self.results) / len(self.results) if self.results else 0.0
        return {"task_id": self.TASK_ID, "final_score": max(0.01, min(0.99, total)), "passed": total >= 0.7}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.prs):
            return Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
        c = self.prs[self.current_idx]
        pr_obj = PullRequest(pr_id=c["pr_id"], title=c["title"], description=c["description"], diff=c["diff"], author=c["author"])
        return Observation(task_id=self.TASK_ID, step=self.current_idx, current_pr=pr_obj, valid_actions=[AgentAction.CLASSIFY_PR])


class PRBugIdentifyTask:
    TASK_ID = "pr_bug_identification"

    def __init__(self):
        self.current_idx = 0
        self.prs = []
        self.results = []

    def reset(self) -> Observation:
        self.current_idx = 0
        self.results = []
        # Filter to only PRs with actual bugs or security issues for this task
        issue_prs = [pr for pr in PULL_REQUESTS if pr["true_pr_type"] in [PRType.SECURITY, PRType.FEATURE, PRType.REFACTOR]]
        self.prs = random.sample(issue_prs, min(5, len(issue_prs)))
        return self._make_obs()

    def step(self, action: Action):
        if self.current_idx >= len(self.prs):
            obs = Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
            return obs, Reward(total=0.0), True, {}
        current = self.prs[self.current_idx]
        
        # Heuristic grading: did they identify the core issue?
        score = 0.0
        if action.bug_description and len(action.bug_description) > 10:
            import re as _re
            desc = action.bug_description.lower()
            true_bug = current["true_bug_description"].lower()

            # Check if this is a "no bug" case
            no_bug_signals = ["none", "no bug", "correct", "correctly", "lgtm", "looks good", "addresses"]
            true_is_none = true_bug.startswith("none") or "no bug" in true_bug or "correctly" in true_bug

            if true_is_none:
                # Score 1.0 if agent also says no bug, else 0.0
                agent_says_no_bug = any(s in desc for s in no_bug_signals)
                score = 1.0 if agent_says_no_bug else 0.0
            else:
                # Strip punctuation from keywords for matching
                keywords = [_re.sub(r"[^a-z0-9_]", "", w) for w in true_bug.split() if len(w) > 4]
                keywords = [k for k in keywords if k]  # remove empty after strip
                if not keywords:
                    score = 1.0
                else:
                    matches = sum(1 for k in keywords if k in desc)
                    score = matches / len(keywords)
                    if score > 0.25: score = 1.0  # Lenient threshold
                
        self.results.append({"score": score})
        self.current_idx += 1
        done = self.current_idx >= len(self.prs)
        return self._make_obs(done), Reward(total=score), done, {}

    def state(self): return {"task_id": self.TASK_ID, "step": self.current_idx}
    
    def grader_score(self):
        total = sum(r["score"] for r in self.results) / len(self.results) if self.results else 0.0
        return {"task_id": self.TASK_ID, "final_score": max(0.01, min(0.99, total)), "passed": total >= 0.6}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.prs):
            return Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
        c = self.prs[self.current_idx]
        pr_obj = PullRequest(pr_id=c["pr_id"], title=c["title"], description=c["description"], diff=c["diff"], author=c["author"])
        return Observation(task_id=self.TASK_ID, step=self.current_idx, current_pr=pr_obj, valid_actions=[AgentAction.IDENTIFY_BUG])


class PRReviewTask:
    TASK_ID = "pr_review_comment"

    def __init__(self):
        self.current_idx = 0
        self.prs = []
        self.results = []

    def reset(self) -> Observation:
        self.current_idx = 0
        self.results = []
        self.prs = random.sample(PULL_REQUESTS, min(5, len(PULL_REQUESTS)))
        return self._make_obs()

    def step(self, action: Action):
        if self.current_idx >= len(self.prs):
            obs = Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
            return obs, Reward(total=0.0), True, {}
        current = self.prs[self.current_idx]
        score = 0.0
        if action.review_comment and len(action.review_comment) > 20:
            score += 0.5
            if "looks good" in action.review_comment.lower() or "lgtm" in action.review_comment.lower():
                if current["true_pr_type"] == PRType.BUG_FIX and current["true_bug_description"] == "None":
                    score += 0.5  # Correct to approve
                else:
                    score -= 0.5  # Incorrect to just approve a buggy PR
            else:
                # Some critique was offered
                score += 0.5
                
        # Ensure score within 0-1
        score = max(0.0, min(1.0, score))
        
        self.results.append({"score": score})
        self.current_idx += 1
        done = self.current_idx >= len(self.prs)
        return self._make_obs(done), Reward(total=score), done, {}

    def state(self): return {"task_id": self.TASK_ID, "step": self.current_idx}
    
    def grader_score(self):
        total = sum(r["score"] for r in self.results) / len(self.results) if self.results else 0.0
        return {"task_id": self.TASK_ID, "final_score": max(0.01, min(0.99, total)), "passed": total >= 0.6}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.prs):
            return Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
        c = self.prs[self.current_idx]
        pr_obj = PullRequest(pr_id=c["pr_id"], title=c["title"], description=c["description"], diff=c["diff"], author=c["author"], true_bug_description=c["true_bug_description"])
        return Observation(task_id=self.TASK_ID, step=self.current_idx, current_pr=pr_obj, valid_actions=[AgentAction.REVIEW_PR])
