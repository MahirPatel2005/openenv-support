"""
Clinical Patient Triage Tasks
"""

from app.models import (
    Action, Observation, Reward, ClinicalPatient, AgentAction, BodySystem
)
from data.clinical_data import CLINICAL_PATIENTS
import uuid
import random

class ClinicalTriageTask:
    TASK_ID = "clinical_triage_classification"

    def __init__(self):
        self.current_idx = 0
        self.patients = []
        self.results = []

    def reset(self) -> Observation:
        self.current_idx = 0
        self.results = []
        self.patients = random.sample(CLINICAL_PATIENTS, min(10, len(CLINICAL_PATIENTS)))
        return self._make_obs()

    def step(self, action: Action):
        if self.current_idx >= len(self.patients):
            obs = Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
            return obs, Reward(total=0.0), True, {}
        current = self.patients[self.current_idx]
        score = 1.0 if action.body_system == current["true_body_system"] else 0.0
        self.results.append({"score": score})
        self.current_idx += 1
        done = self.current_idx >= len(self.patients)
        return self._make_obs(done), Reward(total=score), done, {}

    def state(self): return {"task_id": self.TASK_ID, "step": self.current_idx}
    
    def grader_score(self):
        total = sum(r["score"] for r in self.results) / len(self.results) if self.results else 0.0
        return {"task_id": self.TASK_ID, "final_score": max(0.0001, min(0.9999, total)), "passed": total >= 0.7}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.patients):
            return Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
        c = self.patients[self.current_idx]
        patient_obj = ClinicalPatient(patient_id=c["patient_id"], age=c["age"], gender=c["gender"], chief_complaint=c["chief_complaint"], vitals=c["vitals"], history=c["history"])
        return Observation(task_id=self.TASK_ID, step=self.current_idx, current_patient=patient_obj, valid_actions=[AgentAction.CLASSIFY_TRIAGE])


class ClinicalESITask:
    TASK_ID = "clinical_esi_assignment"

    def __init__(self):
        self.current_idx = 0
        self.patients = []
        self.results = []

    def reset(self) -> Observation:
        self.current_idx = 0
        self.results = []
        self.patients = random.sample(CLINICAL_PATIENTS, min(10, len(CLINICAL_PATIENTS)))
        return self._make_obs()

    def step(self, action: Action):
        if self.current_idx >= len(self.patients):
            obs = Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
            return obs, Reward(total=0.0), True, {}
        current = self.patients[self.current_idx]
        score = 1.0 if action.esi_level == current["true_esi_level"] else (0.5 if action.esi_level and abs(action.esi_level - current["true_esi_level"]) == 1 else 0.0)
        self.results.append({"score": score})
        self.current_idx += 1
        done = self.current_idx >= len(self.patients)
        return self._make_obs(done), Reward(total=score), done, {}

    def state(self): return {"task_id": self.TASK_ID, "step": self.current_idx}
    
    def grader_score(self):
        total = sum(r["score"] for r in self.results) / len(self.results) if self.results else 0.0
        return {"task_id": self.TASK_ID, "final_score": max(0.0001, min(0.9999, round(total, 4))), "passed": total >= 0.6}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.patients):
            return Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
        c = self.patients[self.current_idx]
        patient_obj = ClinicalPatient(patient_id=c["patient_id"], age=c["age"], gender=c["gender"], chief_complaint=c["chief_complaint"], vitals=c["vitals"], history=c["history"])
        return Observation(task_id=self.TASK_ID, step=self.current_idx, current_patient=patient_obj, valid_actions=[AgentAction.ASSIGN_ESI])


class ClinicalNoteTask:
    TASK_ID = "clinical_triage_note"

    def __init__(self):
        self.current_idx = 0
        self.patients = []
        self.results = []

    def reset(self) -> Observation:
        self.current_idx = 0
        self.results = []
        self.patients = random.sample(CLINICAL_PATIENTS, min(5, len(CLINICAL_PATIENTS)))
        return self._make_obs()

    def step(self, action: Action):
        if self.current_idx >= len(self.patients):
            obs = Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
            return obs, Reward(total=0.0), True, {}
        current = self.patients[self.current_idx]
        score = 0.0
        if action.triage_note and len(action.triage_note) > 30:
            score += 0.5
            if current["true_esi_level"] in [1, 2] and any(w in action.triage_note.lower() for w in ["stat", "immediate", "iv", "oxygen", "monitor", "resuscitation", "acute"]):
                score += 0.5
            elif current["true_esi_level"] >= 3:
                score += 0.5
                
        self.results.append({"score": score})
        self.current_idx += 1
        done = self.current_idx >= len(self.patients)
        return self._make_obs(done), Reward(total=score), done, {}

    def state(self): return {"task_id": self.TASK_ID, "step": self.current_idx}
    
    def grader_score(self):
        total = sum(r["score"] for r in self.results) / len(self.results) if self.results else 0.0
        return {"task_id": self.TASK_ID, "final_score": max(0.0001, min(0.9999, total)), "passed": total >= 0.6}

    def _make_obs(self, done=False):
        if done or self.current_idx >= len(self.patients):
            return Observation(task_id=self.TASK_ID, step=self.current_idx, episode_done=True)
        c = self.patients[self.current_idx]
        patient_obj = ClinicalPatient(patient_id=c["patient_id"], age=c["age"], gender=c["gender"], chief_complaint=c["chief_complaint"], vitals=c["vitals"], history=c["history"], true_esi_level=c["true_esi_level"])
        return Observation(task_id=self.TASK_ID, step=self.current_idx, current_patient=patient_obj, valid_actions=[AgentAction.WRITE_TRIAGE_NOTE])
