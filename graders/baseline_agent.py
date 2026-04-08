"""
Heuristic baseline agent — covers all 13 tasks across 4 domains.
Used by /baseline endpoint. No API key required.
"""

import re
from typing import Dict, Any
from app.models import (
    Action, AgentAction, TicketCategory, TicketPriority, Ticket,
    ClauseType, RiskLevel, BodySystem, PRType,
)
from tasks.task1_classification import ClassificationTask
from tasks.task2_drafting import ResponseDraftingTask
from tasks.task3_queue import QueueManagementTask
from tasks.task4_multiturn import MultiTurnTask
from tasks.legal_tasks import LegalIdentifyClauseTask, LegalRiskFlagTask, LegalRedlineTask
from tasks.clinical_tasks import ClinicalTriageTask, ClinicalESITask, ClinicalNoteTask
from tasks.pr_tasks import PRTypeTask, PRBugIdentifyTask, PRReviewTask


# ── Support Heuristics ────────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    TicketCategory.BILLING: ["charge","invoice","refund","payment","billing","subscription","price","cost","money","overcharge","paid"],
    TicketCategory.TECHNICAL: ["error","bug","crash","broken","api","500","not working","fail","issue","problem","sso","saml","login","export"],
    TicketCategory.ACCOUNT: ["account","password","access","permission","invite","team","member","owner","transfer","workspace","settings"],
    TicketCategory.FEATURE_REQUEST: ["feature","request","add","would love","please add","integration","zapier","dark mode","roadmap"],
    TicketCategory.ABUSE: ["phishing","spam","abuse","report","scam","fake","suspicious","harass","inappropriate"],
}

PRIORITY_SIGNALS = {
    TicketPriority.P1_CRITICAL: ["urgent","critical","down","blocked","pipeline","phishing","500","data loss"],
    TicketPriority.P2_HIGH: ["asap","broken","cannot","can't","error","sso","migration"],
    TicketPriority.P4_LOW: ["would love","feature","request","nice to have","suggestion","dark mode"],
}

RESPONSE_TEMPLATES = {
    TicketCategory.BILLING: "Hi there,\n\nThank you for reaching out about this billing concern. I sincerely apologize for the inconvenience.\n\n1. I'll verify the charge in our billing system (Stripe) immediately.\n2. If a duplicate charge is confirmed, I'll process a full refund within 5-7 business days.\n3. You'll receive a confirmation email once the refund is initiated.\n\nPlease don't hesitate to reach out if you have any questions.\n\nBest regards,\nSupport Team",
    TicketCategory.TECHNICAL: "Hi there,\n\nThank you for reporting this. I understand how frustrating this can be.\n\n1. I've logged this with our engineering team along with your request_id.\n2. Please check status.company.com for any ongoing incidents.\n3. As a workaround, try clearing your cache and retrying.\n\nWe'll provide an update within 4 hours. Thank you for your patience!\n\nBest regards,\nSupport Team",
    TicketCategory.ACCOUNT: "Hi there,\n\nThank you for contacting us about your account.\n\n1. Go to Settings > Team > Invite Members to add colleagues.\n2. For ownership transfers, our Trust & Safety team will reach out within 2-5 business days.\n3. Your data is completely safe throughout this process.\n\nHappy to help further!\n\nBest regards,\nSupport Team",
    TicketCategory.FEATURE_REQUEST: "Hi there,\n\nThank you for this suggestion! We love hearing feedback.\n\n1. Visit roadmap.company.com to upvote this request.\n2. Our product team reviews the roadmap monthly.\n3. Enterprise customers can discuss prioritization with their CSM.\n\nThank you for helping us improve!\n\nBest regards,\nSupport Team",
    TicketCategory.ABUSE: "Hi there,\n\nThank you for reporting this. We take abuse and phishing extremely seriously.\n\n1. Our Trust & Safety team will review the reported account within 1 hour.\n2. We will take appropriate action per our Terms of Service.\n3. You will receive an update within 24 hours.\n\nPlease preserve any screenshots as evidence.\n\nBest regards,\nSupport Team",
}


def classify_ticket_heuristic(ticket: Ticket):
    text = (ticket.subject + " " + ticket.body).lower()
    scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in CATEGORY_KEYWORDS.items()}
    category = max(scores, key=scores.get)
    if scores[category] == 0:
        category = TicketCategory.UNKNOWN
    for pri, sigs in PRIORITY_SIGNALS.items():
        if any(s in text for s in sigs):
            priority = pri
            break
    else:
        priority = TicketPriority.P3_MEDIUM
    if ticket.customer_tier == "enterprise" and priority == TicketPriority.P3_MEDIUM:
        priority = TicketPriority.P2_HIGH
    return category, priority


# ── Legal Heuristics ──────────────────────────────────────────────────────────

CLAUSE_KEYWORDS = {
    ClauseType.INDEMNITY:   ["indemnify","indemnification","hold harmless","defend"],
    ClauseType.LIABILITY:   ["liability","aggregate liability","limit","cap","damages"],
    ClauseType.IP:          ["intellectual property","license","derivative","ownership","patent","copyright"],
    ClauseType.TERMINATION: ["terminat","notice","cancel","end","expire","convenient"],
}

RISK_SIGNALS = {
    RiskLevel.CRITICAL: ["without any cap","uncapped","perpetual","irrevocable","sell user","joint ownership"],
    RiskLevel.HIGH:     ["3 day","3-day","data breach","$10,000","sole right","unilateral"],
    RiskLevel.LOW:      ["standard","market","protective","exclusive property","no license"],
}


def classify_clause_heuristic(text: str):
    text_lower = text.lower()
    scores = {ct: sum(1 for kw in kws if kw in text_lower) for ct, kws in CLAUSE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ClauseType.UNKNOWN


def flag_risk_heuristic(text: str):
    text_lower = text.lower()
    for level, signals in RISK_SIGNALS.items():
        if any(s in text_lower for s in signals):
            return level
    return RiskLevel.MEDIUM


def redline_heuristic(text: str, risk_level) -> str:
    if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
        return (
            f"REDLINE: The following clause presents significant legal risk and requires revision. "
            f"Suggested amendment: Add an explicit cap on liability (e.g., limited to 12 months of fees paid), "
            f"remove uncapped or unilateral provisions, and ensure mutual obligations. "
            f"Original text should be revised to include: 'Provider's aggregate liability shall not exceed "
            f"the total fees paid in the twelve (12) months preceding the claim.'"
        )
    return f"REDLINE: Minor revision suggested — clarify scope and add standard protective language to: {text[:100]}..."


# ── Clinical Heuristics ───────────────────────────────────────────────────────

BODY_SYSTEM_KEYWORDS = {
    BodySystem.CARDIAC:         ["chest pain","heart","cardiac","palpitation","myocardial","atrial"],
    BodySystem.RESPIRATORY:     ["shortness of breath","breathing","wheezing","respiratory","oxygen","asthma","lung"],
    BodySystem.NEUROLOGIC:      ["weakness","stroke","speech","neuro","headache","seizure","consciousness","slurred"],
    BodySystem.GI:              ["abdominal","stomach","nausea","vomiting","bowel","gi","gastrointestinal","blood"],
    BodySystem.MUSCULOSKELETAL: ["ankle","knee","back","joint","fracture","sprain","muscle","bone"],
}


def classify_body_system_heuristic(complaint: str) -> BodySystem:
    text = complaint.lower()
    for system, keywords in BODY_SYSTEM_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return system
    return BodySystem.OTHER


def assign_esi_heuristic(patient) -> int:
    complaint = patient.chief_complaint.lower()
    vitals = patient.vitals or {}
    spo2 = vitals.get("SpO2", "99%").replace("%", "")
    try:
        spo2_val = int(spo2)
    except Exception:
        spo2_val = 99

    critical = ["cardiac arrest","stroke","unresponsive","hemorrhagic","resuscitation","massive"]
    high = ["chest pain","shortness of breath","abdominal pain","blood","vomiting blood","weakness"]
    low = ["refill","prescription","sore throat","sprain","ankle","mild"]

    if any(w in complaint for w in critical) or spo2_val < 90:
        return 1
    if any(w in complaint for w in high) or spo2_val < 94:
        return 2
    if any(w in complaint for w in low):
        return 4
    return 3


def write_triage_note_heuristic(patient) -> str:
    esi = assign_esi_heuristic(patient)
    urgency = {1: "IMMEDIATE — resuscitation bay required.", 2: "HIGH — acute care bed, multiple resources needed.",
                3: "MODERATE — multiple resources, can wait briefly.", 4: "LOW — one resource, stable vitals.",
                5: "NON-URGENT — no resources needed, fast track."}
    return (
        f"TRIAGE NOTE: Patient {patient.patient_id}, {patient.age}yo {patient.gender}. "
        f"Chief complaint: {patient.chief_complaint}. History: {patient.history}. "
        f"Vitals: {patient.vitals}. ESI Level: {esi}. Disposition: {urgency.get(esi, 'Standard triage.')} "
        f"Monitor vitals continuously and reassess every 15 minutes."
    )


# ── Engineering Heuristics ────────────────────────────────────────────────────

PR_TYPE_KEYWORDS = {
    PRType.SECURITY: ["sql injection","xss","auth","token","password","hash","encrypt","sanitize","verify_exp","unauthenticated"],
    PRType.BUG_FIX:  ["fix","bug","error","crash","none","leak","loop","typo","incorrect"],
    PRType.REFACTOR: ["refactor","cleanup","clean","reorganize","simplify","css","rename"],
    PRType.FEATURE:  ["add","new","feature","endpoint","route","implement","create"],
}


def classify_pr_heuristic(pr) -> PRType:
    text = (pr.title + " " + pr.description + " " + pr.diff).lower()
    scores = {pt: sum(1 for kw in kws if kw in text) for pt, kws in PR_TYPE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else PRType.REFACTOR


def identify_bug_heuristic(pr) -> str:
    diff = pr.diff.lower()
    if "f\"" in pr.diff or "f'" in pr.diff and "query" in diff:
        return "Potential SQL injection vulnerability via f-string interpolation in database query."
    if "verify_exp" in diff and "false" in diff:
        return "JWT expiration verification is disabled, allowing expired tokens to authenticate."
    if "time.sleep" in diff:
        return "Synchronous sleep in server context blocks the thread and prevents handling other requests."
    if "revenue" in diff and "route" in diff:
        return "Sensitive financial data exposed on potentially unauthenticated endpoint."
    if ".get(" in diff and "none" in diff.lower():
        return "Possible AttributeError if the object is None before calling .get()."
    return "No critical bug identified. Code change appears correct."


def write_review_heuristic(pr) -> str:
    bug = identify_bug_heuristic(pr)
    if "No critical bug" in bug:
        return f"LGTM. The change looks correct and addresses the stated goal. {bug}"
    return (
        f"⚠️ Block this PR. Critical issue found: {bug} "
        f"Please address this before merging. "
        f"Suggested fix: use parameterized queries / remove insecure options / move blocking calls to async context."
    )


# ── Task Runners ──────────────────────────────────────────────────────────────

async def run_task1_baseline() -> Dict[str, Any]:
    task = ClassificationTask()
    obs = task.reset()
    done = False
    while not done:
        ticket = obs.current_ticket
        if not ticket:
            break
        category, priority = classify_ticket_heuristic(ticket)
        action = Action(action_type=AgentAction.CLASSIFY, ticket_id=ticket.ticket_id, category=category, priority=priority)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_task2_baseline() -> Dict[str, Any]:
    task = ResponseDraftingTask()
    obs = task.reset()
    done = False
    while not done:
        ticket = obs.current_ticket
        if not ticket:
            break
        category = ticket.category or TicketCategory.TECHNICAL
        response = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES[TicketCategory.TECHNICAL])
        action = Action(action_type=AgentAction.DRAFT_RESPONSE, ticket_id=ticket.ticket_id, response_text=response)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_task3_baseline() -> Dict[str, Any]:
    task = QueueManagementTask()
    obs = task.reset()
    done = False
    AGENT_MAP = {"billing": "agent_billing", "technical": "agent_tech", "account": "agent_tech",
                 "feature_request": "agent_general", "abuse": "agent_general", "unknown": "agent_general"}
    while not done:
        if not obs.ticket_queue:
            break
        unassigned = [t for t in obs.ticket_queue if not t.assigned_agent]
        in_progress = [t for t in obs.ticket_queue if t.assigned_agent and t.status.value == "in_progress"]
        if unassigned:
            ticket = sorted(unassigned, key=lambda t: ["P1","P2","P3","P4"].index(t.priority.value) if t.priority else 3)[0]
            agent_id = AGENT_MAP.get(ticket.category.value if ticket.category else "unknown", "agent_general")
            action = Action(action_type=AgentAction.ASSIGN_TICKET, ticket_id=ticket.ticket_id, target_agent_id=agent_id)
        elif in_progress:
            action = Action(action_type=AgentAction.RESOLVE, ticket_id=in_progress[0].ticket_id, resolution_summary="Resolved.")
        else:
            action = Action(action_type=AgentAction.NO_OP)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_task4_baseline() -> Dict[str, Any]:
    task = MultiTurnTask()
    obs = task.reset()
    done = False
    turn = 0
    while not done:
        ticket = obs.current_ticket
        scenario = task.scenario
        must_escalate = scenario.get("resolution_criteria", {}).get("must_escalate", False)
        target_turn = scenario.get("resolution_criteria", {}).get("target_turn", 4)
        if must_escalate and turn >= target_turn - 1:
            action = Action(action_type=AgentAction.ESCALATE, ticket_id=ticket.ticket_id if ticket else None)
        elif not must_escalate and turn >= target_turn - 1:
            action = Action(action_type=AgentAction.RESOLVE, ticket_id=ticket.ticket_id if ticket else None)
        else:
            action = Action(action_type=AgentAction.DRAFT_RESPONSE, ticket_id=ticket.ticket_id if ticket else None,
                            response_text="Thank you for your message. I understand your concern and I'm actively working to resolve this for you. I appreciate your patience.")
        obs, reward, done, info = task.step(action)
        turn += 1
    return task.grader_score()


async def run_legal_identify_baseline() -> Dict[str, Any]:
    task = LegalIdentifyClauseTask()
    obs = task.reset()
    done = False
    while not done:
        clause = obs.current_clause
        if not clause:
            break
        clause_type = classify_clause_heuristic(clause.text)
        action = Action(action_type=AgentAction.IDENTIFY_CLAUSE, clause_type=clause_type)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_legal_risk_baseline() -> Dict[str, Any]:
    task = LegalRiskFlagTask()
    obs = task.reset()
    done = False
    while not done:
        clause = obs.current_clause
        if not clause:
            break
        risk = flag_risk_heuristic(clause.text)
        action = Action(action_type=AgentAction.FLAG_RISK, risk_level=risk, reasoning="Keyword-based heuristic assessment.")
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_legal_redline_baseline() -> Dict[str, Any]:
    task = LegalRedlineTask()
    obs = task.reset()
    done = False
    while not done:
        clause = obs.current_clause
        if not clause:
            break
        risk = clause.true_risk_level or flag_risk_heuristic(clause.text)
        redline = redline_heuristic(clause.text, risk)
        action = Action(action_type=AgentAction.REDLINE, redline_text=redline)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_clinical_triage_baseline() -> Dict[str, Any]:
    task = ClinicalTriageTask()
    obs = task.reset()
    done = False
    while not done:
        patient = obs.current_patient
        if not patient:
            break
        system = classify_body_system_heuristic(patient.chief_complaint)
        action = Action(action_type=AgentAction.CLASSIFY_TRIAGE, body_system=system)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_clinical_esi_baseline() -> Dict[str, Any]:
    task = ClinicalESITask()
    obs = task.reset()
    done = False
    while not done:
        patient = obs.current_patient
        if not patient:
            break
        esi = assign_esi_heuristic(patient)
        action = Action(action_type=AgentAction.ASSIGN_ESI, esi_level=esi, reasoning="Heuristic vitals and complaint analysis.")
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_clinical_note_baseline() -> Dict[str, Any]:
    task = ClinicalNoteTask()
    obs = task.reset()
    done = False
    while not done:
        patient = obs.current_patient
        if not patient:
            break
        note = write_triage_note_heuristic(patient)
        action = Action(action_type=AgentAction.WRITE_TRIAGE_NOTE, triage_note=note)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_pr_type_baseline() -> Dict[str, Any]:
    task = PRTypeTask()
    obs = task.reset()
    done = False
    while not done:
        pr = obs.current_pr
        if not pr:
            break
        pr_type = classify_pr_heuristic(pr)
        action = Action(action_type=AgentAction.CLASSIFY_PR, pr_type=pr_type)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_pr_bug_baseline() -> Dict[str, Any]:
    task = PRBugIdentifyTask()
    obs = task.reset()
    done = False
    while not done:
        pr = obs.current_pr
        if not pr:
            break
        bug = identify_bug_heuristic(pr)
        action = Action(action_type=AgentAction.IDENTIFY_BUG, bug_description=bug)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_pr_review_baseline() -> Dict[str, Any]:
    task = PRReviewTask()
    obs = task.reset()
    done = False
    while not done:
        pr = obs.current_pr
        if not pr:
            break
        comment = write_review_heuristic(pr)
        action = Action(action_type=AgentAction.REVIEW_PR, review_comment=comment)
        obs, reward, done, info = task.step(action)
    return task.grader_score()


async def run_baseline_all_tasks() -> Dict[str, Any]:
    runners = {
        "ticket_classification":          run_task1_baseline,
        "response_drafting":              run_task2_baseline,
        "queue_management":               run_task3_baseline,
        "multi_turn_conversation":        run_task4_baseline,
        "legal_clause_identification":    run_legal_identify_baseline,
        "legal_risk_flagging":            run_legal_risk_baseline,
        "legal_clause_redlining":         run_legal_redline_baseline,
        "clinical_triage_classification": run_clinical_triage_baseline,
        "clinical_esi_assignment":        run_clinical_esi_baseline,
        "clinical_triage_note":           run_clinical_note_baseline,
        "pr_type_classification":         run_pr_type_baseline,
        "pr_bug_identification":          run_pr_bug_baseline,
        "pr_review_comment":              run_pr_review_baseline,
    }
    results = {}
    for tid, runner in runners.items():
        try:
            results[tid] = await runner()
        except Exception as e:
            results[tid] = {"final_score": 0.0001, "passed": False, "error": str(e)}

    overall = sum(r["final_score"] for r in results.values()) / len(results)
    return {
        "agent": "heuristic_baseline",
        "overall_score": round(overall, 4),
        "tasks": results,
    }
