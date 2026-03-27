"""
Heuristic baseline agent — deterministic, rule-based agent that scores
reproducibly across all 3 tasks. Used by /baseline endpoint and baseline_inference.py.
"""

import re
from typing import Dict, Any, List
from app.models import (
    Action, AgentAction, TicketCategory, TicketPriority, Ticket
)
from tasks.task1_classification import ClassificationTask
from tasks.task2_drafting import ResponseDraftingTask
from tasks.task3_queue import QueueManagementTask


# ── Heuristic Classification Rules ───────────────────────────────────────────

CATEGORY_KEYWORDS = {
    TicketCategory.BILLING: [
        "charge", "invoice", "refund", "payment", "billing", "subscription",
        "price", "pricing", "cost", "money", "overcharge", "paid",
    ],
    TicketCategory.TECHNICAL: [
        "error", "bug", "crash", "broken", "api", "500", "not working",
        "fail", "issue", "problem", "sso", "saml", "login", "export",
    ],
    TicketCategory.ACCOUNT: [
        "account", "password", "access", "permission", "invite", "team",
        "member", "owner", "transfer", "workspace", "settings",
    ],
    TicketCategory.FEATURE_REQUEST: [
        "feature", "request", "add", "would love", "please add", "integration",
        "support for", "zapier", "dark mode", "roadmap",
    ],
    TicketCategory.ABUSE: [
        "phishing", "spam", "abuse", "report", "scam", "fake", "suspicious",
        "harass", "inappropriate",
    ],
}

PRIORITY_SIGNALS = {
    TicketPriority.P1_CRITICAL: ["urgent", "critical", "down", "blocked", "sla breach", "pipeline", "phishing", "500"],
    TicketPriority.P2_HIGH: ["asap", "broken", "cannot", "can't", "error", "sso", "migration"],
    TicketPriority.P4_LOW: ["would love", "feature", "request", "nice to have", "suggestion", "dark mode"],
}


def classify_ticket_heuristic(ticket: Ticket):
    text = (ticket.subject + " " + ticket.body).lower()

    # Category
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in kws if kw in text)
    category = max(scores, key=scores.get)
    if scores[category] == 0:
        category = TicketCategory.UNKNOWN

    # Priority
    for pri, sigs in PRIORITY_SIGNALS.items():
        if any(s in text for s in sigs):
            priority = pri
            break
    else:
        priority = TicketPriority.P3_MEDIUM

    # Enterprise → bump priority
    if ticket.customer_tier == "enterprise" and priority == TicketPriority.P3_MEDIUM:
        priority = TicketPriority.P2_HIGH

    return category, priority


RESPONSE_TEMPLATES = {
    TicketCategory.BILLING: (
        "Thank you for reaching out about this billing concern. I sincerely apologize for "
        "the inconvenience this has caused. I've reviewed your account and I'm happy to help resolve this. "
        "Here's what I'll do:\n\n"
        "1. I'll verify the charge in our billing system immediately.\n"
        "2. If a duplicate charge is confirmed, I'll process a full refund within 5-7 business days "
        "to your original payment method.\n"
        "3. You'll receive a confirmation email once the refund is initiated.\n\n"
        "Please don't hesitate to reach out if you have any questions. We're here to help and want "
        "to make sure this is resolved to your satisfaction."
    ),
    TicketCategory.TECHNICAL: (
        "Thank you for reporting this technical issue. I understand how frustrating it can be when "
        "something isn't working as expected, and I appreciate you providing the details.\n\n"
        "Here are the steps we'll take:\n\n"
        "1. I've logged this issue with our engineering team along with your request_id for investigation.\n"
        "2. Please check status.company.com for any ongoing incidents that may be related.\n"
        "3. Our team will investigate and provide an update within 4 hours for high-priority issues.\n\n"
        "As a workaround, please try clearing your cache and retrying the request. If the issue persists, "
        "please reach out and we'll escalate immediately. Thank you for your patience!"
    ),
    TicketCategory.ACCOUNT: (
        "Thank you for contacting us about your account. I understand this is important and "
        "I'm here to help navigate this process.\n\n"
        "To assist you:\n\n"
        "1. Please go to Settings > Team > Invite Members to add colleagues to your workspace.\n"
        "2. For ownership transfers, our Trust & Safety team will reach out within 2-5 business days "
        "after reviewing the required documentation.\n"
        "3. Rest assured your data is completely safe throughout this process.\n\n"
        "If you have any questions or need further assistance, please don't hesitate to reach out. "
        "We're happy to help!"
    ),
    TicketCategory.FEATURE_REQUEST: (
        "Thank you so much for this suggestion! We genuinely appreciate hearing from customers "
        "about features that would make your experience better.\n\n"
        "Here's how to have the most impact:\n\n"
        "1. Visit roadmap.company.com to see our public roadmap and existing feature requests.\n"
        "2. Upvote this feature request to signal demand to our product team.\n"
        "3. Our product team reviews the roadmap monthly and prioritizes based on community feedback.\n\n"
        "Enterprise customers can also discuss roadmap prioritization with their Customer Success Manager. "
        "Thank you for helping us build a better product!"
    ),
    TicketCategory.ABUSE: (
        "Thank you for bringing this to our attention. We take all reports of abuse and phishing "
        "extremely seriously and I want to assure you we will investigate this immediately.\n\n"
        "Here's what happens next:\n\n"
        "1. Our Trust & Safety team has been notified and will review the reported account within 1 hour.\n"
        "2. Appropriate action will be taken in accordance with our Terms of Service.\n"
        "3. You will receive an update on the outcome within 24 hours.\n\n"
        "Please preserve any screenshots or evidence you have. For your security, do not click any "
        "suspicious links. Thank you for helping keep our platform safe for everyone."
    ),
}


def draft_response_heuristic(ticket: Ticket) -> str:
    category = ticket.category or TicketCategory.TECHNICAL
    template = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES[TicketCategory.TECHNICAL])

    # Personalize with subject
    greeting = f"Hi there,\n\n"
    return greeting + template + f"\n\nBest regards,\nCustomer Support Team\n\nRef: {ticket.ticket_id}"


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
        action = Action(
            action_type=AgentAction.CLASSIFY,
            ticket_id=ticket.ticket_id,
            category=category,
            priority=priority,
            reasoning="Heuristic keyword-based classification",
        )
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
        response = draft_response_heuristic(ticket)
        action = Action(
            action_type=AgentAction.DRAFT_RESPONSE,
            ticket_id=ticket.ticket_id,
            response_text=response,
            reasoning="Heuristic template-based response",
        )
        obs, reward, done, info = task.step(action)

    return task.grader_score()


async def run_task3_baseline() -> Dict[str, Any]:
    task = QueueManagementTask()
    obs = task.reset()
    done = False

    AGENT_MAP = {
        "billing": "agent_billing",
        "technical": "agent_tech",
        "account": "agent_tech",
        "feature_request": "agent_general",
        "abuse": "agent_general",
        "unknown": "agent_general",
    }

    while not done:
        if not obs.ticket_queue:
            break

        action = None
        # Find highest priority unassigned ticket
        unassigned = [t for t in obs.ticket_queue if not t.assigned_agent]
        in_progress = [t for t in obs.ticket_queue if t.assigned_agent and t.status.value == "in_progress"]

        if unassigned:
            ticket = sorted(
                unassigned,
                key=lambda t: ["P1", "P2", "P3", "P4"].index(t.priority.value) if t.priority else 3
            )[0]
            cat = ticket.category.value if ticket.category else "unknown"
            agent_id = AGENT_MAP.get(cat, "agent_general")
            action = Action(
                action_type=AgentAction.ASSIGN_TICKET,
                ticket_id=ticket.ticket_id,
                target_agent_id=agent_id,
            )
        elif in_progress:
            ticket = in_progress[0]
            action = Action(
                action_type=AgentAction.RESOLVE,
                ticket_id=ticket.ticket_id,
                resolution_summary="Resolved via standard procedure.",
            )
        else:
            action = Action(action_type=AgentAction.NO_OP)

        obs, reward, done, info = task.step(action)

    return task.grader_score()


async def run_baseline_all_tasks() -> Dict[str, Any]:
    t1 = await run_task1_baseline()
    t2 = await run_task2_baseline()
    t3 = await run_task3_baseline()

    avg = (t1["final_score"] + t2["final_score"] + t3["final_score"]) / 3

    return {
        "agent": "heuristic_baseline",
        "overall_score": round(avg, 4),
        "tasks": {
            "ticket_classification": t1,
            "response_drafting": t2,
            "queue_management": t3,
        },
    }
