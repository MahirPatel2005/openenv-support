"""
Realistic synthetic ticket dataset for the Customer Support Triage environment.
All tickets are designed to reflect real SaaS support scenarios.
"""

from typing import List
from app.models import Ticket, TicketCategory, TicketPriority, TicketStatus
from datetime import datetime, timedelta
import random
import uuid


def _future(hours: int) -> str:
    return (datetime.utcnow() + timedelta(hours=hours)).isoformat() + "Z"


def _past(hours: int) -> str:
    return (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"


TICKET_POOL: List[dict] = [
    # ── BILLING ──
    {
        "subject": "Double charged for subscription this month",
        "body": (
            "Hi, I just checked my bank statement and noticed I was charged twice for my Pro plan "
            "on March 15th. Both transactions show $49.99 from your company. I only have one account "
            "and never upgraded or changed my plan. Please refund the duplicate charge immediately. "
            "My account email is james.h@acme.io"
        ),
        "customer_tier": "pro",
        "category": TicketCategory.BILLING,
        "priority": TicketPriority.P2_HIGH,
        "sentiment_score": -0.7,
        "tags": ["duplicate-charge", "refund"],
    },
    {
        "subject": "Invoice shows wrong company name",
        "body": (
            "Our accountant flagged that the last three invoices show 'Acme Corp' instead of "
            "'Acme Corporation LLC'. This is causing issues with our bookkeeping. Can you please "
            "reissue invoices #INV-2024-089, #INV-2024-090, and #INV-2024-091 with the correct name?"
        ),
        "customer_tier": "enterprise",
        "category": TicketCategory.BILLING,
        "priority": TicketPriority.P3_MEDIUM,
        "sentiment_score": -0.2,
        "tags": ["invoice", "company-name"],
    },
    {
        "subject": "Need annual plan pricing for 50 seats",
        "body": (
            "We're evaluating your platform for our team of 50 users. Can you share the annual "
            "pricing for the Business tier? Also, do you offer non-profit discounts? We're a 501(c)(3) "
            "organization and budget is tight."
        ),
        "customer_tier": "free",
        "category": TicketCategory.BILLING,
        "priority": TicketPriority.P4_LOW,
        "sentiment_score": 0.3,
        "tags": ["pricing", "annual-plan", "non-profit"],
    },
    # ── TECHNICAL ──
    {
        "subject": "API returning 500 errors on /v2/exports endpoint",
        "body": (
            "Since around 14:00 UTC today, our integration is getting 500 Internal Server Error "
            "responses from POST /v2/exports. This is blocking our nightly data pipeline and affecting "
            "3 downstream systems. Error response: {'error': 'Internal server error', 'request_id': 'req_8f2k1'}. "
            "We've verified our payload matches the docs. This is urgent — SLA breach in 2 hours."
        ),
        "customer_tier": "enterprise",
        "category": TicketCategory.TECHNICAL,
        "priority": TicketPriority.P1_CRITICAL,
        "sentiment_score": -0.85,
        "tags": ["api", "500-error", "pipeline", "urgent"],
    },
    {
        "subject": "SSO login broken after domain migration",
        "body": (
            "We migrated our company domain from oldco.com to newco.io last week. Now none of our "
            "team can log in via SSO. The error is 'Invalid SAML assertion'. I've updated the ACS URL "
            "in our IdP (Okta) but still broken. Our SSO config shows the old domain in 'Entity ID'. "
            "Can you help update this without losing our team's data?"
        ),
        "customer_tier": "pro",
        "category": TicketCategory.TECHNICAL,
        "priority": TicketPriority.P2_HIGH,
        "sentiment_score": -0.5,
        "tags": ["sso", "saml", "okta", "domain-migration"],
    },
    {
        "subject": "Export to CSV missing some columns",
        "body": (
            "When I export my project data to CSV, the 'custom fields' columns are missing. "
            "I have 4 custom fields set up (Status, Owner, Due Quarter, Budget). The export only "
            "includes the default columns. Is this a known issue? Chrome 122, macOS Sonoma."
        ),
        "customer_tier": "pro",
        "category": TicketCategory.TECHNICAL,
        "priority": TicketPriority.P3_MEDIUM,
        "sentiment_score": -0.3,
        "tags": ["csv-export", "custom-fields", "bug"],
    },
    {
        "subject": "Mobile app crashes on iOS 17.4 when opening notifications",
        "body": (
            "Ever since updating to iOS 17.4, the app crashes every time I tap on a push notification. "
            "It also crashes occasionally when switching between the Dashboard and Reports tabs. "
            "App version: 3.2.1. Crash happens 100% of the time. I've tried reinstalling."
        ),
        "customer_tier": "free",
        "category": TicketCategory.TECHNICAL,
        "priority": TicketPriority.P3_MEDIUM,
        "sentiment_score": -0.4,
        "tags": ["mobile", "ios", "crash", "notifications"],
    },
    # ── ACCOUNT ──
    {
        "subject": "Need to transfer account ownership — founder left company",
        "body": (
            "Our co-founder who created the account has left the company. We need to transfer "
            "full account ownership to me (current CEO). I have documentation (board resolution) "
            "proving I have authority to make this request. The original owner's email is "
            "ex-founder@company.com and my email is ceo@company.com."
        ),
        "customer_tier": "enterprise",
        "category": TicketCategory.ACCOUNT,
        "priority": TicketPriority.P2_HIGH,
        "sentiment_score": -0.1,
        "tags": ["ownership-transfer", "account-admin"],
    },
    {
        "subject": "How do I add team members to my workspace?",
        "body": "Hi! I just upgraded to Pro and want to invite my 3 colleagues. Where do I find the team invite option? I looked in Settings but can't find it.",
        "customer_tier": "pro",
        "category": TicketCategory.ACCOUNT,
        "priority": TicketPriority.P4_LOW,
        "sentiment_score": 0.2,
        "tags": ["how-to", "team-invite"],
    },
    # ── FEATURE REQUEST ──
    {
        "subject": "Please add Zapier integration",
        "body": (
            "We use Zapier extensively to connect our tools. Having a native Zapier integration "
            "would save us hours every week. Specifically we need: new ticket triggers, update "
            "ticket actions, and create report actions. This would be a dealbreaker for our renewal "
            "if not on the roadmap."
        ),
        "customer_tier": "pro",
        "category": TicketCategory.FEATURE_REQUEST,
        "priority": TicketPriority.P3_MEDIUM,
        "sentiment_score": 0.0,
        "tags": ["zapier", "integration", "automation"],
    },
    {
        "subject": "Dark mode request",
        "body": "Would love a dark mode option! Working late nights and the bright interface is hard on my eyes. Many of us in the community forums are asking for this too.",
        "customer_tier": "free",
        "category": TicketCategory.FEATURE_REQUEST,
        "priority": TicketPriority.P4_LOW,
        "sentiment_score": 0.4,
        "tags": ["dark-mode", "ui"],
    },
    # ── ABUSE ──
    {
        "subject": "Reporting spam/phishing from another user",
        "body": (
            "I received a message through your platform from user ID user_7x92k that contains "
            "a phishing link trying to steal my login credentials. The message asks me to 'verify my "
            "account' at a suspicious URL. I've screenshotted everything. Please investigate and "
            "remove this user immediately."
        ),
        "customer_tier": "pro",
        "category": TicketCategory.ABUSE,
        "priority": TicketPriority.P1_CRITICAL,
        "sentiment_score": -0.8,
        "tags": ["phishing", "abuse", "safety"],
    },
]

try:
    from datasets import load_dataset
    import random

    # Map bitext intents to real TicketCategory values
    _BITEXT_CATEGORY_MAP = {
        "cancel_order": TicketCategory.BILLING,
        "change_order": TicketCategory.BILLING,
        "check_invoice": TicketCategory.BILLING,
        "check_cancellation_fee": TicketCategory.BILLING,
        "get_invoice": TicketCategory.BILLING,
        "get_refund": TicketCategory.BILLING,
        "payment_issue": TicketCategory.BILLING,
        "refund_request": TicketCategory.BILLING,
        "track_order": TicketCategory.BILLING,
        "track_refund": TicketCategory.BILLING,
        "change_shipping": TicketCategory.ACCOUNT,
        "contact_customer_service": TicketCategory.ACCOUNT,
        "contact_human_agent": TicketCategory.ACCOUNT,
        "create_account": TicketCategory.ACCOUNT,
        "delete_account": TicketCategory.ACCOUNT,
        "edit_account": TicketCategory.ACCOUNT,
        "place_order": TicketCategory.ACCOUNT,
        "recover_password": TicketCategory.ACCOUNT,
        "registration_problems": TicketCategory.TECHNICAL,
        "set_up_shipping": TicketCategory.ACCOUNT,
        "switch_account": TicketCategory.ACCOUNT,
        "delivery_period": TicketCategory.ACCOUNT,
        "complaint": TicketCategory.TECHNICAL,
        "delivery_options": TicketCategory.FEATURE_REQUEST,
        "newsletter_subscription": TicketCategory.FEATURE_REQUEST,
        "review": TicketCategory.FEATURE_REQUEST,
    }

    _BITEXT_DS = list(load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train", streaming=True
    ).take(50))

    for row in _BITEXT_DS:
        intent = row.get("intent", "").lower().strip()
        category = _BITEXT_CATEGORY_MAP.get(intent, TicketCategory.ACCOUNT)
        TICKET_POOL.append({
            "subject": intent.replace("_", " ").title(),
            "body": row.get("instruction", "I need help with my account."),
            "customer_tier": "free",
            "category": category,
            "priority": TicketPriority.P3_MEDIUM,
            "sentiment_score": 0.0,
            "tags": ["hf-dataset"],
        })
except Exception:
    pass

DIFFICULTY_VARIANTS = {
    "ambiguous": "I have a problem with my account or maybe billing not sure. ",
    "angry": "THIS IS ABSOLUTELY UNACCEPTABLE I WANT MY MONEY BACK NOW!!! ",
    "vague": "It's not working properly but anyway: ",
    "multilingual": "Hola, tengo un problema. ",
    "multi_issue": "Billing is wrong AND I can't login AND the app crashes. Also: ",
}

def generate_ticket(override: dict = None) -> Ticket:
    base = random.choice(TICKET_POOL).copy()
    if override:
        base.update(override)

    # Inject combinatorial difficulty permutations to generate practically infinite variants
    if random.random() < 0.6:  # 60% chance to modify the ticket
        variant = random.choice(list(DIFFICULTY_VARIANTS.keys()))
        prefix = DIFFICULTY_VARIANTS[variant]
        
        if variant == "angry":
            base["subject"] = base["subject"].upper() + "!!!"
            base["sentiment_score"] = -1.0
        elif variant == "multilingual":
            base["subject"] = "Ayuda: " + base["subject"]
            
        base["body"] = prefix + base["body"]

    tid = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    cid = f"CUST-{random.randint(10000, 99999)}"
    created = _past(random.randint(1, 48))

    # SLA based on priority
    sla_hours = {"P1": 4, "P2": 8, "P3": 24, "P4": 72}
    priority_val = base.get("priority", TicketPriority.P3_MEDIUM)
    hours = sla_hours.get(priority_val.value if hasattr(priority_val, "value") else priority_val, 24)
    sla = _future(random.randint(1, hours))

    return Ticket(
        ticket_id=tid,
        subject=base["subject"],
        body=base["body"],
        customer_id=cid,
        customer_tier=base.get("customer_tier", "pro"),
        created_at=created,
        sla_deadline=sla,
        category=base.get("category"),
        priority=base.get("priority"),
        sentiment_score=base.get("sentiment_score", 0.0),
        tags=base.get("tags", []),
    )


def generate_queue(size: int = 20) -> List[Ticket]:
    """Generate a realistic mixed queue of tickets for the hard task."""
    tickets = []
    for _ in range(size):
        t = generate_ticket()
        # Reset category/priority for the hard task (agent must manage, not classify)
        tickets.append(t)
    return tickets
