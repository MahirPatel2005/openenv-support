"""Knowledge base articles for agent context."""
from app.models import KnowledgeBaseArticle, TicketCategory

KNOWLEDGE_BASE = [
    KnowledgeBaseArticle(
        article_id="KB-001",
        title="How to Request a Refund",
        content=(
            "Refunds are processed within 5-7 business days to the original payment method. "
            "Eligible: duplicate charges (full refund), unused annual plan (prorated), outages > 99.5% SLA (credit). "
            "Process: verify in Stripe, create refund in Admin > Billing > Customer > Refund. "
            "Always send confirmation email with amount and timeline."
        ),
        applicable_categories=[TicketCategory.BILLING],
        relevance_score=0.95,
    ),
    KnowledgeBaseArticle(
        article_id="KB-002",
        title="SSO / SAML Configuration Guide",
        content=(
            "Update SSO: Admin Panel > Security > SSO Configuration. "
            "Entity ID and ACS URL must exactly match IdP config. "
            "Domain migration: update Entity ID, re-download metadata XML, upload to IdP, test first. "
            "Error 'Invalid SAML assertion': Entity ID mismatch or clock skew > 5 min."
        ),
        applicable_categories=[TicketCategory.TECHNICAL, TicketCategory.ACCOUNT],
        relevance_score=0.9,
    ),
    KnowledgeBaseArticle(
        article_id="KB-003",
        title="Account Ownership Transfer",
        content=(
            "Requires: written request from owner OR legal docs if unavailable, identity verification of new owner, "
            "Trust & Safety approval. Timeline: 2-5 business days. Data preserved. "
            "Escalate to account-security@company.com with documentation."
        ),
        applicable_categories=[TicketCategory.ACCOUNT],
        relevance_score=0.95,
    ),
    KnowledgeBaseArticle(
        article_id="KB-004",
        title="API Error Troubleshooting",
        content=(
            "For 500 errors: check status.company.com. If no incident, collect: request_id, endpoint, payload, timestamp. "
            "Escalate to engineering. For enterprise pipeline impact, page on-call immediately. "
            "Provide workaround if available."
        ),
        applicable_categories=[TicketCategory.TECHNICAL],
        relevance_score=0.92,
    ),
    KnowledgeBaseArticle(
        article_id="KB-005",
        title="Abuse and Phishing Handling",
        content=(
            "Steps: (1) acknowledge within 1hr, (2) disable reported account pending investigation, "
            "(3) escalate to Trust & Safety, (4) preserve evidence, (5) notify reporter within 24hr. "
            "Never share investigation details with either party until resolved."
        ),
        applicable_categories=[TicketCategory.ABUSE],
        relevance_score=0.98,
    ),
    KnowledgeBaseArticle(
        article_id="KB-006",
        title="Team Member Invitations",
        content=(
            "Invite: Settings > Team > Invite Members > Enter emails. "
            "Pro: up to 5 members. Business: up to 25. Enterprise: unlimited. "
            "Invites expire after 7 days. Roles: Admin, Editor, Viewer."
        ),
        applicable_categories=[TicketCategory.ACCOUNT],
        relevance_score=0.88,
    ),
    KnowledgeBaseArticle(
        article_id="KB-007",
        title="Feature Request Process",
        content=(
            "Feature requests logged at roadmap.company.com. Customers can upvote or submit. "
            "Enterprise can request acceleration through CSM. "
            "Never promise delivery timelines. Encourage upvoting."
        ),
        applicable_categories=[TicketCategory.FEATURE_REQUEST],
        relevance_score=0.85,
    ),
]


def get_relevant_articles(category, top_k=2):
    relevant = [a for a in KNOWLEDGE_BASE if category in a.applicable_categories]
    relevant.sort(key=lambda x: x.relevance_score, reverse=True)
    return relevant[:top_k]
