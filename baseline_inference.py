#!/usr/bin/env python3
"""
Baseline inference script — OpenAI API client against all 3 tasks.
Reads: OPENAI_API_KEY, OPENENV_BASE_URL (default: http://localhost:7860)
Usage:
    python baseline_inference.py
    python baseline_inference.py --task ticket_classification
    python baseline_inference.py --model gpt-4o-mini
"""

import argparse
import asyncio
import json
import os
import sys
import httpx
from openai import OpenAI

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENENV_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are an expert customer support triage AI agent.

CRITICAL RULES — follow these exactly:
1. Always respond with ONLY a valid JSON object. No text before or after it.
2. Never use markdown fences like ```json. Raw JSON only.
3. Keep all string values SHORT. response_text must be under 300 words.

=== TASK: ticket_classification ===
Priority guide (be conservative, most tickets are P3 or P4):
  P1 = production down, data loss, security breach, phishing, pipeline blocked
  P2 = broken feature, SSO failure, cannot login, enterprise customer issue
  P3 = bug with workaround, billing question, account change (DEFAULT for most tickets)
  P4 = feature request, how-to question, cosmetic issue, dark mode, nice-to-have
Action format:
{"action_type": "classify", "ticket_id": "TKT-XXXXX", "category": "billing|technical|account|feature_request|abuse|unknown", "priority": "P1|P2|P3|P4"}

=== TASK: response_drafting ===
Write a SHORT professional response (100-200 words max). Use KB article info.
Action format:
{"action_type": "draft_response", "ticket_id": "TKT-XXXXX", "response_text": "Hi, thank you for reaching out..."}

=== TASK: queue_management ===
STRATEGY — follow this order every step:
  1. If any ticket has assigned_agent AND status=in_progress → RESOLVE it immediately
  2. If any unassigned ticket exists → ASSIGN it to the right agent
  3. Only use close/escalate/no_op if nothing else to do
Agent specializations:
  agent_billing  → billing tickets
  agent_tech     → technical, account tickets
  agent_general  → feature_request, abuse, account tickets
Action format for assign: {"action_type": "assign_ticket", "ticket_id": "TKT-XXXXX", "target_agent_id": "agent_billing|agent_tech|agent_general"}
Action format for resolve: {"action_type": "resolve", "ticket_id": "TKT-XXXXX", "resolution_summary": "Resolved."}
"""


def obs_to_prompt(obs: dict, task_id: str) -> str:
    lines = [f"TASK: {task_id}", f"STEP: {obs.get('step', 0)}"]

    ticket = obs.get("current_ticket")
    if ticket:
        lines += [
            f"\nCURRENT TICKET:",
            f"  ID: {ticket['ticket_id']}",
            f"  Subject: {ticket['subject']}",
            f"  Body: {ticket['body'][:500]}",
            f"  Customer Tier: {ticket['customer_tier']}",
            f"  Sentiment Score: {ticket.get('sentiment_score', 0)}",
        ]
        if ticket.get("category"):
            lines.append(f"  Category: {ticket['category']}")
        if ticket.get("priority"):
            lines.append(f"  Priority: {ticket['priority']}")

    kb = obs.get("knowledge_base", [])
    if kb:
        lines.append("\nKNOWLEDGE BASE:")
        for art in kb[:2]:
            lines += [f"  [{art['article_id']}] {art['title']}", f"  {art['content'][:300]}"]

    queue = obs.get("ticket_queue", [])
    if queue:
        # Separate in-progress (need resolve) from unassigned (need assign)
        in_progress = [t for t in queue if t.get("assigned_agent") and t.get("status") == "in_progress"]
        unassigned = [t for t in queue if not t.get("assigned_agent")]

        if in_progress:
            lines.append(f"\n⚡ IN-PROGRESS (RESOLVE THESE FIRST):")
            for t in in_progress[:3]:
                lines.append(
                    f"  {t['ticket_id']} | {t.get('priority','?')} | agent={t.get('assigned_agent')} | {t['subject'][:60]}"
                )

        if unassigned:
            lines.append(f"\n📋 UNASSIGNED (ASSIGN THESE NEXT):")
            for t in unassigned[:5]:
                lines.append(
                    f"  {t['ticket_id']} | {t.get('priority','?')} | cat={t.get('category','?')} | {t['subject'][:60]}"
                )

    agents = obs.get("agents", [])
    if agents:
        lines.append("\nAGENTS:")
        for a in agents:
            lines.append(
                f"  {a['agent_id']}: load={a['current_load']}/{a['max_load']} "
                f"specialization={a['specialization']}"
            )

    sla = obs.get("sla_status", {})
    if sla:
        breached = [k for k, v in sla.items() if v == "breached"]
        warning = [k for k, v in sla.items() if v == "warning"]
        if breached:
            lines.append(f"\nSLA BREACHED: {breached}")
        if warning:
            lines.append(f"SLA WARNING: {warning}")

    info = obs.get("info", {})
    if info:
        lines.append(f"\nINFO: {json.dumps(info)}")

    lines.append("\nValid actions: " + str(obs.get("valid_actions", [])))
    lines.append("\nRespond with a JSON action object:")

    return "\n".join(lines)


async def run_task(client: OpenAI, task_id: str) -> dict:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as http:
        # Reset
        r = await http.post("/reset", params={"task_id": task_id})
        r.raise_for_status()
        data = r.json()
        obs = data["observation"]

        rewards = []
        step = 0

        while not obs.get("episode_done", False):
            step += 1
            prompt = obs_to_prompt(obs, task_id)

            try:
                # Use more tokens for response_drafting, less for others
                max_tok = 600 if task_id == "response_drafting" else 300

                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tok,
                )
                raw = completion.choices[0].message.content.strip()

                # Strip markdown fences if present
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:].strip()

                # Find the JSON object even if there's surrounding text
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start != -1 and end > start:
                    raw = raw[start:end]

                action_dict = json.loads(raw)

            except Exception as e:
                print(f"  [step {step}] LLM/parse error: {e}. Using no_op.", file=sys.stderr)
                action_dict = {"action_type": "no_op"}

            # Post action
            try:
                r = await http.post("/step", json=action_dict, params={"task_id": task_id})
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                print(f"  [step {step}] Step error: {e}", file=sys.stderr)
                break

            obs = result["observation"]
            rewards.append(result["reward"]["total"])
            print(f"  step {step:02d} | reward={result['reward']['total']:.3f} | done={result['done']}")

        # Get grader score
        r = await http.post("/grader", params={"task_id": task_id})
        r.raise_for_status()
        score = r.json()
        score["reward_history"] = rewards
        return score


async def main():
    parser = argparse.ArgumentParser(description="OpenEnv Customer Support Baseline Inference")
    parser.add_argument("--task", default="all", help="Task ID or 'all'")
    parser.add_argument("--model", default=MODEL, help="OpenAI model to use")
    parser.add_argument("--base-url", default=BASE_URL, help="OpenEnv server URL")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    tasks_to_run = (
        ["ticket_classification", "response_drafting", "queue_management"]
        if args.task == "all"
        else [args.task]
    )

    results = {}
    for task_id in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print(f"{'='*60}")
        result = await run_task(client, task_id)
        results[task_id] = result
        print(f"  ✓ Final score: {result['final_score']:.4f} | Passed: {result['passed']}")
        print(f"    Metrics: {json.dumps(result.get('metrics', {}), indent=2)}")

    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    scores = [r["final_score"] for r in results.values()]
    overall = sum(scores) / len(scores)
    for tid, r in results.items():
        print(f"  {tid:<30} {r['final_score']:.4f}  {'✓ PASS' if r['passed'] else '✗ FAIL'}")
    print(f"  {'OVERALL':<30} {overall:.4f}")

    # Write results to file for reproducibility
    with open("baseline_results.json", "w") as f:
        json.dump({"model": args.model, "overall": overall, "tasks": results}, f, indent=2)
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    asyncio.run(main())
