#!/usr/bin/env python3
"""OpenEnv pre-submission validation — checks all 13 tasks."""

import sys, json, argparse
import urllib.request

def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())

def post(url, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

def check(label, condition, detail=""):
    status = "✓ PASS" if condition else "✗ FAIL"
    print(f"  {status}  {label}" + (f"\n         {detail}" if detail else ""))
    return condition

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"\n{'='*60}\nOpenEnv Validator — {base}\n{'='*60}\n")
    results = []

    # 1. Health
    try:
        h = get(f"{base}/health")
        results.append(check("Server responds (GET /health)", h.get("status") == "ok"))
    except Exception as e:
        results.append(check("Server responds", False, str(e)))

    # 2. Tasks endpoint
    try:
        t = get(f"{base}/tasks")
        tasks = t.get("tasks", [])
        results.append(check("GET /tasks returns 3+ tasks", len(tasks) >= 3, f"found {len(tasks)} tasks"))
        results.append(check("Tasks have action_schema", all("action_schema" in t for t in tasks)))
        results.append(check("Tasks have difficulty range",
            any(t["difficulty"] == "easy" for t in tasks) and
            any(t["difficulty"] == "hard" for t in tasks)))
    except Exception as e:
        results.append(check("GET /tasks", False, str(e)))

    # 3. Reset — test all 13 tasks
    all_task_ids = [
        "ticket_classification", "response_drafting", "queue_management",
        "multi_turn_conversation", "legal_clause_identification", "legal_risk_flagging",
        "legal_clause_redlining", "clinical_triage_classification", "clinical_esi_assignment",
        "clinical_triage_note", "pr_type_classification", "pr_bug_identification", "pr_review_comment",
    ]
    # Only deeply test core 3 to keep validation fast
    core_task_ids = ["ticket_classification", "response_drafting", "queue_management"]

    for tid in all_task_ids:
        try:
            r = post(f"{base}/reset?task_id={tid}")
            obs = r.get("observation", {})
            results.append(check(f"POST /reset ({tid})",
                "task_id" in obs and "step" in obs and "valid_actions" in obs))
        except Exception as e:
            results.append(check(f"POST /reset ({tid})", False, str(e)))

    # 4. Step
    try:
        post(f"{base}/reset?task_id=ticket_classification")
        step_result = post(f"{base}/step?task_id=ticket_classification",
            {"action_type": "classify", "category": "billing", "priority": "P3"})
        results.append(check("POST /step returns observation+reward+done",
            all(k in step_result for k in ["observation","reward","done"])))
        total = step_result.get("reward", {}).get("total")
        results.append(check("Reward.total in [-1.0, 1.0]",
            total is not None and -1.0 <= total <= 1.0, f"total={total}"))
    except Exception as e:
        results.append(check("POST /step", False, str(e)))

    # 5. State
    try:
        s = get(f"{base}/state?task_id=ticket_classification")
        results.append(check("GET /state returns task_id", "task_id" in s))
    except Exception as e:
        results.append(check("GET /state", False, str(e)))

    # 6. Grader — core tasks
    for tid in core_task_ids:
        try:
            post(f"{base}/reset?task_id={tid}")
            g = post(f"{base}/grader?task_id={tid}")
            score = g.get("final_score", -1)
            results.append(check(f"POST /grader ({tid}) score in [0,1]",
                0.0 <= score <= 1.0, f"score={score}"))
        except Exception as e:
            results.append(check(f"POST /grader ({tid})", False, str(e)))

    # 7. Baseline
    try:
        b = post(f"{base}/baseline")
        results.append(check("POST /baseline returns overall_score", "overall_score" in b))
        results.append(check("POST /baseline covers all 13 tasks",
            all(tid in b.get("tasks", {}) for tid in all_task_ids)))
    except Exception as e:
        results.append(check("POST /baseline", False, str(e)))

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("🏆 ALL CHECKS PASSED — ready to submit!")
    elif passed >= total * 0.8:
        print("⚠️  Most checks passed — review failures above.")
    else:
        print("❌  Multiple failures — fix before submitting.")
    print(f"{'='*60}\n")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
