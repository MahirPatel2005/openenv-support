#!/usr/bin/env python3
"""
OpenEnv pre-submission validation script.
Run against a live server: python validate.py --url http://localhost:7860
"""

import sys
import json
import argparse
import urllib.request
import urllib.error

def get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())

def post(url, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
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

    print(f"\n{'='*60}")
    print(f"OpenEnv Validator — {base}")
    print(f"{'='*60}\n")

    results = []

    # 1. Health check
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
            any(t["difficulty"] == "hard" for t in tasks)
        ))
    except Exception as e:
        results.append(check("GET /tasks", False, str(e)))

    # 3. Reset endpoint — all tasks
    task_ids = ["ticket_classification", "response_drafting", "queue_management"]
    for tid in task_ids:
        try:
            r = post(f"{base}/reset?task_id={tid}")
            obs = r.get("observation", {})
            results.append(check(
                f"POST /reset ({tid})",
                "task_id" in obs and "step" in obs and "valid_actions" in obs,
                f"obs keys: {list(obs.keys())[:5]}"
            ))
        except Exception as e:
            results.append(check(f"POST /reset ({tid})", False, str(e)))

    # 4. Step endpoint
    try:
        post(f"{base}/reset?task_id=ticket_classification")
        step_result = post(f"{base}/step?task_id=ticket_classification", {
            "action_type": "classify",
            "category": "billing",
            "priority": "P3",
        })
        results.append(check("POST /step returns observation",
            "observation" in step_result and "reward" in step_result and "done" in step_result))
        reward = step_result.get("reward", {})
        total = reward.get("total", None)
        results.append(check("Reward.total in [-1.0, 1.0]",
            total is not None and -1.0 <= total <= 1.0, f"total={total}"))
    except Exception as e:
        results.append(check("POST /step", False, str(e)))

    # 5. State endpoint
    try:
        s = get(f"{base}/state?task_id=ticket_classification")
        results.append(check("GET /state returns task_id", "task_id" in s))
    except Exception as e:
        results.append(check("GET /state", False, str(e)))

    # 6. Grader endpoint — scores in [0,1]
    for tid in task_ids:
        try:
            post(f"{base}/reset?task_id={tid}")
            g = post(f"{base}/grader?task_id={tid}")
            score = g.get("final_score", -1)
            results.append(check(f"POST /grader ({tid}) score in [0,1]",
                0.0 <= score <= 1.0, f"score={score}"))
        except Exception as e:
            results.append(check(f"POST /grader ({tid})", False, str(e)))

    # 7. Baseline endpoint
    try:
        b = post(f"{base}/baseline")
        results.append(check("POST /baseline returns overall_score", "overall_score" in b))
        results.append(check("POST /baseline has all 3 tasks",
            all(tid in b.get("tasks", {}) for tid in task_ids)))
    except Exception as e:
        results.append(check("POST /baseline", False, str(e)))

    # Summary
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
