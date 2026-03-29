from app.models import PRType

PULL_REQUESTS = [
    {
        "pr_id": "PR-501",
        "title": "Fix memory leak in background worker",
        "description": "The worker was not releasing DB connections back to the pool, exhausting it.",
        "author": "dev_alice",
        "diff": "@@ -24,4 +24,5 @@ def process_job(job):\n     conn = db.get_connection()\n     try:\n         conn.execute('UPDATE jobs SET status = \"done\" WHERE id = ?', (job.id,))\n+    finally:\n+        conn.close()",
        "true_pr_type": PRType.BUG_FIX,
        "true_bug_description": "None. The diff successfully addresses the connection leak by using a finally block.",
        "true_review_comment": "Looks good! Releasing the connection in the finally block guarantees it returns to the pool even if the DB execute throws an exception.",
    },
    {
        "pr_id": "PR-502",
        "title": "Add user profile endpoint",
        "description": "New feature to fetch the logged in user profile data.",
        "author": "junior_bob",
        "diff": "@@ -55,0 +56,3 @@ @app.route('/api/profile')\n+def get_profile():\n+    user_id = request.args.get('user_id')\n+    return db.query(f\"SELECT * FROM users WHERE id = {user_id}\")",
        "true_pr_type": PRType.FEATURE,
        "true_bug_description": "Critical SQL injection vulnerability in f-string query.",
        "true_review_comment": "Block this PR. The database query uses an f-string directly incorporating user input `user_id`. This is a classic SQL injection vector. Please use parameterized queries instead: `db.query('SELECT * FROM users WHERE id = ?', (user_id,))`.",
    },
    {
        "pr_id": "PR-503",
        "title": "Refactor auth middleware",
        "description": "Cleaning up the token verification function to be reusable.",
        "author": "senior_charlie",
        "diff": "@@ -10,8 +10,4 @@ def verify_token(req):\n-    auth_header = req.headers.get('Authorization')\n-    if not auth_header:\n-        return False\n-    token = auth_header.split(' ')[1]\n-    return jwt.decode(token, SECRET, algorithms=['HS256']) is not None\n+    token = req.headers.get('Authorization', '').split('Bearer ')[-1]\n+    return bool(jwt.decode(token, SECRET, algorithms=['HS256'], options={'verify_exp': False}))",
        "true_pr_type": PRType.SECURITY,
        "true_bug_description": "Disabled JWT expiration verification (`verify_exp`: False).",
        "true_review_comment": "Security risk: You have intentionally disabled the expiration verification of the JWT in the decoding options. This allows expired tokens to be accepted indefinitely. Please remove `options={'verify_exp': False}`.",
    },
    {
        "pr_id": "PR-504",
        "title": "Add retry logic to webhook sender",
        "description": "Webhooks occasionally fail, adding exponential backoff.",
        "author": "dev_dave",
        "diff": "@@ -80,0 +81,7 @@ def send_webhook(payload):\n+    import time\n+    for i in range(3):\n+        res = requests.post('https://webhook.site', json=payload)\n+        if res.status_code == 200:\n+            return True\n+        time.sleep(2 ** i)\n+    return False",
        "true_pr_type": PRType.FEATURE,
        "true_bug_description": "Synchronous `time.sleep` in what might be an async or web context blocks the main thread.",
        "true_review_comment": "Using `time.sleep` in the server thread will completely block that worker from handling other requests. We should move this logic to a background async task (like Celery) or use an async sleep `await asyncio.sleep()` depending on the framework.",
    },
    {
        "pr_id": "PR-505",
        "title": "Cleanup CSS classes for buttons",
        "description": "Standardizing button classes across the admin dashboard.",
        "author": "ui_diana",
        "diff": "@@ -120,3 +120,3 @@ <form>\n-    <button class='btn btn-primary submit-btn'>Save</button>\n+    <button class='btn-primary'>Save</button>\n </form>",
        "true_pr_type": PRType.REFACTOR,
        "true_bug_description": "Removed default `btn` class which usually provides structural CSS, and `submit-btn` which might be used by JS handlers.",
        "true_review_comment": "You removed the base `.btn` class and the specialized `.submit-btn` class. Usually, UI frameworks rely on `.btn` for padding and borders, and our JavaScript might be using `.submit-btn` for event binding. Are you sure removing these won't break the layout or forms?",
    },
    {
        "pr_id": "PR-506",
        "title": "Fix typo in onboarding email",
        "description": "Typo fix.",
        "author": "dev_alice",
        "diff": "@@ -5,3 +5,3 @@ def send_welcome():\n     body = \"\"\"\n-    Welcome to our plateform!\n+    Welcome to our platform!\n     \"\"\"",
        "true_pr_type": PRType.BUG_FIX,
        "true_bug_description": "None",
        "true_review_comment": "LGTM. Thanks for catching that!",
    },
    {
        "pr_id": "PR-507",
        "title": "Update React Hook usage",
        "description": "Fixing a state update loop.",
        "author": "ui_diana",
        "diff": "@@ -22,3 +22,3 @@ function Counter() {\n-    useEffect(() => {\n-        setCount(count + 1);\n-    });\n+    useEffect(() => {\n+        setCount(c => c + 1);\n+    }, []);",
        "true_pr_type": PRType.BUG_FIX,
        "true_bug_description": "None. Adding dependency array `[]` correctly stops the infinite re-render loop.",
        "true_review_comment": "Great fix! Changing to the functional state updater and adding the empty dependency array perfectly resolves the infinite re-render loop.",
    },
    {
        "pr_id": "PR-508",
        "title": "Fix NoneType error on user deletion",
        "description": "Sometimes user.metadata is None, crashing the script.",
        "author": "dev_dave",
        "diff": "@@ -40,1 +40,1 @@ def delete_user(user):\n-    log(f\"Deleting {user.metadata['name']}\")\n+    log(f\"Deleting {user.metadata.get('name', 'Unknown')}\")\n     db.delete(user)",
        "true_pr_type": PRType.BUG_FIX,
        "true_bug_description": "If `user.metadata` is `None`, calling `.get()` on it will throw an AttributeError.",
        "true_review_comment": "This doesn't fully fix the bug. If `user.metadata` is entirely `None`, then `None.get()` will raise an `AttributeError`. You should do: `user.metadata.get('name') if user.metadata else 'Unknown'`.",
    },
    {
        "pr_id": "PR-509",
        "title": "Expose internal metric endpoint",
        "description": "Adding a route for Datadog to ping our health and queue sizes.",
        "author": "junior_bob",
        "diff": "@@ -100,0 +100,3 @@ @app.route('/_internal/metrics')\n+def metrics():\n+    return jsonify({'queue_size': redis.llen('queue'), 'revenue': db.fetch_revenue()})",
        "true_pr_type": PRType.FEATURE,
        "true_bug_description": "Exposing highly sensitive financial metrics (`revenue`) on an unauthenticated endpoint.",
        "true_review_comment": "We absolutely cannot expose the daily `revenue` on an unauthenticated endpoint, even if the path starts with `_internal`. It's accessible to the internet. Please wrap this endpoint in an authentication decorator or IP whitelist, and remove the revenue metric—Datadog only needs the queue sizes.",
    },
    {
        "pr_id": "PR-510",
        "title": "Hash passwords on user creation",
        "description": "Ensure passwords aren't stored in plaintext.",
        "author": "dev_alice",
        "diff": "@@ -15,2 +15,3 @@ def create_user(username, password):\n-    hashed = md5(password.encode()).hexdigest()\n+    import bcrypt\n+    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())\n     db.insert('users', username=username, password=hashed)",
        "true_pr_type": PRType.SECURITY,
        "true_bug_description": "None. Upgrading from MD5 to bcrypt is exactly the right security patch.",
        "true_review_comment": "Excellent change! Moving from MD5 to a strong adaptive hash like bcrypt is crucial. Approved.",
    }
]

try:
    from datasets import load_dataset
    import random
    _PR_DS = list(load_dataset("github-code-clean", split="train", streaming=True).take(50))
    for row in _PR_DS:
        PULL_REQUESTS.append({
            "pr_id": f"PR-HF-{random.randint(1000,9999)}",
            "title": "Update logic",
            "description": "Extracted from HuggingFace github-code-clean dataset.",
            "diff": row.get("text", "No diff available")[:1000],
            "author": "hf_user",
            "true_pr_type": PRType.REFACTOR,
            "true_bug_description": "None",
            "true_review_comment": "LGTM."
        })
except Exception:
    pass
