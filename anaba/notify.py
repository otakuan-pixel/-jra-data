"""通知：GitHub Issue（1日1件・レースごとにコメント）と、任意で Discord / Slack の Webhook

- GitHub Actions 上では GITHUB_TOKEN と gh コマンドで Issue に投稿する（スマホの GitHub アプリに通知が届く）
- 環境変数 ANABA_WEBHOOK_URL があれば、そこにも送る（discord.com なら Discord、hooks.slack.com なら Slack 形式）
"""
import json
import os
import shutil
import subprocess

from .live import now_jst


def _gh(*args, input_text=None):
    return subprocess.run(["gh"] + list(args), input=input_text, capture_output=True, text=True, timeout=60)


def github_issue_comment(ymd, body, log=print):
    if not (shutil.which("gh") and (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))):
        return False
    title = "大穴レポート %s-%s-%s" % (ymd[:4], ymd[4:6], ymd[6:])
    r = _gh("issue", "list", "--state", "open", "--search", "in:title \"%s\"" % title, "--json", "number,title")
    num = None
    if r.returncode == 0:
        for it in json.loads(r.stdout or "[]"):
            if it["title"] == title:
                num = it["number"]
    if num is None:
        r = _gh("issue", "create", "--title", title, "--body",
                "発走30分前の大穴分析をレースごとにコメントします。\n\n※期待値は推定です。的中や利益を保証するものではありません。")
        if r.returncode != 0:
            log("Issue作成に失敗：" + r.stderr.strip()[:200])
            return False
        num = r.stdout.strip().rsplit("/", 1)[-1]
    r = _gh("issue", "comment", str(num), "--body-file", "-", input_text=body)
    if r.returncode != 0:
        log("Issueコメントに失敗：" + r.stderr.strip()[:200])
        return False
    return True


def webhook(body, log=print):
    url = os.environ.get("ANABA_WEBHOOK_URL")
    if not url:
        return False
    import requests
    text = body if len(body) < 1900 else body[:1900] + "\n…（続きはGitHubのレポート）"
    payload = {"content": text} if "discord" in url else {"text": text}
    try:
        r = requests.post(url, json=payload, timeout=20)
        return r.status_code < 300
    except Exception as e:
        log("Webhook失敗：%r" % e)
        return False


def send(ymd, body, log=print):
    ok1 = github_issue_comment(ymd, body, log)
    ok2 = webhook(body, log)
    log("通知：GitHub Issue=%s Webhook=%s（%s）" % (ok1, ok2, now_jst().strftime("%H:%M")))
