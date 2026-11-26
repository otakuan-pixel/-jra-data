"""今、報告すべきレースがあるか（標準ライブラリだけで判定。Actions で重い準備の前に呼ぶ）

  python -m anaba.due [--date YYYYMMDD] [--lo 25] [--hi 35]
  → 標準出力に due=true / due=false（GITHUB_OUTPUT があればそこにも書く）

state.json（前夜の prefetch で作る）が無い日は true（初回の一覧作成が必要）。
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parent.parent


def is_due(ymd, now, lo=25, hi=35, root=ROOT):
    p = root / "reports" / ymd / "state.json"
    if not p.exists():
        return True, "state.json なし（初回）"
    st = json.loads(p.read_text(encoding="utf-8"))
    races = st.get("races") or {}
    if not races:
        # 前回「開催なし」だった：30分おきに確認し直す
        chk = st.get("checked")
        if chk and (now - datetime.fromisoformat(chk)).total_seconds() < 1800:
            return False, "開催なし（確認済み）"
        return True, "開催の再確認"
    for k, r in races.items():
        if r.get("reported") or not r.get("post") or r.get("surf") == "障":
            continue
        h, m = r["post"].split(":")
        post = datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]), int(h), int(m), tzinfo=JST)
        mins = (post - now).total_seconds() / 60
        if lo <= mins <= hi or 3 <= mins < lo:
            return True, "%s（発走まで%d分）" % (k, mins)
    return False, "対象なし"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--lo", type=int, default=25)
    ap.add_argument("--hi", type=int, default=35)
    a = ap.parse_args()
    now = datetime.now(JST)
    ymd = a.date or now.strftime("%Y%m%d")
    due, why = is_due(ymd, now, a.lo, a.hi)
    print("due=%s  # %s %s" % (str(due).lower(), ymd, why))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write("due=%s\n" % str(due).lower())


if __name__ == "__main__":
    main()
