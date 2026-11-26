"""発走前のライブデータ

- JRA公式の出馬表（当日の最新：取消・乗り替わり・馬体重）… www.jra.go.jp（トークンをたどって取得）
- 単勝・複勝オッズ（発走前の最新）と人気 … netkeiba の JSON（取れなければJRA出馬表のオッズ）
- 当日の馬場状態 … netkeiba の出馬表ページ
"""
import json
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import JO_NAME, USER_AGENT
from .entries import decode, parse_entry

JST = timezone(timedelta(hours=9))
JRA_URL = "https://www.jra.go.jp/JRADB/accessD.html"
NK_ODDS = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=%s&type=%d&action=update"
NK_SHUTUBA = "https://race.netkeiba.com/race/shutuba.html?race_id=%s"

TOK = re.compile(r"pw01d[a-z]{2}[0-9A-Za-z]*/[0-9A-F]{2}")
DDE = re.compile(r"pw01dde\d{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})/[0-9A-F]{2}")


def now_jst():
    return datetime.now(JST)


class Http:
    def __init__(self, interval=1.0):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"})
        self.interval = interval
        self.last = 0.0

    def _wait(self):
        w = self.interval - (time.time() - self.last)
        if w > 0:
            time.sleep(w)
        self.last = time.time()

    def post(self, url, data):
        for i in range(3):
            self._wait()
            try:
                r = self.s.post(url, data=data, timeout=60)
                if r.status_code == 200:
                    return r.content
            except Exception:
                pass
            time.sleep(3 * (i + 1))
        return None

    def get(self, url):
        for i in range(3):
            self._wait()
            try:
                r = self.s.get(url, timeout=30)
                if r.status_code == 200:
                    return r.content
            except Exception:
                pass
            time.sleep(3 * (i + 1))
        return None


# ───────────────────────── JRA 出馬表 ─────────────────────────
def jra_race_tokens(http, ymd, max_pages=60, log=print):
    """JRAの出馬表トップから、指定日の全レースの cname を集める → {(jo, rno): {"cname", "kai", "nichi"}}"""
    top = http.post(JRA_URL, {"cname": "pw01dli00/F3"})
    if top is None:
        raise RuntimeError("JRAの出馬表トップを取得できませんでした")
    queue = sorted(set(TOK.findall(decode(top))))
    seen, fetched, races = set(queue), 0, {}
    while queue and fetched < max_pages:
        t = queue.pop(0)
        m = DDE.match(t)
        if m:
            jo, _, kai, nichi, r, d = m.groups()
            if d == ymd:
                races[(jo, int(r))] = {"cname": t, "kai": int(kai), "nichi": int(nichi)}
            continue
        dates_in = re.findall(r"(20\d{6})", t)
        if dates_in and ymd not in dates_in:
            continue
        b = http.post(JRA_URL, {"cname": t})
        fetched += 1
        if b is None:
            continue
        for n in TOK.findall(decode(b)):
            if n not in seen:
                seen.add(n)
                queue.append(n)
    log("JRA：%s のレース %d 件（一覧ページ %d 件）" % (ymd, len(races), fetched))
    return races


def jra_race_card(http, cname, ymd, jo, rno):
    b = http.post(JRA_URL, {"cname": cname})
    return parse_entry(b, ymd, jo, rno) if b else None


# ───────────────────────── netkeiba オッズ・馬場 ─────────────────────────
def nk_race_id(ymd, jo, kai, nichi, rno):
    return "%s%s%02d%02d%02d" % (ymd[:4], jo, int(kai), int(nichi), int(rno))


def nk_odds(http, race_id):
    """→ ({馬番: {"win", "pop", "place_lo", "place_hi"}}, 時刻)。取れなければ ({}, None)"""
    out, ts = {}, None
    b = http.get(NK_ODDS % (race_id, 1))
    if not b:
        return out, ts
    try:
        j = json.loads(b.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return out, ts
    data = (j.get("data") or {})
    ts = data.get("official_datetime")
    odds = data.get("odds") or {}
    for num, v in (odds.get("1") or {}).items():
        try:
            out.setdefault(int(num), {}).update(win=float(v[0]), pop=int(v[2]))
        except (ValueError, IndexError, TypeError):
            pass
    for num, v in (odds.get("2") or {}).items():
        try:
            out.setdefault(int(num), {}).update(place_lo=float(v[0]), place_hi=float(v[1]))
        except (ValueError, IndexError, TypeError):
            pass
    # 発売前・取消は 0.0 や "---" になる
    return {k: v for k, v in out.items() if v.get("win", 0) > 0}, ts


def nk_going(http, race_id):
    """→ ("芝"/"ダ", "良"/"稍重"/"重"/"不良") 発表前なら cond は ""。"""
    b = http.get(NK_SHUTUBA % race_id)
    if not b:
        return "", ""
    s = b.decode("utf-8", "replace") if b[:3] != b"\x1b" else decode(b)
    m = re.search(r'<div class="RaceData01">(.*?)</div>', s, re.S)
    if not m:
        return "", ""
    t = re.sub(r"<[^>]+>|&nbsp;", " ", m.group(1))
    surf = "ダ" if "ダ" in t else ("芝" if "芝" in t else "")
    mc = re.search(r"馬場\s*[:：]\s*(良|稍|重|不)", t) or re.search(r"/\s*(良|稍重|稍|重|不良|不)\s*/", t)
    cond = {"稍": "稍重", "不": "不良"}.get(mc.group(1), mc.group(1)) if mc else ""
    return surf, cond


# ───────────────────────── 出馬表 → 成績データと同じ形の行 ─────────────────────────
def card_to_runs(card, cond="", odds=None):
    """出馬表（parse_entry の結果）→ 成績データ（runs）と同じ列の DataFrame（着順は空）"""
    rows = []
    odds = odds or {}
    for h in card["horses"]:
        if h["num"] is None:
            continue
        o = odds.get(h["num"], {})
        rows.append({
            "ymd": card["ymd"], "jo": card["jo"], "kai": card["kai"], "nichi": card["nichi"], "rno": card["rno"],
            "horse": h["horse"], "place": float("nan"), "status": h["status"] if h["status"] in ("取消", "除外") else "",
            "waku": h["waku"], "num": h["num"], "sex": h["sex"], "age": h["age"], "wt": h["wt"],
            "jockey": h["jockey"], "hw": h["hw"], "hwd": h["hwd"], "time": None, "tsec": float("nan"),
            "odds": o.get("win", h["odds"]), "pop": o.get("pop", h["pop"]),
            "c1": float("nan"), "c2": float("nan"), "c3": float("nan"), "c4": float("nan"), "corners": "",
            "trainer": h["trainer"], "blinker": h["blinker"], "owner": h["owner"], "breeder": h["breeder"],
            "surf": card["surf"], "dist": card["dist"], "inout": card["inout"], "turn": card["turn"],
            "weather": "", "cond": cond or "良", "heads": float("nan"), "name": card["name"], "grade": card["grade"],
            "cls_txt": card["cls_txt"], "last3": None, "first3": None, "cls": card["cls"],
            "place_lo": o.get("place_lo"), "place_hi": o.get("place_hi"),
            "horse_id": h["horse_id"], "sire": h["sire"], "dam": h["dam"], "bms": h["bms"],
        })
    df = pd.DataFrame(rows)
    for c in ("waku", "num", "age", "wt", "hw", "hwd", "odds", "pop", "dist", "place_lo", "place_hi", "kai", "nichi", "rno"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("kai", "nichi", "rno"):
        if c in df.columns:
            df[c] = df[c].astype(int)
    # 発売前で人気が無いときは単勝オッズの順で付ける
    if len(df) and df["pop"].isna().any() and df["odds"].notna().any():
        df["pop"] = df["odds"].rank(method="first")
    return df


def post_datetime(ymd, post):
    if not post:
        return None
    h, m = post.split(":")
    return datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]), int(h), int(m), tzinfo=JST)


def race_label(card):
    return "%s%dR %s" % (JO_NAME.get(card["jo"], card["jo"]), card["rno"], card["name"])
