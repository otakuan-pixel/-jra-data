"""レポートで挙げた馬の答え合わせ（実際の着順と払戻）→ reports/ledger.csv と reports/ledger.md

発走前のオッズで判定した「本番の成績」なので、確定オッズでの過去検証より厳しい（オッズは締切までに動く）。
"""
import json

import pandas as pd

from .backfill import fetch_race, load_nk_races
from .config import REPORT_DIR
from .live import nk_race_id
from .pedigree import Fetcher


def _picks():
    fs = sorted(REPORT_DIR.glob("2*/picks.csv"))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f, dtype={"ymd": str, "jo": str}) for f in fs], ignore_index=True)
    d["jo"] = d["jo"].str.zfill(2)
    return d.drop_duplicates(["ymd", "jo", "rno", "num"], keep="last")


def settle(log=print):
    p = _picks()
    if p.empty:
        log("まだレポートがありません")
        return None
    cache = load_nk_races()
    fetcher = Fetcher(interval=1.1)
    rows = []
    for (ymd, jo, rno), g in p.groupby(["ymd", "jo", "rno"]):
        st = REPORT_DIR / ymd / "state.json"
        if not st.exists():
            continue
        info = json.loads(st.read_text(encoding="utf-8"))["races"].get("%s_%02d" % (jo, int(rno)))
        if not info:
            continue
        rid = nk_race_id(ymd, jo, info["kai"], info["nichi"], rno)
        r = cache.get(rid) or fetch_race(fetcher, cache, rid)
        if not r or not any(h["place"] for h in r["horses"]):
            continue
        place = {h["num"]: h["place"] for h in r["horses"]}
        fuku = {int(c): a for c, a in r["pays"].get("複勝", []) if c.isdigit()}
        tan = {int(c): a for c, a in r["pays"].get("単勝", []) if c.isdigit()}
        for _, x in g.iterrows():
            n = int(x["num"])
            rows.append({**x.to_dict(), "place": place.get(n), "place_pay": fuku.get(n, 0), "win_pay": tan.get(n, 0)})
    if not rows:
        log("結果の出ているレースがありません")
        return None
    led = pd.DataFrame(rows)
    led.to_csv(REPORT_DIR / "ledger.csv", index=False)
    write_md(led)
    log("答え合わせ：%d 頭（%d レース）" % (len(led), led.groupby(["ymd", "jo", "rno"]).ngroups))
    return led


def _sum(d):
    return {"頭数": len(d), "3着内": int((d["place"] <= 3).sum()), "3着内率": (d["place"] <= 3).mean(),
            "複勝回収率": d["place_pay"].sum() / (100 * len(d)) if len(d) else float("nan"),
            "単勝回収率": d["win_pay"].sum() / (100 * len(d)) if len(d) else float("nan")}


def write_md(led):
    from .config import MARK_RULES, TIERS
    led = led.copy()
    led["mark"] = led["mark"].fillna("")
    if "tier" not in led.columns:
        led["tier"] = ""
    L = ["# 本番の成績（発走30分前に判定した馬）", "",
         "期間 %s〜%s。1頭100円ずつ買ったときの回収率。件数が少ないうちは偶然の振れが非常に大きい。" % (led["ymd"].min(), led["ymd"].max()), "",
         "| 区分 | 選び方 | 頭数 | 3着内 | 3着内率 | 複勝回収率 | 単勝回収率 |", "|---|---|--:|--:|--:|--:|--:|"]
    for tier in TIERS:
        c = led[led["tier"] == tier]
        groups = [("印%s（%s向き）" % (mk, bet), c[c["mark"] == mk]) for mk, _, _, bet in MARK_RULES[tier]]
        groups.append(("全部（比較用）", c))
        for name, d in groups:
            s = _sum(d)
            if s["頭数"] == 0:
                continue
            L.append("| %s | %s | %d | %d | %.1f%% | %d%% | %d%% |" % (tier, name, s["頭数"], s["3着内"], 100 * s["3着内率"],
                                                                 round(100 * s["複勝回収率"]), round(100 * s["単勝回収率"])))
    hits = led[(led["mark"] != "") & (led["place"] <= 3)]
    if len(hits):
        L += ["", "## 印の馬で3着内に来た馬", "", "| 日付 | レース | 区分 | 馬 | 人気 | 着順 | 複勝 | 単勝 |",
              "|---|---|---|---|--:|--:|--:|--:|"]
        for _, r in hits.sort_values("ymd").iterrows():
            L.append("| %s | %s%dR | %s | %s %s | %d | %d | %d円 | %d円 |" % (
                r["ymd"], r["jo"], r["rno"], r["tier"], r["mark"], r["horse"], r["pop"], r["place"], r["place_pay"], r["win_pay"]))
    (REPORT_DIR / "ledger.md").write_text("\n".join(L) + "\n", encoding="utf-8")
