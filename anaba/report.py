"""発走30分前レポート（Markdown）"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANDIDATE_POP, JO_NAME, REPORT_DIR
from .pedigree import PedStore
from .scoring import load_json, matched_factors

# 好走歴のある「条件」の名前（コース適性の欄に出す）
COURSE_FACTORS = ("同コース", "同競馬場", "同場・同芝ダ", "同距離帯", "道悪で", "同じ回り", "激走歴")
ROTATION_FACTORS = ("休み明け", "叩き", "中1週", "距離短縮", "距離延長", "替わり", "降級", "昇級", "乗り替わり",
                    "ブリンカー", "斤量", "馬体重")


# 印の基準：ウォークフォワード検証で「毎年」安定していた層に合わせる（それより上の閾値は件数が少なく年ごとのブレが大きい）
MARKS = {"★": 0.9, "☆": 0.8}
WIN_MARK = 1.0


def mark_of(r):
    if r["ev_place"] >= MARKS["★"]:
        return "★"
    if r["ev_place"] >= MARKS["☆"]:
        return "☆"
    if r["ev_win"] >= WIN_MARK:
        return "単"
    return ""


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else "%.1f%%" % (100 * x)


def _f(x, fmt="%.2f"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt % x


def _factor_note(names, fstats):
    """条件ごとに、過去の大穴（10番人気以下）での 実績/期待 を添える"""
    plus, priced = [], []
    for n in names:
        st = fstats.get(n)
        if not st:
            continue
        if st["ae"] >= 1.0 and st.get("ae_a", 0) >= 0.97 and st.get("ae_b", 0) >= 0.97:
            plus.append("%s（%.2f）" % (n, st["ae"]))
        elif st["ae"] < 0.95:
            priced.append(n)
    return plus, priced


def _ped_line(store, hid, sire, bms, ped_effect):
    s = store.summary(hid) if hid else None
    head = "父%s／母父%s" % (sire or "?", bms or "?")
    if not s:
        return head + "｜5代血統：未取得", False
    cs = s["crosses"][:3]
    cx = "、".join("%s %s" % (c["name"], c["label"]) for c in cs) if cs else "クロスなし（5代内）"
    mir = ""
    if s["miracle"]:
        mir = "｜**奇跡の血量（18.75%%）：%s**" % "、".join(s["miracle_names"])
        e = (ped_effect or {}).get("miracle")
        if e:
            lo_, hi_ = e.get("ci", [float("nan"), float("nan")])
            sig = "有意差なし" if (lo_ <= 1.0 <= hi_) else ("好走しやすい" if lo_ > 1.0 else "好走しにくい")
            mir += "（過去の大穴での検証：オッズ比 %.2f、95%%区間 %.2f〜%.2f＝%s）" % (e["or"], lo_, hi_, sig)
    elif s["miracle_eq"]:
        mir = "｜18.75%相当のクロスあり"
    return head + "｜" + cx + mir, s["miracle"]


def race_markdown(df, card, odds_ts=None, now=None, minutes_to_post=None):
    fstats = load_json("factors.json", {}) or {}
    val = load_json("validation.json", {}) or {}
    ped_effect = load_json("pedigree_effect.json", {}) or {}
    store = PedStore()
    jo = JO_NAME.get(card["jo"], card["jo"])
    cond = df["cond"].iloc[0] if len(df) else ""
    title = "%s%dR %s（%s%dm・%s）発走 %s" % (jo, card["rno"], card["name"], "ダ" if card["surf"] == "ダ" else card["surf"],
                                         card["dist"], cond or "馬場未発表", card["post"])
    lines = ["## " + title, ""]
    meta = []
    if odds_ts:
        meta.append("オッズ %s 時点" % odds_ts)
    if minutes_to_post is not None:
        meta.append("発走まで約%d分" % minutes_to_post)
    meta.append("%d頭" % len(df))
    lines += ["<sub>" + "／".join(meta) + "</sub>", ""]

    d = df.copy()
    d["mark"] = d.apply(mark_of, axis=1)
    cand = d[d["pop"] >= CANDIDATE_POP].sort_values("ev_place", ascending=False)
    lines += ["| 印 | 馬番 | 馬名 | 単勝 | 人気 | 複勝 | 3着内率 モデル／市場 | 複勝期待値 | 単勝期待値 |",
              "|:-:|--:|---|--:|--:|--:|--:|--:|--:|"]
    for _, r in cand.head(8).iterrows():
        pl = "%.1f-%.1f" % (r["place_lo"], r["place_hi"]) if pd.notna(r.get("place_lo")) else "—"
        lines.append("| %s | %d | %s | %.1f | %d | %s | %s／%s | %s | %s |" % (
            r["mark"] or "", r["num"], r["horse"], r["odds"], r["pop"], pl, _pct(r["p_top3"]), _pct(r["p_mkt"]),
            _f(r["ev_place"]), _f(r["ev_win"])))
    lines.append("")

    # 詳細：印のある馬 ＋ 奇跡の血量の馬
    picks = []
    for _, r in cand.iterrows():
        one = d[d["num"] == r["num"]]
        names = matched_factors(one)
        plus, priced = _factor_note(names, fstats)
        course = [n for n in names if n.startswith(COURSE_FACTORS) or "激走歴" in n]
        rot = [n for n in names if any(k in n for k in ROTATION_FACTORS)]
        ped, miracle = _ped_line(store, r.get("horse_id"), r.get("sire"), r.get("bms"), ped_effect)
        if not (r["mark"] or miracle):
            continue
        picks.append(r)
        head = "#### %s %d %s（%d番人気・単勝%.1f倍）" % (r["mark"] or "血", r["num"], r["horse"], r["pop"], r["odds"])
        body = ["- 推定3着内率 %s（オッズからの見込み %s の %.1f倍）" % (_pct(r["p_top3"]), _pct(r["p_mkt"]), r["edge"])]
        if plus:
            body.append("- 市場が見落としがちな材料：" + "、".join(plus))
        if course:
            body.append("- 好走歴のある条件：" + "、".join(course) +
                        ("（※このうち「%s」は人気に織り込まれやすい）" % "、".join(x for x in course if x in priced) if any(x in priced for x in course) else ""))
        if rot:
            body.append("- 臨戦過程：" + "、".join(rot))
        body.append("- 血統：" + ped)
        lines += [head] + body + [""]
    if not picks:
        lines += ["印の付く大穴候補はありません（複勝期待値0.8以上・単勝期待値1.0以上の8番人気以下がいない）。", ""]

    # 割高（人気だが期待値が低い）
    over = d[(d["pop"] <= 5) & (d["ev_place"] < 0.75)]
    if len(over):
        lines += ["割高（人気のわりに見込みが低い）：" + "、".join("%d %s（期待値%.2f）" % (r["num"], r["horse"], r["ev_place"])
                                                   for _, r in over.iterrows()), ""]
    v = val.get("ev_place>=0.9")
    yr = val.get("by_year_place_roi", {}).get("ev_place>=0.9", {})
    if v:
        rng = ("（年ごと %d〜%d%%）" % (round(100 * min(yr.values())), round(100 * max(yr.values())))) if yr else ""
        lines.append("<sub>★＝複勝期待値0.9以上、☆＝0.8以上、単＝単勝期待値1.0以上（期待値1.0＝回収率100%%の見込み）。"
                     "過去の検証（%s、確定オッズ・10番人気以下）：★相当 %d頭で3着内率 %.1f%%・複勝回収率 %d%%%s。"
                     "大穴を全部買うと %d%%。期待値は推定で、100%%を超える保証はありません。</sub>"
                     % (v["period"], v["n"], 100 * v["top3"], round(100 * v["place_roi"]), rng,
                        round(100 * val["all"]["place_roi"])))
    return "\n".join(lines), picks


def write_picks(path, df, card, reported_at):
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ymd", "jo", "rno", "num", "horse", "odds", "pop", "place_lo", "place_hi", "p_top3", "p_mkt", "p_win",
            "ev_place", "ev_win", "mark", "reported_at"]
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        for _, r in df[df["pop"] >= CANDIDATE_POP].iterrows():
            w.writerow([card["ymd"], card["jo"], card["rno"], r["num"], r["horse"], r["odds"], r["pop"],
                        r.get("place_lo"), r.get("place_hi"), round(r["p_top3"], 5), round(r["p_mkt"], 5),
                        round(r["p_win"], 5), round(r["ev_place"], 4), round(r["ev_win"], 4), mark_of(r), reported_at])


def day_dir(ymd):
    return REPORT_DIR / ymd
