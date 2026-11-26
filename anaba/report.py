"""発走30分前レポート（Markdown）

区分（config.tier_of）：6番人気以下のうち 単勝10〜30倍＝中穴、30倍以上＝大穴。
印と推奨券種は区分ごと（config.MARK_RULES）：
  中穴 ★＝単勝期待値1.1以上（単勝向き） ☆＝複勝期待値0.9以上
  大穴 ★＝複勝期待値0.9以上（複勝向き） ☆＝複勝期待値0.8以上
  血  ＝印は無いが奇跡の血量（3×4／4×3）を持つ
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .config import JO_NAME, MARK_RULES, REPORT_DIR, TIERS, tier_of
from .pedigree import PedStore
from .scoring import load_json, matched_factors

# 好走歴のある「条件」の名前（コース適性の欄に出す）
COURSE_FACTORS = ("同コース", "同競馬場", "同場・同芝ダ", "同距離帯", "道悪で", "同じ回り", "激走歴")
ROTATION_FACTORS = ("休み明け", "叩き", "中1週", "距離短縮", "距離延長", "替わり", "降級", "昇級", "乗り替わり",
                    "ブリンカー", "斤量", "馬体重")


def tier_row(r):
    return tier_of(r["odds"], r["pop"])


def mark_of(r):
    """→ (印, 推奨券種)。印が無ければ ("", "")"""
    t = tier_row(r)
    for mk, col, thr, bet in MARK_RULES.get(t, []):
        if r[col] >= thr:
            return mk, bet
    return "", ""


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else "%.1f%%" % (100 * x)


def _f(x, fmt="%.2f"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt % x


def _factor_note(names, fstats):
    """条件ごとに、同じ区分の過去の馬での 実績/期待（A/E）を見て、見落とされがちか・織り込み済みかを分ける"""
    plus, priced = [], []
    for n in names:
        st = fstats.get(n)
        if not st:
            continue
        if st["ae"] >= 1.02 and st.get("ae_a", 0) >= 1.0 and st.get("ae_b", 0) >= 1.0:
            plus.append("%s（%.2f）" % (n, st["ae"]))
        elif st["ae"] < 0.95:
            priced.append(n)
    return plus, priced


def _ped_line(store, hid, sire, bms, ped_effect):
    s = store.summary(hid) if isinstance(hid, str) and hid else None
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
            mir += "（過去の10番人気以下での検証：オッズ比 %.2f、95%%区間 %.2f〜%.2f＝%s）" % (e["or"], lo_, hi_, sig)
    elif s["miracle_eq"]:
        mir = "｜18.75%相当のクロスあり"
    return head + "｜" + cx + mir, s["miracle"]


def _rules_text(tier):
    return "、".join("%s＝%s期待値%.1f以上" % (mk, "単勝" if col == "ev_win" else "複勝", thr)
                    for mk, col, thr, _ in MARK_RULES[tier])


def race_markdown(df, card, odds_ts=None, minutes_to_post=None):
    fstats_all = load_json("factors.json", {}) or {}
    val = load_json("validation.json", {}) or {}
    ped_effect = load_json("pedigree_effect.json", {}) or {}
    store = PedStore()
    jo = JO_NAME.get(card["jo"], card["jo"])
    cond = df["cond"].iloc[0] if len(df) else ""
    title = "%s%dR %s（%s%dm・%s）発走 %s" % (jo, card["rno"], card["name"], card["surf"], card["dist"],
                                         cond or "馬場未発表", card["post"])
    lines = ["## " + title, ""]
    meta = []
    if odds_ts:
        meta.append("オッズ %s 時点" % odds_ts)
    if minutes_to_post is not None:
        meta.append("発走まで約%d分" % minutes_to_post)
    meta.append("%d頭" % len(df))
    lines += ["<sub>" + "／".join(meta) + "</sub>", ""]

    d = df.copy()
    d["tier"] = d.apply(tier_row, axis=1)
    mb = d.apply(mark_of, axis=1)
    d["mark"] = [m for m, _ in mb]
    d["bet"] = [b for _, b in mb]
    picks = []
    for tier in TIERS:
        cand = d[d["tier"] == tier]
        rng = "単勝10〜30倍" if tier == "中穴" else "単勝30倍以上"
        lines.append("### %s（%s）" % (tier, rng))
        if cand.empty:
            lines += ["該当馬なし", ""]
            continue
        key = "ev_win" if MARK_RULES[tier][0][1] == "ev_win" else "ev_place"
        cand = cand.sort_values(key, ascending=False)
        lines += ["| 印 | 馬番 | 馬名 | 単勝 | 人気 | 複勝 | 3着内率 モデル／市場 | 複勝期待値 | 単勝期待値 |",
                  "|:-:|--:|---|--:|--:|--:|--:|--:|--:|"]
        for _, r in cand.head(8).iterrows():
            pl = "%.1f-%.1f" % (r["place_lo"], r["place_hi"]) if pd.notna(r.get("place_lo")) else "—"
            lines.append("| %s | %d | %s | %.1f | %d | %s | %s／%s | %s | %s |" % (
                r["mark"], r["num"], r["horse"], r["odds"], r["pop"], pl, _pct(r["p_top3"]), _pct(r["p_mkt"]),
                _f(r["ev_place"]), _f(r["ev_win"])))
        lines.append("")
        fstats = fstats_all.get(tier, {})
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
            bet = "（%s向き）" % r["bet"] if r["bet"] else ""
            head = "#### %s %d %s ― %s・%d番人気・単勝%.1f倍%s" % (r["mark"] or "血", r["num"], r["horse"], tier,
                                                           r["pop"], r["odds"], bet)
            body = ["- 推定3着内率 %s（オッズからの見込み %s の %.1f倍）／推定勝率 %s" % (
                _pct(r["p_top3"]), _pct(r["p_mkt"]), r["edge"], _pct(r["p_win"]))]
            if plus:
                body.append("- 市場が見落としがちな材料：" + "、".join(plus))
            if course:
                pr = [x for x in course if x in priced]
                body.append("- 好走歴のある条件：" + "、".join(course) +
                            ("（※このうち「%s」は人気に織り込まれやすい）" % "、".join(pr) if pr else ""))
            if rot:
                body.append("- 臨戦過程：" + "、".join(rot))
            body.append("- 血統：" + ped)
            lines += [head] + body + [""]
    if not picks:
        lines += ["印の付く中穴・大穴はありません。", ""]

    over = d[(d["pop"] <= 5) & (d["ev_place"] < 0.75)]
    if len(over):
        lines += ["割高（人気のわりに見込みが低い）：" + "、".join("%d %s（複勝期待値%.2f）" % (r["num"], r["horse"], r["ev_place"])
                                                   for _, r in over.iterrows()), ""]
    foot = []
    for tier in TIERS:
        v = (val.get("tiers") or {}).get(tier)
        if not v:
            continue
        ms = []
        for mk, r in v["marks"].items():
            ms.append("%s %d頭・%s回収率 %d%%（年ごと %d〜%d%%）" % (mk, r["n"], r["bet"], round(100 * r["roi"]),
                                                          round(100 * r["roi_min"]), round(100 * r["roi_max"])))
        base = v["全部"]
        foot.append("%s［%s］：%s。全部買うと 複勝%d%%・単勝%d%%" % (
            tier, _rules_text(tier), "／".join(ms), round(100 * base["place_roi"]), round(100 * base["win_roi"])))
    if foot:
        lines.append("<sub>過去の検証（%s、その年より前だけで学習・確定オッズ）　%s。期待値1.0＝回収率100%%の見込み。"
                     "期待値は推定で、100%%を超える保証はありません。</sub>" % (val.get("period", ""), "　".join(foot)))
    return "\n".join(lines), picks


PICK_COLS = ["ymd", "jo", "rno", "num", "horse", "tier", "odds", "pop", "place_lo", "place_hi", "p_top3", "p_mkt",
             "p_win", "ev_place", "ev_win", "mark", "bet", "reported_at"]


def write_picks(path, df, card, reported_at):
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(PICK_COLS)
        for _, r in df.iterrows():
            t = tier_row(r)
            if not t:
                continue
            mk, bet = mark_of(r)
            w.writerow([card["ymd"], card["jo"], card["rno"], r["num"], r["horse"], t, r["odds"], r["pop"],
                        r.get("place_lo"), r.get("place_hi"), round(r["p_top3"], 5), round(r["p_mkt"], 5),
                        round(r["p_win"], 5), round(r["ev_place"], 4), round(r["ev_win"], 4), mk, bet, reported_at])


def day_dir(ymd):
    return REPORT_DIR / ymd
