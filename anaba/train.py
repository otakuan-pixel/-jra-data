"""学習一式：複勝払戻モデル → 特徴量 → 条件表（research）→ 検証（walk-forward）→ 本番モデル"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import horses, model as M, payout
from .config import MARK_RULES, MODEL_DIR, REPORT_DIR, TIERS, tier_series
from .data import KEY, load_results
from .features import prepare
from .research import course_table, expected_top3, factor_table, profile_table


def build_frame(log=print):
    races, runs = load_results()
    pm = payout.fit(races, runs, log=log)
    pp = payout.place_payouts(races, runs, pm)
    df = prepare(runs, horses.load())
    df = df.merge(pp, on=KEY + ["num"], how="left")
    df["place_pay"] = df["place_pay"].fillna(0.0)
    df["exp_top3"] = expected_top3(df)
    log("特徴量：%d 行 × %d 列" % df.shape)
    return df


def _md_table(t, fmt=None):
    fmt = fmt or {}
    cols = list(t.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            if f and isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
                cells.append(f % v)
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


FT_FMT = {"該当率": "%.3f", "3着内率": "%.3f", "非該当の3着内率": "%.3f", "リフト": "%.2f", "A/E": "%.3f",
          "A/E前半": "%.3f", "A/E後半": "%.3f", "単勝回収": "%.2f", "複勝回収": "%.2f", "複勝前半": "%.2f", "複勝後半": "%.2f"}
CT_FMT = {"top3": "%.3f", "ae": "%.3f", "win_roi": "%.2f", "place_roi": "%.2f"}
SEL_FMT = {"3着内率": "%.3f", "複勝回収": "%.3f", "単勝回収": "%.3f", "平均オッズ": "%.1f"}


def _ints(t, cols):
    return t.assign(**{c: t[c].astype(int) for c in cols if c in t.columns})


def _year_md(v):
    ys = sorted(v["by_year"]["全部"]["n"])
    L = ["| 選び方 | " + " | ".join(ys) + " |", "|---|" + "|".join("--:" for _ in ys) + "|"]
    for k, r in v["by_year"].items():
        L.append("| %s | " % k + " | ".join(
            "%d%%／%d%%（%d頭）" % (round(100 * r["place_roi"][y]), round(100 * r["win_roi"][y]), r["n"][y])
            if y in r["n"] else "—" for y in ys) + " |")
    return "\n".join(L)


def research(df, log=print):
    out, fj = {}, {}
    for tier in TIERS:
        base, ft = factor_table(df, tier)
        out[tier] = (base, ft, profile_table(df, tier), course_table(df, tier))
        fj[tier] = {r["条件"]: {"n": int(r["該当数"]), "top3": r["3着内率"], "lift": r["リフト"], "ae": r["A/E"],
                               "ae_a": r["A/E前半"], "ae_b": r["A/E後半"], "place_roi": r["複勝回収"], "win_roi": r["単勝回収"]}
                    for _, r in ft.iterrows()}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "factors.json").write_text(json.dumps(fj, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def validate(df, log=print):
    wf = M.walk_forward(df, log=log)
    period = "%s〜%s" % (wf["ymd"].min()[:6], wf["ymd"].max()[:6])

    def js(x):
        return {"n": int(len(x)), "top3": float(x["top3"].mean()), "place_roi": float(x["place_pay"].mean() / 100),
                "win_roi": float((x["win"] * x["odds"]).mean())}
    val = {"period": period, "tiers": {}}
    sels = {}
    for tier in TIERS:
        d = wf[tier_series(wf) == tier]
        ss = M.selections(d, tier)
        v = {k: js(d[m]) for k, m in ss.items()}
        v["by_year"] = {k: {"n": {str(y): int(len(g)) for y, g in d[m].groupby("year")},
                            "place_roi": {str(y): float(g["place_pay"].mean() / 100) for y, g in d[m].groupby("year")},
                            "win_roi": {str(y): float((g["win"] * g["odds"]).mean()) for y, g in d[m].groupby("year")}}
                        for k, m in ss.items()}
        v["marks"] = {}
        for mk, col, thr, bet in MARK_RULES[tier]:
            m = d[col] >= thr
            r = js(d[m])
            r["bet"] = bet
            r["roi"] = r["win_roi"] if bet == "単勝" else r["place_roi"]
            yr = [float((g["win"] * g["odds"]).mean()) if bet == "単勝" else float(g["place_pay"].mean() / 100)
                  for _, g in d[m].groupby("year")]
            r["roi_min"], r["roi_max"] = min(yr), max(yr)
            r["base_roi"] = v["全部"]["win_roi"] if bet == "単勝" else v["全部"]["place_roi"]
            v["marks"][mk] = r
        v["logloss"] = {"market": float(M.logloss(d["top3"], d["p_mkt"])), "model": float(M.logloss(d["top3"], d["p_top3"]))}
        val["tiers"][tier] = v
        sels[tier] = M.selection_table(wf, tier)
    (MODEL_DIR / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1), encoding="utf-8")
    return wf, sels, val


def write_research_md(res, sels=None, val=None, path=None):
    path = path or REPORT_DIR / "research" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["# 中穴・大穴の好走条件の検証", "",
         "更新：%s／対象：JRA平地 2021年1月〜（全10場）" % datetime.now().strftime("%Y-%m-%d"), "",
         "## 区分", "",
         "- **中穴**＝6〜10番人気・単勝10〜30倍、**大穴**＝10番人気以下・単勝30倍以上",
         "- 10番人気は両方に入り、人気とオッズがずれる馬もいるので、6番人気以下をオッズで分ける（払戻はオッズで決まるため）", "",
         "## 前提", ""]
    for tier in TIERS:
        b = res[tier][0]
        L.append("- %s：%d頭、3着内率 %.1f%%、単勝回収率 %d%%、複勝回収率 %d%%（複勝は実測＋推定払戻）"
                 % (tier, b["n"], 100 * b["top3"], round(100 * b["win_roi"]), round(100 * b["place_roi"])))
    L += ["- 大穴ほど「買われすぎ」が大きい（控除率20%を引いた平均80%より、さらに低い）", "",
          "## 読み方", "",
          "- **リフト**＝当てはまる馬の3着内率 ÷ 当てはまらない馬の3着内率（好走馬の「背景」としてよく出てくるか）",
          "- **A/E**＝実際の3着内数 ÷ オッズから期待される3着内数。**1.0を超えると市場が過小評価**。リフトが高くてもA/Eが1未満なら、すでに人気に織り込まれている",
          "- 前半＝2021〜23年、後半＝2024〜26年。両方で同じ向きなら「安定」", ""]
    if sels is not None:
        L += ["## モデルの検証（ウォークフォワード：その年より前だけで学習 → その年を予想）", "",
              "確定オッズ。期間 %s。回収率は「複勝／単勝」。" % val["period"], ""]
        for tier in TIERS:
            v = val["tiers"][tier]
            L += ["### %s（対数損失：市場 %.5f → モデル %.5f）" % (tier, v["logloss"]["market"], v["logloss"]["model"]), "",
                  _md_table(_ints(sels[tier].reset_index().rename(columns={"index": "選び方"}), ["頭数", "3着内"]), SEL_FMT), "",
                  "年ごと（複勝／単勝・頭数）", "", _year_md(v), ""]
        L += ["閾値を上げるほど良くなるわけではない（件数が減り、推定の誤差も大きくなる）。"
              "レポートの印は、区分ごとに毎年安定していた層に合わせている：" +
              "、".join("%s%s＝%s期待値%.1f以上" % (t, mk, "単勝" if col == "ev_win" else "複勝", thr)
                       for t in TIERS for mk, col, thr, _ in MARK_RULES[t]), ""]
    for tier in TIERS:
        base, ft, prof, ct = res[tier]
        L += ["## %s：条件ごとの成績（A/E順）" % tier, "", _md_table(ft, FT_FMT), "",
              "### %s：好走馬と凡走馬の比較" % tier, "",
              _md_table(prof, {"好走馬の平均": "%.3f", "凡走馬の平均": "%.3f", "差（標準偏差単位）": "%.3f"}), "",
              "### %s：コース別の出やすさ（300頭以上のコース、上位と下位）" % tier, "",
              _md_table(_ints(ct.head(12), ["n"]), CT_FMT), "", "…", "", _md_table(_ints(ct.tail(6), ["n"]), CT_FMT), ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def train(do_validate=True, log=print):
    df = build_frame(log)
    res = research(df, log)
    sels = val = None
    if do_validate:
        _, sels, val = validate(df, log)
    b = M.fit_all(df, log=log)
    b.save()
    p = write_research_md(res, sels, val)
    log("保存：%s / %s" % (MODEL_DIR / "longshot.pkl", p))
    return df, b
