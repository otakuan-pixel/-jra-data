"""学習一式：複勝払戻モデル → 特徴量 → 条件表（research）→ 検証（walk-forward）→ 本番モデル"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import horses, model as M, payout
from .config import LONGSHOT_POP, MODEL_DIR, REPORT_DIR
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


def _year_md(val):
    ys = sorted(val["by_year_all"])
    L = ["| 選び方 | " + " | ".join(ys) + " |", "|---|" + "|".join("--:" for _ in ys) + "|",
         "| 10番人気以下 全部（複勝） | " + " | ".join("%d%%" % round(100 * val["by_year_all"][y]) for y in ys) + " |"]
    for k in val["by_year_place_roi"]:
        pr, wr, n = val["by_year_place_roi"][k], val["by_year_win_roi"][k], val["by_year_n"][k]
        L.append("| %s | " % k.replace("ev_place", "複勝期待値").replace("ev_win", "単勝期待値") + " | ".join(
            "%d%%／%d%%（%d頭）" % (round(100 * pr.get(y, float("nan"))), round(100 * wr.get(y, float("nan"))), n.get(y, 0))
            if y in pr else "—" for y in ys) + " |")
    return "\n".join(L)


def research(df, log=print):
    base, ft = factor_table(df)
    prof = profile_table(df)
    ct = course_table(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fj = {r["条件"]: {"n": int(r["該当数"]), "top3": r["3着内率"], "lift": r["リフト"], "ae": r["A/E"],
                     "ae_a": r["A/E前半"], "ae_b": r["A/E後半"], "place_roi": r["複勝回収"], "win_roi": r["単勝回収"]}
          for _, r in ft.iterrows()}
    (MODEL_DIR / "factors.json").write_text(json.dumps(fj, ensure_ascii=False, indent=1), encoding="utf-8")
    return base, ft, prof, ct


def validate(df, log=print):
    wf = M.walk_forward(df, log=log)
    sel = M.selection_table(wf)
    d = wf[wf["pop"] >= LONGSHOT_POP]
    period = "%s〜%s" % (d["ymd"].min()[:6], d["ymd"].max()[:6])

    def js(x):
        return {"n": int(len(x)), "top3": float(x["top3"].mean()), "place_roi": float(x["place_pay"].mean() / 100),
                "win_roi": float((x["win"] * x["odds"]).mean()), "period": period}
    val = {"all": js(d), "period": period}
    sels = {"ev_place>=%.1f" % t: d["ev_place"] >= t for t in (0.8, 0.9, 1.0, 1.1)}
    sels.update({"ev_win>=%.1f" % t: d["ev_win"] >= t for t in (1.0, 1.2)})
    for k, m in sels.items():
        val[k] = js(d[m])
    val["by_year_place_roi"] = {k: {str(y): float(g["place_pay"].mean() / 100) for y, g in d[m].groupby("year")}
                                for k, m in sels.items()}
    val["by_year_win_roi"] = {k: {str(y): float((g["win"] * g["odds"]).mean()) for y, g in d[m].groupby("year")}
                              for k, m in sels.items()}
    val["by_year_n"] = {k: {str(y): int(len(g)) for y, g in d[m].groupby("year")} for k, m in sels.items()}
    val["by_year_all"] = {str(y): float(g["place_pay"].mean() / 100) for y, g in d.groupby("year")}
    val["logloss"] = {"market": float(M.logloss(d["top3"], d["p_mkt"])), "model": float(M.logloss(d["top3"], d["p_top3"]))}
    (MODEL_DIR / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1), encoding="utf-8")
    return wf, sel, val


def write_research_md(base, ft, prof, ct, sel=None, val=None, wf=None, path=None):
    path = path or REPORT_DIR / "research" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["# 大穴（10番人気以下）の好走条件の検証", "",
         "更新：%s／対象：JRA平地 2021年1月〜（全10場）" % datetime.now().strftime("%Y-%m-%d"), "",
         "## 前提", "",
         "- 10番人気以下：%d頭、3着内率 %.1f%%、単勝回収率 %d%%、複勝回収率 %d%%（複勝は実測＋推定払戻）"
         % (base["n"], 100 * base["top3"], round(100 * base["win_roi"]), round(100 * base["place_roi"])),
         "- 大穴は構造的に「買われすぎ」。回収率100%を超えるには、3着内率を市場の想定の約1.6倍にする必要がある", "",
         "## 読み方", "",
         "- **リフト**＝当てはまる馬の3着内率 ÷ 当てはまらない馬の3着内率（好走馬の「背景」としてよく出てくるか）",
         "- **A/E**＝実際の3着内数 ÷ オッズから期待される3着内数。**1.0を超えると市場が過小評価**。リフトが高くてもA/Eが1未満なら、すでに人気に織り込まれている",
         "- 前半＝2021〜23年、後半＝2024〜26年。両方で同じ向きなら「安定」", "",
         "## 条件ごとの成績（A/E順）", "",
         _md_table(ft, {"該当率": "%.3f", "3着内率": "%.3f", "非該当の3着内率": "%.3f", "リフト": "%.2f", "A/E": "%.3f",
                        "A/E前半": "%.3f", "A/E後半": "%.3f", "単勝回収": "%.2f", "複勝回収": "%.2f",
                        "複勝前半": "%.2f", "複勝後半": "%.2f"}), "",
         "## 好走馬と凡走馬の比較（10番人気以下）", "",
         _md_table(prof, {"好走馬の平均": "%.3f", "凡走馬の平均": "%.3f", "差（標準偏差単位）": "%.3f"}), "",
         "## コース別：大穴の出やすさ（300頭以上のコース）", "",
         _md_table(ct.head(15).assign(n=lambda x: x["n"].astype(int)),
                   {"top3": "%.3f", "ae": "%.3f", "win_roi": "%.2f", "place_roi": "%.2f"}), "",
         "…（下位）", "",
         _md_table(ct.tail(8).assign(n=lambda x: x["n"].astype(int)),
                   {"top3": "%.3f", "ae": "%.3f", "win_roi": "%.2f", "place_roi": "%.2f"}), ""]
    if sel is not None:
        L += ["## モデルの検証（ウォークフォワード：その年より前だけで学習 → その年を予想）", "",
              "10番人気以下・確定オッズ。期間 %s。対数損失：市場 %.5f → モデル %.5f" % (
                  val["period"], val["logloss"]["market"], val["logloss"]["model"]), "",
              _md_table(sel.reset_index().rename(columns={"index": "選び方"}).assign(**{"頭数": lambda x: x["頭数"].astype(int), "3着内": lambda x: x["3着内"].astype(int)}),
                        {"3着内率": "%.3f", "複勝回収": "%.3f", "単勝回収": "%.3f", "平均オッズ": "%.1f"}), "",
              "### 年ごとの回収率（複勝／単勝）と頭数", "",
              "閾値を上げるほど良くなるわけではない。期待値1.0以上は件数が少なく年ごとのブレが大きいので、"
              "レポートの印は毎年安定していた層（★＝複勝期待値0.9以上、☆＝0.8以上、単＝単勝期待値1.0以上）に合わせている。", "",
              _year_md(val), ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def train(do_validate=True, log=print):
    df = build_frame(log)
    base, ft, prof, ct = research(df, log)
    sel = val = wf = None
    if do_validate:
        wf, sel, val = validate(df, log)
    b = M.fit_all(df, log=log)
    b.save()
    p = write_research_md(base, ft, prof, ct, sel, val, wf)
    log("保存：%s / %s" % (MODEL_DIR / "longshot.pkl", p))
    return df, b
