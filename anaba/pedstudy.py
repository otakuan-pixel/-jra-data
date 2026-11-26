"""血統の検証：「奇跡の血量」などのクロスは、大穴の好走と関係があるか

症例対照（同じレースの中で比べる）：
  症例 = 10番人気以下で3着内、対照 = 同じレースの10番人気以下で4着以下（無作為抽出）
指標：
  - マンテル・ヘンツェルのオッズ比（レースごとに層別）… 粗い比較
  - ロジスティック回帰のオッズ比（log単勝オッズで補正）… 人気の差を除いた「市場が見落としているか」に近い比較
  - 95%信頼区間はレース単位のブートストラップ
オッズ比 > 1 で信頼区間が1をまたがなければ「関係あり」。
"""
import json

import numpy as np
import pandas as pd

from .backfill import CC_PATH
from .config import MODEL_DIR, REPORT_DIR
from .pedigree import PedStore


def load_cc():
    if not CC_PATH.exists():
        return pd.DataFrame()
    cc = pd.read_csv(CC_PATH, dtype={"race_id": str, "ymd": str, "jo": str, "horse_id": str})
    store = PedStore()
    rows = []
    for _, r in cc.iterrows():
        s = store.summary(r["horse_id"]) if isinstance(r["horse_id"], str) else None
        if not s:
            continue
        rows.append({**r.to_dict(), "miracle": s["miracle"], "miracle_eq": s["miracle_eq"], "max_pct": s["max_pct"],
                     "n_cross": s["n_cross"], "close": s["close"], "both": s["n_cross_both"],
                     "total": s["inbreed_total"], "sire": s["sire"], "bms": s["bms"], "sire_sire": s["sire_sire"],
                     "miracle_names": "、".join(s["miracle_names"]),
                     "cross_names": [c["name"] for c in s["crosses"]],
                     "case": 1 if r["role"] == "case" else 0})
    return pd.DataFrame(rows)


def mh_or(d, col):
    num = den = 0.0
    for _, g in d.groupby("race_id"):
        n = len(g)
        a = ((g["case"] == 1) & g[col]).sum()
        b = ((g["case"] == 1) & ~g[col]).sum()
        c = ((g["case"] == 0) & g[col]).sum()
        dd = ((g["case"] == 0) & ~g[col]).sum()
        num += a * dd / n
        den += b * c / n
    return num / den if den > 0 else np.nan


def logit_or(d, col):
    """case ~ 1 + log(odds) + col のロジスティック回帰（ニュートン法）→ col のオッズ比"""
    X = np.column_stack([np.ones(len(d)), np.log(d["odds"].values), d[col].astype(float).values])
    y = d["case"].values.astype(float)
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return float(np.exp(b[2]))


def boot_ci(d, col, fn, n=400, seed=0):
    rng = np.random.default_rng(seed)
    ids = d["race_id"].unique()
    groups = {k: g for k, g in d.groupby("race_id")}
    vals = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([groups[i] for i in pick])
        if s[col].sum() == 0:
            continue
        v = fn(s, col)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def study(min_n=30, boot=400, log=print):
    d = load_cc()
    if d.empty:
        log("症例対照データがありません（python -m anaba backfill で集める）")
        return None
    d = d[d["race_id"].map(d.groupby("race_id")["case"].agg(lambda x: x.nunique() == 2))]
    log("症例 %d 頭・対照 %d 頭（%d レース）" % ((d["case"] == 1).sum(), (d["case"] == 0).sum(), d["race_id"].nunique()))
    tests = {
        "miracle": ("奇跡の血量（3×4／4×3）", d["miracle"]),
        "miracle_eq": ("血量18.75%ちょうどのクロス（4×4×4なども含む）", d["miracle_eq"]),
        "max_ge_12": ("最も濃いクロスが12.5%以上（4×4より濃い）", d["max_pct"] >= 12.5),
        "close": ("2代以内の近い近親交配（2×3など）", d["close"]),
        "no_cross": ("5代内にクロスなし", d["n_cross"] == 0),
        "many_cross": ("クロス4本以上", d["n_cross"] >= 4),
    }
    # よく出てくる祖先ごとの「奇跡の血量」
    mir_names = d.loc[d["miracle"], "miracle_names"].str.split("、").explode().str.replace(r"\s+\d×\d$", "", regex=True)
    for name, cnt in mir_names.value_counts().items():
        if cnt >= min_n:
            tests["miracle:" + name] = ("奇跡の血量：%s" % name,
                                        d["miracle_names"].str.contains(name + " ", regex=False))
    # よく出てくるクロス（どの濃さでも）
    allx = d["cross_names"].explode().value_counts()
    for name, cnt in allx.items():
        if cnt >= max(min_n * 3, 100):
            tests["cross:" + name] = ("%sのクロスあり（濃さは問わない）" % name, d["cross_names"].map(lambda xs: name in xs))
    res = {}
    for key, (label, col) in tests.items():
        d["_x"] = col.fillna(False).astype(bool).values
        ne = int(d["_x"].sum())
        if ne < min_n:
            continue
        r = {"label": label, "n_exposed": ne, "case_rate": float(d.loc[d["case"] == 1, "_x"].mean()),
             "ctrl_rate": float(d.loc[d["case"] == 0, "_x"].mean()),
             "mh_or": float(mh_or(d, "_x")), "or": logit_or(d, "_x")}
        r["ci"] = boot_ci(d, "_x", logit_or, n=boot)
        res[key] = r
    d = d.drop(columns="_x")
    # 父ごと（症例・対照の比較。参考）
    sires = d.groupby("sire").agg(n=("case", "size"), case=("case", "sum"))
    sires = sires[sires["n"] >= min_n]
    sire_rows = []
    for s, row in sires.iterrows():
        d["_x"] = (d["sire"] == s).values
        sire_rows.append({"sire": s, "n": int(row["n"]), "case": int(row["case"]), "or": logit_or(d, "_x")})
    d = d.drop(columns="_x")
    out = {"n_case": int((d["case"] == 1).sum()), "n_ctrl": int((d["case"] == 0).sum()),
           "n_race": int(d["race_id"].nunique()), "period": "%s〜%s" % (d["ymd"].min(), d["ymd"].max()),
           "tests": res, "sires": sorted(sire_rows, key=lambda x: -x["or"])}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    eff = {}
    if "miracle" in res:
        eff["miracle"] = res["miracle"]
    eff["detail"] = out
    (MODEL_DIR / "pedigree_effect.json").write_text(json.dumps(eff, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(out, min_n=min_n)
    return out


def write_md(out, path=None, min_n=30):
    path = path or REPORT_DIR / "research" / "pedigree.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    L = ["# 血統の検証：奇跡の血量と大穴", "",
         "症例＝10番人気以下で3着内、対照＝同じレースの10番人気以下で4着以下（無作為）。期間 %s、%d レース（症例 %d 頭・対照 %d 頭）。"
         % (out["period"], out["n_race"], out["n_case"], out["n_ctrl"]), "",
         "- **オッズ比（オッズ補正）**：単勝オッズの差を除いたうえで、その血統の特徴がある馬が3着内に来やすいか。1.0より大きければ来やすい",
         "- 95%信頼区間が1.0をまたいでいれば「差があるとは言えない」", "",
         "| 特徴 | 該当頭数 | 症例での割合 | 対照での割合 | オッズ比（層別） | オッズ比（オッズ補正） | 95%信頼区間 |",
         "|---|--:|--:|--:|--:|--:|---|"]
    for k, r in sorted(out["tests"].items(), key=lambda kv: (not kv[0].startswith(("miracle", "max", "close", "no_", "many")), kv[0])):
        L.append("| %s | %d | %.1f%% | %.1f%% | %.2f | %.2f | %.2f〜%.2f |" % (
            r["label"], r["n_exposed"], 100 * r["case_rate"], 100 * r["ctrl_rate"], r["mh_or"], r["or"], r["ci"][0], r["ci"][1]))
    L += ["", "## 父ごと（参考：%d頭以上）" % min_n, "", "| 父 | 頭数 | 症例 | オッズ比（オッズ補正） |", "|---|--:|--:|--:|"]
    rows = out["sires"] if len(out["sires"]) <= 25 else out["sires"][:15] + [None] + out["sires"][-10:]
    for s in rows:
        L.append("| … | | | |" if s is None else "| %s | %d | %d | %.2f |" % (s["sire"], s["n"], s["case"], s["or"]))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
