#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage2.py — 2段目のモデル（条件付きロジット／LightGBM）を四半期ごとの通し方式で学習・採点する

各テスト四半期について、それより前のレースだけで学習し、その四半期を予想して採点する（未来のデータは使わない）。
比べるもの：
  market   確定単勝オッズから見た勝率（1/オッズを合計1に）
  sim_v4   第4版SIMの勝率
  mk_v4    第4版のオッズ込み勝率（MK。係数が2025〜26年で推定済みなので、この期間は有利に出る＝参考）
  cl_sim   2段目・オッズなし（SIMの中身だけで学習）
  cl_mkt   2段目・オッズあり（市場＋SIMの中身）
  gbm_*    LightGBM版（任意）

使い方: python3 -I stage2.py --feats feats.csv.gz [--test-from 20250101] [--gbm] [--save-coef coef.json]
"""
import argparse, json, math, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

WET = {"良": 0.0, "稍重": 0.45, "重": 0.8, "不良": 1.0}
COMP = ["rec", "west", "wt_now", "age", "fem", "jd", "layoff", "jk_wr", "jk_vm", "gate", "draw", "n2", "tr_young",
        "tr_debut", "winrate", "front", "last", "trend", "drop", "maxcls", "quick", "short", "hw", "course", "new_surf", "new_dist"]
MKF = ["jk_wr", "closer", "n1", "a3", "tr_wr", "l10", "l6_9", "drop", "quick"]


# ───────────────────────── 特徴量 ─────────────────────────
def build(df):
    """1頭1行の DataFrame → 特徴量の表（列名→配列）"""
    X = {}
    g = df.groupby("rid")
    lp = np.log(df["p_sim"].clip(1e-4))
    X["log_p_sim"] = lp
    X["mu"] = df["mu"] - g["mu"].transform("mean")
    X["mu0"] = df["mu0"] - g["mu0"].transform("mean")
    for c in COMP:
        X["c_" + c] = df["c_" + c]
    for c in MKF:
        X["k_" + c] = df["k_" + c]
    wet = df["cond"].map(WET).fillna(0.0)
    X["apt_wet"] = df["apt"] * wet
    X["e"] = df["e"] - 0.5
    X["e_fidx"] = (df["e"] - 0.5) * df["fidx"]
    X["e_m"] = (df["e"] - 0.5) * (df["m"] - 12) / 6.0
    sd = df["f_sd"]
    X["f_sd"] = sd.fillna(sd.median()) - sd.median()
    X["f_sd_na"] = sd.isna().astype(float)
    X["log_runs"] = np.log1p(df["n_runs"])
    X["f_last_rel"] = (df["f_last"] - df["mu0"]).fillna(0.0).clip(-2, 2)
    X["f_top_rel"] = (df["f_top"] - df["mu0"]).fillna(0.0).clip(-2, 2)
    lay = df["lay"].fillna(365)
    X["lay_log"] = np.log1p(lay) - math.log(29)
    X["lay_long"] = (lay > 180).astype(float)
    # 新しい候補
    X["b_first"] = df["b_first"].astype(float)
    X["b_off"] = df["b_off"].astype(float)
    X["b_now"] = df["b_now"].astype(float)
    hd = df["hw_diff"]
    X["hw_diff"] = hd.fillna(0).clip(-20, 20) / 10.0
    X["hw_diff_abs"] = hd.fillna(0).abs().clip(0, 30) / 10.0
    X["jk_change"] = df["jk_change"].astype(float)
    X["own_vm"] = np.log((df["own_w"] + 5.0) / (df["own_x"] + 5.0))
    X["brd_vm"] = np.log((df["brd_w"] + 5.0) / (df["brd_x"] + 5.0))
    X["own_wr"] = np.log((df["own_w"] + 3.5) / (df["own_n"] + 50.0) / 0.07)
    X["brd_wr"] = np.log((df["brd_w"] + 3.5) / (df["brd_n"] + 50.0) / 0.07)
    dp = df["dist_prev"]
    X["dist_chg"] = (np.log(df["dist"] / dp.where(dp > 0))).fillna(0.0).clip(-0.5, 0.5)
    X["surf_chg"] = ((df["surf_prev"].notna()) & (df["surf_prev"] != df["surf"])).astype(float)
    # 市場
    X["log_pm"] = np.log(df["pm"])
    return pd.DataFrame(X)


SIM_ONLY = None   # 下で決める
NEW = ["b_first", "b_off", "b_now", "hw_diff", "hw_diff_abs", "jk_change", "own_vm", "brd_vm", "own_wr", "brd_wr", "dist_chg", "surf_chg"]


# ───────────────────────── 条件付きロジット（上位3着まで使う Plackett–Luce） ─────────────────────────
def choice_sets(df, stages=3, stage_w=(1.0, 0.6, 0.4)):
    """(各選択の行番号の配列, 選ばれた行番号, 重み) のリスト → まとめて CSR 風の配列で返す"""
    rows, starts, chosen, wts = [], [], [], []
    for rid, idx in df.groupby("rid").indices.items():
        pl = df["place"].values[idx]
        order = idx[np.argsort(pl, kind="stable")]
        plo = np.sort(pl)
        avail = list(idx)
        for s in range(min(stages, len(idx) - 1)):
            if plo[s] != s + 1:   # 同着・欠けはそこで打ち切り
                break
            y = order[s]
            starts.append(len(rows)); rows.extend(avail); chosen.append(len(rows) - len(avail) + avail.index(y)); wts.append(stage_w[s])
            avail.remove(y)
    starts.append(len(rows))
    return np.array(rows), np.array(starts), np.array(chosen), np.array(wts)


def fit_cl(X, cs, l2=1.0, w0=None):
    rows, starts, chosen, wts = cs
    Z = X[rows]
    seg = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    nset = len(starts) - 1
    W = wts.sum()

    def f(w):
        s = Z @ w
        mx = np.maximum.reduceat(s, starts[:-1])
        ex = np.exp(s - mx[seg])
        den = np.add.reduceat(ex, starts[:-1])
        lse = np.log(den) + mx
        ll = (wts * (s[chosen] - lse)).sum()
        p = ex / den[seg]
        gw = wts[seg] * p
        grad = (Z[chosen] * wts[:, None]).sum(0) - (Z * gw[:, None]).sum(0)
        return -ll / W + l2 * (w @ w) / W, -grad / W + 2 * l2 * w / W

    w0 = np.zeros(X.shape[1]) if w0 is None else w0
    r = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return r.x


COMBO = ["log_pm", "log_pcl", "log_pm2"]


def combo_X(df, pcl):
    lpm = np.log(df["pm"].values)
    lpc = np.log(np.clip(np.nan_to_num(pcl, nan=1e-3), 1e-4, 1))
    return np.column_stack([lpm, lpc, (lpm + 2.5) ** 2 / 4.0])


def race_softmax(score, rid_codes):
    s = pd.Series(score)
    mx = s.groupby(rid_codes).transform("max")
    ex = np.exp(s - mx)
    return (ex / ex.groupby(rid_codes).transform("sum")).values


# ───────────────────────── 採点 ─────────────────────────
def score(df, p, lab):
    d = df.assign(p=p)
    w = d[d["place"] == 1]
    ll = -np.log(w["p"].clip(1e-6)).mean()
    top = d.loc[d.groupby("rid")["p"].idxmax()]
    hit = (top["place"] == 1).mean(); t3 = (top["place"] <= 3).mean()
    roi = (top["odds"] * (top["place"] == 1)).mean()
    ev = d["p"] * d["odds"]
    out = {"model": lab, "races": d["rid"].nunique(), "logloss": ll, "hit": hit, "top3": t3, "roi_top": roi}
    for th, lim in ((1.0, 50), (1.1, 50), (1.2, 30)):
        b = d[(ev >= th) & (d["odds"] < lim)]
        out["ev%.1f_n" % th] = len(b)
        out["ev%.1f_roi" % th] = (b["odds"] * (b["place"] == 1)).sum() / max(len(b), 1)
    return out


def calib(df, p, lab, bins=(0, .02, .05, .1, .15, .2, .3, .4, .6, 1.01)):
    d = df.assign(p=p, y=(df["place"] == 1).astype(float))
    d["b"] = pd.cut(d["p"], bins)
    t = d.groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean"))
    print("  当たり具合 %s" % lab)
    for b, r in t.iterrows():
        print("    %-14s %6d頭  予測 %5.1f%%  実際 %5.1f%%" % (b, r["n"], 100 * r["pred"], 100 * r["act"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--test-from", default="20250101")
    ap.add_argument("--oos-from", default="20240101", help="cl_sim を通しで出し始める日（市場との合わせ方の学習に使う）")
    ap.add_argument("--l2", type=float, default=5.0)
    ap.add_argument("--gbm", action="store_true")
    ap.add_argument("--no-new", action="store_true", help="新しい候補の特徴を使わない（効果を測る用）")
    ap.add_argument("--save-coef")
    ap.add_argument("--dump-pred")
    a = ap.parse_args()

    df = pd.read_csv(a.feats, dtype={"date": str, "jo": str})
    df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
    df["rid"] = df["date"] + ":" + df["jo"] + ":" + df["rno"].astype(str)
    inv = 1.0 / df["odds"]
    df["pm"] = inv / inv.groupby(df["rid"]).transform("sum")
    # 1着がちょうど1頭のレースだけ
    nw = df.assign(w=(df["place"] == 1)).groupby("rid")["w"].transform("sum")
    df = df[nw == 1].reset_index(drop=True)
    df["q"] = df["date"].str[:4] + "Q" + ((df["date"].str[4:6].astype(int) - 1) // 3 + 1).astype(str)
    XF = build(df)
    feats_all = list(XF.columns)
    sim_feats = [c for c in feats_all if c != "log_pm" and not (a.no_new and c in NEW)]
    mkt_feats = sim_feats + ["log_pm"]
    print("頭数 %d・レース %d（%s〜%s）・特徴 %d" % (len(df), df["rid"].nunique(), df["date"].min(), df["date"].max(), len(mkt_feats)))

    test_q = sorted(df.loc[df["date"] >= a.test_from, "q"].unique())
    oos_q = sorted(df.loc[df["date"] >= a.oos_from, "q"].unique())
    pred = {k: np.full(len(df), np.nan) for k in ("cl_sim", "cl_mkt", "combo", "gbm_sim", "gbm_mkt")}
    coefs = {}
    for q in oos_q:
        tr = df.index[df["q"] < q].values
        te = df.index[df["q"] == q].values
        dtr = df.loc[tr]
        for lab, fs in (("cl_sim", sim_feats), ("cl_mkt", mkt_feats)):
            if lab != "cl_sim" and q not in test_q:
                continue
            Xtr = XF.loc[tr, fs].values
            mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
            cs = choice_sets(dtr.reset_index(drop=True))
            w = fit_cl((Xtr - mu_) / sd_, cs, l2=a.l2)
            s = ((XF.loc[te, fs].values - mu_) / sd_) @ w
            pred[lab][te] = race_softmax(s, df.loc[te, "rid"].values)
            coefs[lab] = dict(zip(fs, (w / sd_).tolist()))
            coefs[lab + "_std"] = dict(zip(fs, w.tolist()))
        # 2段目＋市場（ベンター方式）：それより前の四半期の「通しで出した cl_sim の勝率」と市場の勝率を、少数の係数で合わせる
        if q in test_q:
            ctr = df.index[(df["q"] < q) & ~np.isnan(pred["cl_sim"])].values
            Xc = combo_X(df, pred["cl_sim"])
            cs = choice_sets(df.loc[ctr].reset_index(drop=True), stages=1)
            w = fit_cl(Xc[ctr], cs, l2=0.1)
            pred["combo"][te] = race_softmax(Xc[te] @ w, df.loc[te, "rid"].values)
            coefs["combo"] = dict(zip(COMBO, w.tolist()))
        if a.gbm and q in test_q:
            import lightgbm as lgb
            for lab, fs in (("gbm_sim", sim_feats), ("gbm_mkt", mkt_feats)):
                dtr_s = dtr.sort_values("rid")
                grp = dtr_s.groupby("rid", sort=False).size().values
                rel = np.clip(4 - dtr_s["place"].values, 0, 3)
                mdl = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=31,
                                     min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
                mdl.fit(XF.loc[dtr_s.index, fs].values, rel, group=grp)
                # 順位の点数 → 学習データで温度を1つ合わせて勝率にする
                s_tr = mdl.predict(XF.loc[tr, fs].values)
                best = min((sum(-np.log(race_softmax(t * s_tr, dtr["rid"].values)[dtr["place"].values == 1].clip(1e-6))), t)
                           for t in np.linspace(0.3, 3.0, 28))[1]
                pred[lab][te] = race_softmax(best * mdl.predict(XF.loc[te, fs].values), df.loc[te, "rid"].values)
        print("  %s 学習 %d 頭 → テスト %d 頭" % (q, len(tr), len(te)), flush=True)

    T = df["date"] >= a.test_from
    dt_ = df[T]
    res = [score(dt_, dt_["pm"].values, "market（確定オッズ）"), score(dt_, dt_["p_sim"].values, "sim_v4（第4版SIM）")]
    if dt_["p_mk"].notna().all():
        res.append(score(dt_, dt_["p_mk"].values, "mk_v4（第4版オッズ込み・参考）"))
    for k in ("cl_sim", "cl_mkt", "combo", "gbm_sim", "gbm_mkt"):
        if not np.isnan(pred[k][T.values]).all():
            res.append(score(dt_, pred[k][T.values], k))
    r = pd.DataFrame(res).set_index("model")
    pd.set_option("display.width", 200)
    print("\n■ テスト期間 %s〜（四半期ごとに、それより前だけで学習）" % a.test_from)
    print(r.round(4).to_string())
    for y in sorted(dt_["date"].str[:4].unique()):
        m = (dt_["date"].str[:4] == y).values
        print("\n  %s年" % y)
        rr = [score(dt_[m], dt_["pm"].values[m], "market"), score(dt_[m], dt_["p_sim"].values[m], "sim_v4")]
        for k in ("cl_sim", "cl_mkt", "combo", "gbm_sim", "gbm_mkt"):
            if not np.isnan(pred[k][T.values]).all():
                rr.append(score(dt_[m], pred[k][T.values][m], k))
        print(pd.DataFrame(rr).set_index("model")[["races", "logloss", "hit", "top3", "roi_top", "ev1.0_n", "ev1.0_roi"]].round(4).to_string())
    print()
    calib(dt_, dt_["p_sim"].values, "sim_v4")
    calib(dt_, pred["cl_sim"][T.values], "cl_sim")
    calib(dt_, pred["cl_mkt"][T.values], "cl_mkt")
    calib(dt_, pred["combo"][T.values], "combo")
    print("\n■ 最後の四半期の係数（標準化後。大きいほど効いている）")
    for lab in ("cl_sim", "cl_mkt"):
        cs_ = sorted(coefs[lab + "_std"].items(), key=lambda kv: -abs(kv[1]))
        print("  %s: " % lab + "  ".join("%s %+.3f" % kv for kv in cs_[:25]))
    print("  combo: " + "  ".join("%s %+.3f" % kv for kv in coefs.get("combo", {}).items()))
    if a.save_coef:
        json.dump(coefs, open(a.save_coef, "w"), ensure_ascii=False, indent=1)
    if a.dump_pred:
        out = dt_[["rid", "num", "horse", "place", "odds", "pm", "p_sim"]].copy()
        for k in pred:
            out[k] = pred[k][T.values]
        out.to_csv(a.dump_pred, index=False)


if __name__ == "__main__":
    main()
