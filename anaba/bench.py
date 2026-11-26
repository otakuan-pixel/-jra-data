"""予測方法の総当たり比較（ベンチマーク）

すべての方法を同じ条件で比べる：
  - 2023〜2026年の各年を、その年より前のデータだけで学習して予想（ウォークフォワード）
  - 予想は全頭ぶん出す（穴に絞ることが最適かどうかも比べるため）
  - 運用ルール（閾値など）は前半（2023〜24年）で選び、後半（2025〜26年）で確かめる

方法（methods）：
  mkt        市場モデル（オッズ関連の特徴だけ）
  gbm6       現行：市場＋全特徴、6番人気以下で学習
  gbm_all    同、全頭で学習
  gbm_tier   区分別に学習（中穴用＝7〜40倍、大穴用＝20倍以上）
  gbm_small  正則化を強めたGBM／gbm_big 木を大きくしたGBM
  logit      ロジスティック回帰（線形）
  benter     Benter式2段階：能力モデル（オッズを使わない）→ 市場と合成（勝率は条件付きロジット、3着内率はロジスティック）
  rank       ランキング学習（LambdaRank）→ 市場と合成
  shrink     現行の予測を市場へ縮める（縮め方は前の年で決める）
  recency    直近の年ほど重く学習
  ens        gbm6 と benter の平均
  day        gbm6 ＋ 当日の馬場傾向（その日の前のレース）
  -<群>      gbm6 から特徴の群を抜いたもの（効いているかの確認）
"""
import json
import pickle
import time

import numpy as np
import pandas as pd

from .config import CACHE, tier_series
from .features import CATEGORICAL, NUMERIC
from .model import MARKET, PARAMS, _predict, _train, fit_pay, exp_pay

YEARS = (2023, 2024, 2025, 2026)
VAL_YEARS = (2023, 2024)
TEST_YEARS = (2025, 2026)
FRAME = CACHE / "bench_frame.pkl"

GROUPS = {
    "speed": ["p1_fig", "best3_fig", "avg3_fig", "best5_fig", "p1_pace_dev", "p1_front_cost", "fig_rel", "fig_rank",
              "avgfig_rank", "p1_tdiff", "p2_tdiff", "avg3_tdiff"],
    "course": [c for c in NUMERIC if c.startswith("h_")],
    "rotation": ["interval", "since_layoff", "d_dist", "surf_change", "d_cls", "jockey_change", "d_wt", "same_jo_prev",
                 "blinker_on", "blinker_first", "hwd", "hw"],
    "people": [c for c in NUMERIC if c.startswith(("jk_", "tr_", "trs_", "jkjo_", "br_"))],
    "form": ["p1_place", "p1_pop", "p1_odds", "p1_place_pct", "p1_early", "p1_c4", "p1_gain", "p1_beat_mkt",
             "p2_place_pct", "p3_place_pct", "avg3_place_pct", "avg3_early", "avg3_gain", "best5_place_pct",
             "avg5_beat_mkt", "form_rank", "early_rank", "n_front"],
}
DAY = [c for c in NUMERIC if c.startswith("day_")]
BASE = [c for c in NUMERIC if not c.startswith(("sire_", "bms_", "day_"))]
FUND = [c for c in BASE if c not in MARKET] + ["heads"]


def frame(rebuild=False, log=print):
    if FRAME.exists() and not rebuild:
        with FRAME.open("rb") as f:
            return pickle.load(f)
    from .train import build_frame
    df = build_frame(log)
    df = df[df["done"] == 1].reset_index(drop=True)
    df["tier"] = tier_series(df)
    df["logit_mkt_win"] = np.log(df["mkt_win"].clip(1e-6, 1 - 1e-6))
    FRAME.parent.mkdir(parents=True, exist_ok=True)
    with FRAME.open("wb") as f:
        pickle.dump(df, f)
    return df


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def race_softmax(score, race_codes):
    s = pd.Series(score)
    m = s.groupby(race_codes).transform("max")
    e = np.exp(s - m)
    return (e / e.groupby(race_codes).transform("sum")).values


def cond_logit_fit(X, race_codes, win):
    """条件付きロジット（1レース＝1つの選択）：P(i 1着) ∝ exp(X_i·w) → w"""
    from scipy.optimize import minimize
    order = np.argsort(race_codes, kind="stable")
    X, rc, y = X[order], race_codes[order], win[order]
    starts = np.r_[0, np.flatnonzero(np.diff(rc)) + 1]

    sizes = np.diff(np.r_[starts, len(rc)])

    def nll(w):
        z = X @ w
        zz = z - np.repeat(np.maximum.reduceat(z, starts), sizes)
        lse = np.log(np.add.reduceat(np.exp(zz), starts))
        lp = zz - np.repeat(lse, sizes)
        p = np.exp(lp)
        # 勾配：Σ_レース（1着馬の x − 1着の数 × Σ p_i x_i）
        px = np.add.reduceat(p[:, None] * X, starts)
        xw = np.add.reduceat(y[:, None] * X, starts)
        nw = np.add.reduceat(y, starts)[:, None]
        return -(lp * y).sum(), -(xw - px * nw).sum(axis=0)
    r = minimize(nll, np.ones(X.shape[1]), jac=True, method="L-BFGS-B")
    return r.x


def place_from_win(p, race_codes, lam2=0.81, lam3=0.65):
    """勝率 → 3着内率（Harville を Lo–Bacon-Shone 型に補正：2着・3着は勝率を lam 乗して平らにする）"""
    out = np.zeros(len(p))
    for _, idx in pd.Series(np.arange(len(p))).groupby(race_codes).indices.items():
        q = p[idx] / p[idx].sum()
        n = len(q)
        if n <= 3:
            out[idx] = 1.0
            continue
        q2 = q ** lam2
        q3 = q ** lam3
        tot = q.copy()
        for i in range(n):          # i が1着
            w2 = q2.copy()
            w2[i] = 0
            w2 = w2 / w2.sum()
            tot += q[i] * w2            # 2着
            for j in range(n):      # j が2着
                if j == i or w2[j] == 0:
                    continue
                w3 = q3.copy()
                w3[[i, j]] = 0
                s3 = w3.sum()
                if s3 > 0:
                    tot += q[i] * w2[j] * w3 / s3   # 3着
        out[idx] = np.clip(tot, 0, 1)
    return out


# ───────────────────────── 各方法の学習と予想 ─────────────────────────
def _gbm(tr, te, cols, target="top3", params=None, rounds=None, weight=None):
    import lightgbm as lgb
    if params is None and rounds is None and weight is None:
        m = _train(tr, cols + CATEGORICAL, target=target)
        return _predict(m, te, cols + CATEGORICAL)
    p = dict(PARAMS)
    p.update(params or {})
    X = tr[cols].copy()
    Xt = te[cols].copy()
    ds = lgb.Dataset(X, tr[target].values, weight=weight, free_raw_data=False)
    m = lgb.train(p, ds, num_boost_round=rounds or 500)
    return m.predict(Xt)


def predict_method(name, df, y, prev=None):
    """name の方法で year=y を予想 → DataFrame(index=te.index, p_top3, [p_win])"""
    tr = df[df["year"] < y]
    te = df[df["year"] == y]
    out = pd.DataFrame(index=te.index)
    tr6 = tr[tr["pop"] >= 6]
    if name == "mkt":
        out["p_top3"] = _gbm(tr, te, MARKET)
        out["p_win"] = _gbm(tr, te, MARKET, "win")
    elif name == "gbm6":
        out["p_top3"] = _gbm(tr6, te, BASE)
        out["p_win"] = _gbm(tr6, te, BASE, "win")
    elif name == "gbm_all":
        out["p_top3"] = _gbm(tr, te, BASE)
        out["p_win"] = _gbm(tr, te, BASE, "win")
    elif name == "gbm_tier":
        a = tr[(tr["odds"] >= 7) & (tr["odds"] < 40) & (tr["pop"] >= 5)]
        b = tr[(tr["odds"] >= 20) & (tr["pop"] >= 5)]
        pa, pb = _gbm(a, te, BASE), _gbm(b, te, BASE)
        wa, wb = _gbm(a, te, BASE, "win"), _gbm(b, te, BASE, "win")
        big = (te["odds"] >= 30).values
        out["p_top3"] = np.where(big, pb, pa)
        out["p_win"] = np.where(big, wb, wa)
    elif name == "gbm_small":
        prm = dict(num_leaves=15, min_child_samples=800, feature_fraction=0.6, lambda_l2=20.0)
        out["p_top3"] = _gbm(tr6, te, BASE, params=prm, rounds=400)
        out["p_win"] = _gbm(tr6, te, BASE, "win", params=prm, rounds=400)
    elif name == "gbm_big":
        prm = dict(num_leaves=63, min_child_samples=80, lambda_l2=1.0)
        out["p_top3"] = _gbm(tr6, te, BASE, params=prm, rounds=900)
        out["p_win"] = _gbm(tr6, te, BASE, "win", params=prm, rounds=900)
    elif name == "recency":
        w = 0.6 ** (y - 1 - tr6["year"]).clip(lower=0).values
        out["p_top3"] = _gbm(tr6, te, BASE, weight=w, params={}, rounds=500)
        out["p_win"] = _gbm(tr6, te, BASE, "win", weight=w, params={}, rounds=500)
    elif name == "day":
        out["p_top3"] = _gbm(tr6, te, BASE + DAY)
        out["p_win"] = _gbm(tr6, te, BASE + DAY, "win")
    elif name.startswith("-"):
        cols = [c for c in BASE if c not in GROUPS[name[1:]]]
        out["p_top3"] = _gbm(tr6, te, cols)
        out["p_win"] = _gbm(tr6, te, cols, "win")
    elif name == "logit":
        out["p_top3"], out["p_win"] = _logit_model(tr6, te)
    elif name in ("benter", "rank"):
        out["p_top3"], out["p_win"] = _two_stage(df, y, kind=name)
    elif name == "shrink":
        # gbm6 の予想を市場へ縮める：logit(p) = w·logit(p_gbm6) + (1−w)·logit(p_mkt)、w は前の年（y−1）で決める
        g, m = prev["gbm6"], prev["mkt"]
        cur_g, cur_m = g.loc[te.index], m.loc[te.index]
        ws = np.linspace(0, 1.5, 31)
        pv = g.index[df.loc[g.index, "year"] == y - 1]
        if len(pv):
            yy = df.loc[pv, "top3"].values
            ll = [_ll(yy, _sig(w * _logit(g.loc[pv, "p_top3"]) + (1 - w) * _logit(m.loc[pv, "p_top3"]))) for w in ws]
            w = ws[int(np.argmin(ll))]
        else:
            w = 1.0
        out["p_top3"] = _sig(w * _logit(cur_g["p_top3"]) + (1 - w) * _logit(cur_m["p_top3"]))
        out["p_win"] = _sig(w * _logit(cur_g["p_win"]) + (1 - w) * _logit(cur_m["p_win"]))
        out.attrs["w"] = float(w)
    elif name == "ens":
        a, b = prev["gbm6"].loc[te.index], prev["benter"].loc[te.index]
        out["p_top3"] = (a["p_top3"] + b["p_top3"]) / 2
        out["p_win"] = (a["p_win"] + b["p_win"]) / 2
    else:
        raise ValueError(name)
    return out


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def _logit_model(tr, te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols = BASE
    med = tr[cols].median()

    def X(d):
        x = d[cols].fillna(med)
        miss = d[cols].isna().astype(float).add_suffix("_na")
        return np.column_stack([x.values, miss.values, _logit(d["mkt_top3"].values), np.log(d["odds"].values)])
    sc = StandardScaler().fit(X(tr))
    res = []
    for tgt in ("top3", "win"):
        m = LogisticRegression(C=0.1, max_iter=400)
        m.fit(sc.transform(X(tr)), tr[tgt].values)
        res.append(m.predict_proba(sc.transform(X(te)))[:, 1])
    return res[0], res[1]


def _stage1(tr, te, kind):
    """能力モデル（オッズの特徴を使わない）→ te のスコア（勝ち・3着内）"""
    if kind == "benter":
        return _gbm(tr, te, FUND, "win"), _gbm(tr, te, FUND, "top3")
    import lightgbm as lgb
    d = tr.sort_values(["race_id", "num"])
    rel = np.where(d["place"] == 1, 3, np.where(d["place"] == 2, 2, np.where(d["place"] == 3, 1, 0)))
    grp = d.groupby("race_id", sort=False).size().values
    p = dict(PARAMS)
    p.update(objective="lambdarank", metric="ndcg", eval_at=[3])
    m = lgb.train(p, lgb.Dataset(d[FUND], rel, group=grp), num_boost_round=400)
    s = m.predict(te[FUND])
    return s, s


def _two_stage(df, y, kind):
    """Benter式：段1（能力）を y−1 年で外挿予想 → 段2（市場との合成）を y−1 年で当てはめ → y 年に適用"""
    tr1, va = df[df["year"] < y - 1], df[df["year"] == y - 1]
    tr, te = df[df["year"] < y], df[df["year"] == y]
    sw_va, s3_va = _stage1(tr1, va, kind)
    sw_te, s3_te = _stage1(tr, te, kind)

    def feats(d, sw):
        rc = d["race_id"].values
        lf = np.log(race_softmax(_logit(sw) if kind == "benter" else sw, rc))
        return np.column_stack([lf, np.log(d["mkt_win"].clip(1e-6).values)])
    # 勝率：条件付きロジット
    Xv = feats(va, sw_va)
    rcv = pd.factorize(va["race_id"])[0]
    w = cond_logit_fit(Xv, rcv, va["win"].values.astype(float))
    Xt = feats(te, sw_te)
    rct = pd.factorize(te["race_id"])[0]
    p_win = race_softmax(Xt @ w, rct)
    # 3着内率：段1の3着内スコアと市場の3着内率をロジスティックで合成
    from sklearn.linear_model import LogisticRegression
    a_v = _logit(s3_va) if kind == "benter" else s3_va
    a_t = _logit(s3_te) if kind == "benter" else s3_te
    Zv = np.column_stack([a_v, _logit(va["mkt_top3"].values), np.log(p_win_from(Xv, w, rcv))])
    Zt = np.column_stack([a_t, _logit(te["mkt_top3"].values), np.log(p_win)])
    lr = LogisticRegression(C=10.0, max_iter=400).fit(Zv, va["top3"].values)
    return lr.predict_proba(Zt)[:, 1], p_win


def p_win_from(X, w, rc):
    return race_softmax(X @ w, rc)


# ───────────────────────── 実行 ─────────────────────────
ORDER = ["mkt", "gbm6", "gbm_all", "gbm_tier", "gbm_small", "gbm_big", "logit", "benter", "rank", "recency", "day",
         "-speed", "-course", "-rotation", "-people", "-form", "shrink", "ens"]


def run(methods=None, years=YEARS, log=print):
    df = frame(log=log)
    methods = methods or ORDER
    preds = {}
    store = CACHE / "bench_preds.pkl"
    if store.exists():
        with store.open("rb") as f:
            preds = pickle.load(f)
    for name in methods:
        if name in preds:
            continue
        t0 = time.time()
        parts = []
        for y in years:
            parts.append(predict_method(name, df, y, prev=preds))
        preds[name] = pd.concat(parts)
        with store.open("wb") as f:
            pickle.dump(preds, f)
        log("  %-10s %.0f 秒" % (name, time.time() - t0))
    return df, preds


def pay_model_by_year(df, years=YEARS):
    """複勝払戻の見込み（年ごとに、その年より前の3着内馬で学習）"""
    out = pd.Series(np.nan, index=df.index)
    for y in years:
        tr = df[(df["year"] < y) & (df["top3"] == 1)]
        m = fit_pay(tr)
        te = df["year"] == y
        out[te] = exp_pay(m, df.loc[te, "odds"].values, df.loc[te, "heads"].values)
    return out


def save_json(obj, name):
    p = CACHE / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
