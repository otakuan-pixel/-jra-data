"""大穴モデル：3着内確率を「オッズ＋過去の条件」から推定し、市場とのズレ（期待値）を出す

- 学習データ：6番人気以下の馬（大穴に近い馬の傾向に絞る）
- 2本立て：
    市場モデル  … オッズ関連の特徴だけ（= 市場が考える3着内確率。Harvilleの偏りを補正したもの）
    穴モデル    … 市場の特徴 ＋ 馬の過去・臨戦過程・騎手/調教師（features.NUMERIC）
  穴モデル ÷ 市場モデル = 「市場が見落としている度合い」（エッジ）
- 複勝の期待値 = 穴モデルの3着内確率 × 3着内に来たときの複勝払戻の見込み（オッズと頭数から）
- 検証は「その年より前のデータで学習 → その年を予想」を年ごとに繰り返す（ウォークフォワード）
"""
import json
import pickle

import numpy as np
import pandas as pd

from .config import MARK_RULES, MODEL_DIR, tier_series
from .features import CATEGORICAL, NUMERIC

TRAIN_POP = 6
MARKET = ["log_odds", "pop", "pop_pct", "mkt_win", "mkt_top3", "fav_odds", "n_under10", "overround_top", "heads"]
# 父・母の父の集計は「2026年秋に出走した馬」しか分からず、生き残った馬に偏る（＝成績が良く見える）ので使わない。
# 出走前に毎週全頭の血統を取るようになってから（偏りが無くなってから）加える。
# 当日の先行レースの傾向（day_*）は、本番の発走30分前には成績データにまだ入っていないので使わない（学習と本番の条件をそろえる）
EXCLUDE = {c for c in NUMERIC if c.startswith(("sire_", "bms_", "day_"))}
FEATURES = [c for c in NUMERIC if c not in EXCLUDE]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=7)
ROUNDS = 500


def _X(df, cols):
    X = df[cols].copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def _train(df, cols, target="top3", rounds=ROUNDS):
    import lightgbm as lgb
    cats = [c for c in CATEGORICAL if c in cols]
    X = _X(df, cols)
    for c in cats:
        X[c] = pd.Categorical(X[c], categories=sorted(df[c].dropna().unique()))
    ds = lgb.Dataset(X, df[target].values, categorical_feature=cats, free_raw_data=False)
    m = lgb.train(PARAMS, ds, num_boost_round=rounds)
    m._cats = {c: list(X[c].cat.categories) for c in cats}
    return m


def _predict(m, df, cols):
    X = _X(df, cols)
    for c, cats in getattr(m, "_cats", {}).items():
        X[c] = pd.Categorical(X[c], categories=cats)
    return m.predict(X)


def fit_pay(placers):
    """3着内に来たときの複勝払戻の見込み：log(オッズ)・頭数 → 払戻（円）"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = placers[placers["place_pay"] > 0]
    m = HistGradientBoostingRegressor(loss="gamma", max_iter=200, learning_rate=0.05, random_state=0)
    m.fit(np.column_stack([np.log(d["odds"]), d["heads"]]), d["place_pay"] / 100.0)
    return m


def exp_pay(m, odds, heads):
    return 100.0 * m.predict(np.column_stack([np.log(np.clip(odds, 1.0, None)), heads]))


class Bundle:
    """学習済みモデル一式"""

    def __init__(self, market, longshot, win, pay, meta):
        self.market, self.longshot, self.win, self.pay, self.meta = market, longshot, win, pay, meta

    def score(self, df):
        out = pd.DataFrame(index=df.index)
        out["p_mkt"] = _predict(self.market, df, MARKET + CATEGORICAL)
        out["p_top3"] = _predict(self.longshot, df, FEATURES + CATEGORICAL)
        out["p_win"] = _predict(self.win, df, FEATURES + CATEGORICAL)
        out["edge"] = out["p_top3"] / out["p_mkt"]
        out["exp_place_pay"] = exp_pay(self.pay, df["odds"].values, df["heads"].values)
        out["ev_place"] = out["p_top3"] * out["exp_place_pay"] / 100.0
        out["ev_win"] = out["p_win"] * df["odds"].values
        return out

    def save(self, path=MODEL_DIR / "longshot.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        (path.parent / "longshot_meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def load(path=MODEL_DIR / "longshot.pkl"):
        with path.open("rb") as f:
            return pickle.load(f)


def fit_all(df, until=None, log=print):
    d = df[(df["done"] == 1) & (df["pop"] >= TRAIN_POP)]
    if until:
        d = d[d["ymd"] < until]
    market = _train(d, MARKET + CATEGORICAL)
    longshot = _train(d, FEATURES + CATEGORICAL)
    win = _train(d, FEATURES + CATEGORICAL, target="win")
    pay = fit_pay(d[d["top3"] == 1])
    meta = {"trained_until": d["ymd"].max(), "n_train": int(len(d)), "train_pop": TRAIN_POP,
            "features": FEATURES}
    log("学習：%d 頭（%s〜%s）" % (len(d), d["ymd"].min(), d["ymd"].max()))
    return Bundle(market, longshot, win, pay, meta)


def importance(bundle, top=25):
    m = bundle.longshot
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    return (imp / imp.sum()).head(top)


# ───────────────────────── 検証（ウォークフォワード） ─────────────────────────
def walk_forward(df, years=(2023, 2024, 2025, 2026), log=print):
    outs = []
    for y in years:
        b = fit_all(df, until="%d0101" % y, log=log)
        te = df[(df["done"] == 1) & (df["year"] == y) & (df["pop"] >= TRAIN_POP)].copy()
        sc = b.score(te)
        te = te.join(sc)
        outs.append(te)
        log("  %d年：検証 %d 頭" % (y, len(te)))
    return pd.concat(outs)


def _roi(d):
    return pd.Series({"頭数": len(d), "3着内": int(d["top3"].sum()), "3着内率": d["top3"].mean(),
                      "複勝回収": d["place_pay"].mean() / 100 if len(d) else np.nan,
                      "単勝回収": (d["win"] * d["odds"]).mean() if len(d) else np.nan,
                      "平均オッズ": d["odds"].mean()})


def selections(d, tier):
    """選び方の一覧 → {名前: 真偽}（印の基準を含む）"""
    sels = {"全部": pd.Series(True, index=d.index)}
    for mk, col, thr, bet in MARK_RULES[tier]:
        sels["印%s（%s期待値%.1f以上）" % (mk, "単勝" if col == "ev_win" else "複勝", thr)] = d[col] >= thr
    for t in (0.8, 0.9, 1.0, 1.1):
        sels["複勝期待値≥%.1f" % t] = d["ev_place"] >= t
    for t in (1.0, 1.1, 1.2):
        sels["単勝期待値≥%.1f" % t] = d["ev_win"] >= t
    return sels


def selection_table(wf, tier):
    d = wf[tier_series(wf) == tier]
    return pd.DataFrame({k: _roi(d[m]) for k, m in selections(d, tier).items()}).T


def by_year(wf, tier, col, thr):
    d = wf[(tier_series(wf) == tier) & (wf[col] >= thr)]
    return d.groupby("year").apply(_roi, include_groups=False)


def calibration(wf, tier, bins=10):
    d = wf[tier_series(wf) == tier].copy()
    d["bin"] = pd.qcut(d["p_top3"], bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(n=("top3", "size"), pred=("p_top3", "mean"),
                                               mkt=("p_mkt", "mean"), actual=("top3", "mean"))


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
