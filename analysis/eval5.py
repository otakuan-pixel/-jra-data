"""eval5.py — 第4版と第5版を同じレースで比べる（2025〜26年、四半期ごとの通し方式）。最後に第5版の MK 係数を全期間で出す"""
import sys, json, numpy as np, pandas as pd
sys.argv = ["x"]
import stage2 as S


def load(path):
    df = pd.read_csv(path, dtype={"date": str, "jo": str})
    df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
    df["rid"] = df["date"] + ":" + df["jo"] + ":" + df["rno"].astype(str)
    inv = 1.0 / df["odds"]; df["pm"] = inv / inv.groupby(df["rid"]).transform("sum")
    nw = df.assign(w=(df["place"] == 1)).groupby("rid")["w"].transform("sum")
    df = df[nw == 1].sort_values(["rid", "num"]).reset_index(drop=True)
    df["q"] = df["date"].str[:4] + "Q" + ((df["date"].str[4:6].astype(int) - 1) // 3 + 1).astype(str)
    return df


d4, d5 = load("feats.csv.gz"), load("feats5.csv.gz")
key = ["rid", "num"]
common = d4[key].merge(d5[key], on=key)
d4 = d4.merge(common, on=key).sort_values(key).reset_index(drop=True)
d5 = d5.merge(common, on=key).sort_values(key).reset_index(drop=True)
assert (d4["place"].values == d5["place"].values).all()
X4, X5 = S.build(d4), S.build(d5)
X5["mu_t"] = d5["mu_t"] - d5.groupby("rid")["mu_t"].transform("mean")
SEL = json.load(open("selected.json"))      # 第4版で2024年だけを使って選んだ15項目
SEL5 = SEL + ["mu_t"]
T = (d5["date"] >= "20250101").values
test_q = sorted(d5.loc[T, "q"].unique())
CS = {}


def rolling(df, XF, fs):
    p = np.full(len(df), np.nan)
    for q in test_q:
        tr = df.index[df["q"] < q].values; te = df.index[df["q"] == q].values
        ck = (id(df), q)
        if ck not in CS:
            CS[ck] = S.choice_sets(df.loc[tr].reset_index(drop=True), stages=1)
        Xtr = XF.loc[tr, fs].values; mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = S.fit_cl((Xtr - mu_) / sd_, CS[ck], l2=1.0)
        p[te] = S.race_softmax(((XF.loc[te, fs].values - mu_) / sd_) @ w, df.loc[te, "rid"].values)
    return p


dt_ = d5[T]
preds = {"market": dt_["pm"].values, "sim_v4": d4.loc[T, "p_sim"].values, "sim_v5": dt_["p_sim"].values,
         "MK15・第4版の指数": rolling(d4, X4, SEL)[T], "MK15・第5版の指数": rolling(d5, X5, SEL)[T],
         "MK16（＋時計重視 mu_t）・第5版": rolling(d5, X5, SEL5)[T]}
rows = [S.score(dt_, v, k) for k, v in preds.items()]
pd.set_option("display.width", 220)
print("■ 2025〜26年 %d レース（オッズ込みは四半期ごとに、それより前だけで学習）" % dt_["rid"].nunique())
print(pd.DataFrame(rows).set_index("model")[["logloss", "hit", "top3", "roi_top", "ev1.0_n", "ev1.0_roi", "ev1.1_n", "ev1.1_roi"]].round(4).to_string())

# ブートストラップ（レース単位）
w = dt_["place"].values == 1
rng = np.random.default_rng(0)
B = 4000


def boot(a, b, lab):
    diff = np.log(a[w]) - np.log(b[w])
    ix = rng.integers(0, len(diff), (B, len(diff)))
    bs = diff[ix].mean(1)
    print("  %s：%+.4f（95%%幅 %+.4f〜%+.4f、良くなっている確率 %.1f%%）" % (lab, diff.mean(), *np.percentile(bs, [2.5, 97.5]), 100 * (bs > 0).mean()))


print("\n■ 対数損失の改善（＋＝前者が良い）")
boot(preds["sim_v5"], preds["sim_v4"], "SIM 第5版 − 第4版")
boot(preds["MK16（＋時計重視 mu_t）・第5版"], preds["market"], "MK16 − 市場")
boot(preds["MK16（＋時計重視 mu_t）・第5版"], preds["MK15・第4版の指数"], "MK16（第5版） − MK15（第4版）")
p = preds["MK16（＋時計重視 mu_t）・第5版"]
ev = p * dt_["odds"].values
rids = dt_["rid"].values; ur, ri = np.unique(rids, return_inverse=True)
print("\n■ 単勝の回収率（確定オッズ・MK16）")
for th in (0.9, 1.0, 1.1):
    m = (ev >= th) & (dt_["odds"].values < 50)
    stake = np.bincount(ri[m], minlength=len(ur)).astype(float)
    ret = np.bincount(ri[m], weights=(dt_["odds"].values * w)[m], minlength=len(ur))
    ix = rng.integers(0, len(ur), (B, len(ur)))
    roi = ret[ix].sum(1) / np.maximum(stake[ix].sum(1), 1)
    print("  期待値%.1f以上・50倍未満：%d点 回収 %.0f%%（95%%幅 %.0f〜%.0f%%、100%%超の確率 %.0f%%）" % (
        th, m.sum(), 100 * ret.sum() / max(stake.sum(), 1), *(100 * np.percentile(roi, [2.5, 97.5])), 100 * (roi > 1).mean()))
m = (ev < 0.8) & (dt_["odds"].values < 10)
print("  期待値0.8未満・10倍未満（割高）：%d点 回収 %.0f%%" % (m.sum(), 100 * (dt_["odds"].values * w)[m].sum() / max(m.sum(), 1)))

# 全期間で最終の係数（エンジンの MK に入れる値。特徴は rate() の mkf と同じ名前に直す）
X = X5[SEL5].values; mu_, sd_ = X.mean(0), X.std(0) + 1e-9
wf = S.fit_cl((X - mu_) / sd_, S.choice_sets(d5, stages=1), l2=1.0) / sd_
name = {"log_pm": "lpm", "mu": "mu", "mu_t": "mu_t", "brd_vm": "brd_vm", "c_rec": "c_rec", "c_last": "c_last", "c_draw": "c_draw"}
mk = {name.get(f, f[2:] if f.startswith("k_") else f): round(float(v), 4) for f, v in zip(SEL5, wf)}
json.dump({"MK": mk, "trained_on": "%s〜%s" % (d5["date"].min(), d5["date"].max()), "races": int(d5["rid"].nunique())},
          open("mk_v5.json", "w"), ensure_ascii=False, indent=1)
print("\n■ 第5版の MK（全期間 %s〜%s・%d レースで推定）" % (d5["date"].min(), d5["date"].max(), d5["rid"].nunique()))
print(json.dumps(mk, ensure_ascii=False))
