"""MKの11項目を出発点に、2024年（検証用）で効く特徴を1つずつ足し、決めた組を2025〜26年（テスト）で採点する"""
import sys, json, numpy as np, pandas as pd
sys.argv = ["x"]
import stage2 as S
df = pd.read_csv("feats.csv.gz", dtype={"date": str, "jo": str})
df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
df["rid"] = df["date"] + ":" + df["jo"] + ":" + df["rno"].astype(str)
inv = 1.0 / df["odds"]; df["pm"] = inv / inv.groupby(df["rid"]).transform("sum")
nw = df.assign(w=(df["place"] == 1)).groupby("rid")["w"].transform("sum")
df = df[nw == 1].reset_index(drop=True)
df["q"] = df["date"].str[:4] + "Q" + ((df["date"].str[4:6].astype(int) - 1) // 3 + 1).astype(str)
XF = S.build(df)
XF["log_pm2"] = (XF["log_pm"] + 2.5) ** 2 / 4.0
XF["lpm_x_mu"] = XF["log_pm"] * XF["mu"]
MK = ["log_pm", "mu"] + ["k_" + k for k in S.MKF]
CAND = [c for c in XF.columns if c not in MK]
CS = {}
def rolling(fs, period):
    lo, hi = period
    qs = sorted(df.loc[(df["date"] >= lo) & (df["date"] <= hi), "q"].unique())
    p = np.full(len(df), np.nan)
    for q in qs:
        tr = df.index[df["q"] < q].values; te = df.index[df["q"] == q].values
        if q not in CS:
            CS[q] = S.choice_sets(df.loc[tr].reset_index(drop=True), stages=1)
        Xtr = XF.loc[tr, fs].values; mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = S.fit_cl((Xtr - mu_) / sd_, CS[q], l2=1.0)
        p[te] = S.race_softmax(((XF.loc[te, fs].values - mu_) / sd_) @ w, df.loc[te, "rid"].values)
    m = (df["date"] >= lo) & (df["date"] <= hi) & (df["place"] == 1)
    return -np.log(p[m.values]).mean(), p
VAL, TEST = ("20240101", "20241231"), ("20250101", "20261231")
cur = list(MK); best, _ = rolling(cur, VAL)
print("出発点（MK 11項目）検証2024: %.4f" % best, flush=True)
while True:
    trials = sorted(((rolling(cur + [c], VAL)[0], c) for c in CAND if c not in cur))
    v, c = trials[0]
    if best - v < 0.0004:
        print("  打ち切り（最良の候補 %s で %.4f）" % (c, v)); break
    cur.append(c); best = v
    print("  + %-12s 検証2024: %.4f" % (c, v), flush=True)
mkt = df.loc[(df["date"] >= VAL[0]) & (df["date"] <= VAL[1]) & (df["place"] == 1), "pm"]
print("市場 検証2024: %.4f" % -np.log(mkt).mean())
T = (df["date"] >= TEST[0]).values
rows = [S.score(df[T], df.loc[T, "pm"].values, "market")]
for lab, fs in (("MK 11項目（通し再推定）", MK), ("選んだ組 %d項目" % len(cur), cur)):
    _, p = rolling(fs, TEST)
    rows.append(S.score(df[T], p[T], lab))
    if lab.startswith("選"):
        np.save("p_sel.npy", p)
print(pd.DataFrame(rows).set_index("model")[["races", "logloss", "hit", "top3", "ev1.0_n", "ev1.0_roi", "ev1.1_n", "ev1.1_roi", "ev1.2_n", "ev1.2_roi"]].round(4).to_string())
json.dump(cur, open("selected.json", "w"))
