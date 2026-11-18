import sys, numpy as np, pandas as pd
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
MKSET = ["log_pm", "mu"] + ["k_" + k for k in S.MKF]
sets = {"mk_walk(第4版と同じ11項目)": (MKSET, 1, 1.0), "mk_walk 3着まで": (MKSET, 3, 1.0),
        "cl_mkt l2=300": ([c for c in XF.columns], 3, 300.0), "cl_mkt l2=3000": ([c for c in XF.columns], 3, 3000.0),
        "cl_sim l2=300": ([c for c in XF.columns if c != "log_pm"], 3, 300.0)}
T = (df["date"] >= "20250101").values
test_q = sorted(df.loc[T, "q"].unique())
res = []
for lab, (fs, st, l2) in sets.items():
    p = np.full(len(df), np.nan)
    for q in test_q:
        tr = df.index[df["q"] < q].values; te = df.index[df["q"] == q].values
        Xtr = XF.loc[tr, fs].values; mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = S.fit_cl((Xtr - mu_) / sd_, S.choice_sets(df.loc[tr].reset_index(drop=True), stages=st), l2=l2)
        p[te] = S.race_softmax(((XF.loc[te, fs].values - mu_) / sd_) @ w, df.loc[te, "rid"].values)
    res.append(S.score(df[T], p[T], lab))
    if lab.startswith("mk_walk(") :
        print("  最後の係数:", dict(zip(fs, np.round(w / sd_, 4))))
dt_ = df[T]
res.insert(0, S.score(dt_, dt_["pm"].values, "market"))
res.insert(1, S.score(dt_, dt_["p_mk"].values, "mk_v4（係数は同期間で推定）"))
print(pd.DataFrame(res).set_index("model")[["logloss", "hit", "top3", "ev1.0_n", "ev1.0_roi", "ev1.1_n", "ev1.1_roi"]].round(4).to_string())
