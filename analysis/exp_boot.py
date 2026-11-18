import numpy as np, pandas as pd
df = pd.read_csv("feats.csv.gz", dtype={"date": str, "jo": str})
df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
df["rid"] = df["date"] + ":" + df["jo"] + ":" + df["rno"].astype(str)
inv = 1.0 / df["odds"]; df["pm"] = inv / inv.groupby(df["rid"]).transform("sum")
nw = df.assign(w=(df["place"] == 1)).groupby("rid")["w"].transform("sum")
df = df[nw == 1].reset_index(drop=True)
df["p_sel"] = np.load("p_sel.npy")
d = df[df["date"] >= "20250101"].copy()
w = d[d["place"] == 1].set_index("rid")
diff = (np.log(w["p_sel"]) - np.log(w["pm"])).values   # ＋＝選んだ組のほうが良い
diff_v4 = (np.log(w["p_sim"].clip(1e-4)) - np.log(w["pm"])).values
rng = np.random.default_rng(0)
B = 4000
idx = rng.integers(0, len(diff), (B, len(diff)))
bs = diff[idx].mean(1)
print("対数損失の改善（市場−選んだ組）: %.4f  95%%幅 %.4f〜%.4f  改善している確率 %.1f%%" % (diff.mean(), *np.percentile(bs, [2.5, 97.5]), 100 * (bs > 0).mean()))
d["ev"] = d["p_sel"] * d["odds"]
rids = d["rid"].unique(); rpos = {r: i for i, r in enumerate(rids)}
for th, lim in ((1.0, 50), (1.1, 50), (0.9, 50)):
    b = d[(d["ev"] >= th) & (d["odds"] < lim)]
    stake = np.zeros(len(rids)); ret = np.zeros(len(rids))
    np.add.at(stake, b["rid"].map(rpos).values, 1)
    np.add.at(ret, b["rid"].map(rpos).values, (b["odds"] * (b["place"] == 1)).values)
    ix = rng.integers(0, len(rids), (B, len(rids)))
    roi = ret[ix].sum(1) / np.maximum(stake[ix].sum(1), 1)
    print("期待値%.1f以上・%d倍未満: %d点 回収 %.0f%%  95%%幅 %.0f%%〜%.0f%%  100%%を超える確率 %.0f%%" % (
        th, lim, len(b), 100 * ret.sum() / max(stake.sum(), 1), *(100 * np.percentile(roi, [2.5, 97.5])), 100 * (roi > 1).mean()))
# 全馬単勝の基準
print("全馬を買った場合: 回収 %.0f%%" % (100 * (d["odds"] * (d["place"] == 1)).sum() / len(d)))
