"""選んだ15項目で、全期間（〜最新）を使って最終の係数を出す（本番の sim --odds に組み込む用）"""
import sys, json, numpy as np, pandas as pd
sys.argv = ["x"]
import stage2 as S
df = pd.read_csv("feats.csv.gz", dtype={"date": str, "jo": str})
df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
df["rid"] = df["date"] + ":" + df["jo"] + ":" + df["rno"].astype(str)
inv = 1.0 / df["odds"]; df["pm"] = inv / inv.groupby(df["rid"]).transform("sum")
nw = df.assign(w=(df["place"] == 1)).groupby("rid")["w"].transform("sum")
df = df[nw == 1].reset_index(drop=True)
XF = S.build(df)
fs = json.load(open("selected.json"))
X = XF[fs].values; mu_, sd_ = X.mean(0), X.std(0) + 1e-9
w = S.fit_cl((X - mu_) / sd_, S.choice_sets(df, stages=1), l2=1.0)
coef = dict(zip(fs, (w / sd_).round(5).tolist()))
out = {"features": fs, "coef": coef, "trained_on": "%s〜%s" % (df["date"].min(), df["date"].max()), "races": int(df["rid"].nunique()),
       "note": "勝率 = レース内 softmax( Σ coef×特徴 )。log_pm は 1/単勝オッズを合計1にした値の対数。c_* は第4版 rate() の各補正の加算量、brd_vm は生産者の直近2年の log((勝ち+5)/(オッズから見た勝ちの見込み+5))"}
json.dump(out, open("coef_final.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(coef, ensure_ascii=False))
