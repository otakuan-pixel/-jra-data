"""複勝の払戻金（実測 or 推定）

成績データには単勝オッズしか無いので、複勝の回収率を測るために「3着内に来た馬の複勝払戻」を作る。
  1. 実測：netkeiba のレースページ（data/netkeiba/races.jsonl）と、jra2 の成績PDFテキスト（阪神・中山）
  2. 推定：実測で学習したモデル（その馬の単勝オッズ・他の2頭の単勝オッズ・頭数 → 複勝払戻）
     2025年以降で検証：10番人気以下の平均 実測1,468円 / 推定1,442円（1レースごとの誤差は±19%程度）
"""
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd

from .config import JRA2_REPO, MODEL_DIR
from .data import KEY

MODEL_PATH = MODEL_DIR / "place_payout.pkl"
_DIG = {9872 + i: str(i) for i in range(10)}
_Z = str.maketrans("０１２３４５６７８９，", "0123456789,")
_TR = {"sapporo": "01", "hakodate": "02", "fukushima": "03", "niigata": "04", "tokyo": "05",
       "nakayama": "06", "chukyo": "07", "kyoto": "08", "hanshin": "09", "kokura": "10"}


def _text_payouts(repo=JRA2_REPO):
    """成績PDFのテキスト → [(year, jo, kai, nichi, page, 単勝払戻, [複勝払戻（着順）])]"""
    out = []
    for f in sorted(glob.glob(str(repo / "data" / "text" / "*" / "*.txt"))):
        m = re.match(r"(\d{4})-(\d+)([a-z]+)(\d+)\.txt", os.path.basename(f))
        if not m or m.group(3) not in _TR:
            continue
        y, kai, tr, nichi = m.groups()
        s = open(f, encoding="utf-8").read()
        s = re.sub(r"\(cid:(\d+)\)", lambda x: _DIG.get(int(x.group(1)), "{%s}" % x.group(1)), s).translate(_Z)
        pages = re.split(r"===== PAGE (\d+) =====", s)
        for i in range(1, len(pages), 2):
            t = pages[i + 1].replace("\n", " ")
            wins = [(x.start(), int(x.group(1).replace(",", ""))) for x in re.finditer(r"単 勝 \{\d+\} ([\d,]+)円", t)]
            for x in re.finditer(r"複 勝 ((?:\{\d+\} [\d,]+円 ?){2,6})", t):
                w = [v for p, v in wins if p < x.start()]
                pays = [int(v.replace(",", "")) for v in re.findall(r"\{\d+\} ([\d,]+)円", x.group(1))]
                out.append((y, _TR[tr], int(kai), int(nichi), int(pages[i]), w[-1] if w else None, pays))
    return out


def actual_payouts(races, runs, repo=JRA2_REPO):
    """実測の複勝払戻 → DataFrame[KEY + num, pay]"""
    rows = []
    # 1) PDFテキスト：ページ番号と左右・単勝払戻でレースを特定
    tp = pd.DataFrame(_text_payouts(repo), columns=["year", "jo", "kai", "nichi", "page", "win", "pays"])
    if len(tp):
        rc = races.copy()
        rc["page"] = rc["src"].str.extract(r"#p(\d+)")[0].astype(float)
        rc["year"] = rc["ymd"].str[:4]
        cand = tp.merge(rc[KEY + ["year", "page"]], on=["year", "jo", "kai", "nichi", "page"])
        w = runs[runs["place"] == 1][KEY + ["odds"]].drop_duplicates(KEY)
        cand = cand.merge(w, on=KEY)
        cand = cand[(cand["odds"] * 100 - cand["win"]).abs() < 1].drop_duplicates(KEY, keep=False)
        top = runs[(runs["place"] <= 3) & (runs["status"] == "")].sort_values(KEY + ["place"])
        tg = {k: g for k, g in top.groupby(KEY)}
        for _, r in cand.iterrows():
            g = tg.get(tuple(r[k] for k in KEY))
            if g is None or len(g) != len(r["pays"]):
                continue
            for num, pay in zip(g["num"], r["pays"]):
                rows.append(tuple(r[k] for k in KEY) + (num, pay))
    # 2) netkeiba のレースページ（実測そのもの）
    from .backfill import load_nk_races
    nk = load_nk_races()
    if nk:
        rc = races.copy()
        rc["race_id"] = rc["ymd"].str[:4] + rc["jo"] + rc["kai"].map("{:02d}".format) + \
            rc["nichi"].map("{:02d}".format) + rc["rno"].map("{:02d}".format)
        rmap = {r.race_id: tuple(r[k] for k in KEY) for _, r in rc.iterrows() if r.race_id in nk}
        for rid, r in nk.items():
            k = rmap.get(rid)
            if k is None:
                continue
            for combo, amt in r["pays"].get("複勝", []):
                if combo.isdigit():
                    rows.append(k + (int(combo), amt))
    df = pd.DataFrame(rows, columns=KEY + ["num", "pay"]).drop_duplicates(KEY + ["num"], keep="last")
    return df


def _features(odds, others_odds, heads):
    o = np.asarray(others_odds, float)
    return [np.log(odds), np.log((1.0 / o).sum()), np.log(o.min()), heads]


def placer_table(runs):
    """3着内の馬ごとに 推定用の特徴（自分と他の2頭の単勝オッズ・頭数）"""
    top = runs[(runs["place"] <= 3) & (runs["status"] == "") & runs["odds"].notna()].copy()
    top = top.sort_values(KEY + ["place"])
    heads = runs[runs["status"].isin(["", "中止", "降着"])].groupby(KEY).size().rename("nh")
    top = top.join(heads, on=KEY)
    feats, idx = [], []
    for _, g in top.groupby(KEY, sort=False):
        od = g["odds"].values
        if len(od) < 2:
            continue
        for i, (ix, o) in enumerate(zip(g.index, od)):
            feats.append(_features(o, np.delete(od, i), g["nh"].iloc[0]))
            idx.append(ix)
    X = pd.DataFrame(feats, index=idx, columns=["lo", "lsi", "lmin", "heads"])
    return top.loc[idx, KEY + ["num", "place", "odds", "pop"]].join(X)


def fit(races, runs, log=print):
    from sklearn.ensemble import HistGradientBoostingRegressor
    act = actual_payouts(races, runs)
    pt = placer_table(runs).merge(act, on=KEY + ["num"])
    pt = pt[pt["heads"] >= 8]
    m = HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=40, random_state=0)
    y = (pt["pay"] / 100.0 - 1.0).clip(lower=0.05)
    m.fit(pt[["lo", "lsi", "lmin", "heads"]].values, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(m, f)
    log("複勝払戻モデル：実測 %d 件で学習" % len(pt))
    return m


def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def place_payouts(races, runs, model=None):
    """3着内の全馬の複勝払戻（実測があれば実測、無ければ推定）→ DataFrame[KEY + num, place_pay, pay_src]"""
    model = model or load_model()
    pt = placer_table(runs)
    pt["est"] = 100.0 * (1.0 + model.predict(pt[["lo", "lsi", "lmin", "heads"]].values))
    pt["est"] = (np.floor(pt["est"] / 10.0) * 10.0).clip(lower=100.0)
    act = actual_payouts(races, runs)
    pt = pt.merge(act, on=KEY + ["num"], how="left")
    pt["place_pay"] = pt["pay"].fillna(pt["est"])
    pt["pay_src"] = np.where(pt["pay"].notna(), "実測", "推定")
    # 7頭以下は2着まで
    pt.loc[(pt["heads"] <= 7) & (pt["place"] == 3), "place_pay"] = 0.0
    return pt[KEY + ["num", "place_pay", "pay_src"]]
