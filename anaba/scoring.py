"""レース（出馬表）を成績データにつなげて特徴量を作り、学習済みモデルで採点する"""
import json

import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .data import KEY, load_results
from .features import prepare
from .model import Bundle
from .research import factors


def score_cards(card_runs_list, hist=None, bundle=None, horses=None, cutoff=None, log=print):
    """card_runs_list: live.card_to_runs の DataFrame のリスト → 採点済みの行（そのレースの馬だけ）

    cutoff（YYYYMMDD）を渡すと、その日以降の成績を履歴から外し、モデルもその日より前のデータだけで学習し直す
    （過去レースの再現テスト用。学習済みの本番モデルは結果を知っているので使わない）
    """
    if hist is None:
        _, hist = load_results()
    if cutoff:
        hist = hist[hist["ymd"] < cutoff]
    cards = pd.concat(card_runs_list, ignore_index=True)
    keys = cards[KEY].drop_duplicates()
    # 同じレースの成績が履歴にあれば外す（再現テストのとき）
    h = hist.merge(keys.assign(_x=1), on=KEY, how="left")
    h = h[h["_x"].isna()].drop(columns="_x")
    extra_cols = ["place_lo", "place_hi", "horse_id", "sire", "dam", "bms"]
    base_cols = [c for c in h.columns]
    all_runs = pd.concat([h, cards[[c for c in base_cols if c in cards.columns]]], ignore_index=True)
    if horses is not None and len(horses):
        hm = pd.DataFrame(horses.values()) if isinstance(horses, dict) else horses
        # 出馬表に載っている父・母の父を優先
        add = cards[["horse", "sire", "bms"]]
        hm = pd.concat([add, hm[["horse", "sire", "bms"]]]).drop_duplicates("horse")
    else:
        hm = cards[["horse", "sire", "bms"]]
    full = prepare(all_runs, hm)
    if cutoff:
        from . import model as M, payout
        races_, _ = load_results()
        pp = payout.place_payouts(races_, h)
        tr = full[full["ymd"] < cutoff].merge(pp, on=KEY + ["num"], how="left")
        tr["place_pay"] = tr["place_pay"].fillna(0.0)
        bundle = M.fit_all(tr, log=log)
    bundle = bundle or Bundle.load()
    df = full.merge(keys.assign(_t=1), on=KEY, how="left")
    df = df[df["_t"] == 1].drop(columns="_t")
    df = df.merge(cards[KEY + ["num"] + extra_cols].drop(columns=["sire", "bms"]), on=KEY + ["num"], how="left")
    sc = bundle.score(df)
    df = pd.concat([df.reset_index(drop=True), sc.reset_index(drop=True)], axis=1)
    # 発走前の複勝オッズ（幅）から見た期待値
    for c in ("place_lo", "place_hi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    mid = np.sqrt(df["place_lo"] * df["place_hi"])
    df["ev_place_lo"] = df["p_top3"] * df["place_lo"]
    df["ev_place_mid"] = df["p_top3"] * mid
    return df


def load_json(name, default=None):
    p = MODEL_DIR / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return default


def matched_factors(row_df):
    """1頭分の行 → 当てはまる条件の名前の list"""
    fs = factors(row_df)
    return [k for k, v in fs.items() if bool(v.iloc[0])]
