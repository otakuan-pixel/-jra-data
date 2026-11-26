"""共通の定数とパス"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 成績データ（otakuan-pixel/jra2 のクローン）。環境変数 JRA2_REPO で上書きできる
JRA2_REPO = Path(os.environ.get("JRA2_REPO", ROOT.parent / "otakuan-pixel" / "jra2"))
DATA = ROOT / "data"            # このリポジトリで作るデータ（血統キャッシュ・学習済みモデル・集計表）
PED_DIR = DATA / "pedigree"     # 5代血統のキャッシュ（1頭1行の jsonl）
MODEL_DIR = DATA / "model"
REPORT_DIR = ROOT / "reports"
CACHE = Path(os.environ.get("ANABA_CACHE", ROOT / ".cache"))

JO_NAME = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
           "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
NAME_JO = {v: k for k, v in JO_NAME.items()}

# 区分（ユーザーの定義）
#   中穴 = 6〜10番人気・単勝10〜30倍
#   大穴 = 10番人気以下・単勝30倍以上
# 10番人気は両方に入り、人気とオッズがずれる馬もいる（多頭数の11番人気が25倍など）ので、
# 「6番人気以下」をオッズで分ける（払戻はオッズで決まるため）。
MIN_POP = 6
CHUANA_ODDS = 10.0
OOANA_ODDS = 30.0
TIERS = ("中穴", "大穴")
# 特徴量（騎手・調教師などの「人気薄での好走率」）に使う人気の線
LONGSHOT_POP = 10
# 出走30分前の時点で候補に入れる人気（＝区分の下限）
CANDIDATE_POP = MIN_POP


# 印の基準（区分ごと）：ウォークフォワード検証（2023〜26年、その年より前だけで学習）で毎年安定していた層
#   中穴は市場がもともと効率的で、モデルの上乗せは単勝で効く（単勝期待値1.1以上：単勝回収率 年ごと86〜106%）
#   大穴は買われすぎが大きく、上乗せは複勝で大きい（複勝期待値0.9以上：複勝回収率 年ごと79〜84%）
MARK_RULES = {
    "中穴": [("★", "ev_win", 1.1, "単勝"), ("☆", "ev_place", 0.9, "複勝")],
    "大穴": [("★", "ev_place", 0.9, "複勝"), ("☆", "ev_place", 0.8, "複勝")],
}


def tier_of(odds, pop):
    """1頭の区分：'中穴' / '大穴' / ''（どちらでもない）"""
    try:
        odds, pop = float(odds), float(pop)
    except (TypeError, ValueError):
        return ""
    if pop != pop or odds != odds or pop < MIN_POP:
        return ""
    if odds >= OOANA_ODDS:
        return "大穴"
    if odds >= CHUANA_ODDS:
        return "中穴"
    return ""


def tier_series(df):
    import numpy as np
    import pandas as pd
    ok = df["pop"].fillna(0) >= MIN_POP
    t = np.where(ok & (df["odds"] >= OOANA_ODDS), "大穴", np.where(ok & (df["odds"] >= CHUANA_ODDS), "中穴", ""))
    return pd.Series(t, index=df.index)

# 「奇跡の血量」= 3×4（または4×3）のクロス = 1/8 + 1/16 = 18.75%
MIRACLE_PCT = 18.75

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
