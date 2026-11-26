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


# 印の基準：総当たり検証（reports/research/benchmark.md）で決めた
#   18通りの予測方法 × 5通りの選び方 × 5通りの対象 × 閾値 を、2023〜24年だけで選んで2025〜26年で確かめた。
#   回収率100%前後に届いたのは「中穴の単勝・期待値の最上位（約5%）」だけ：
#     単勝期待値1.45以上 → 年ごと 101・108・107・121%（前半104%→後半113%、後半の95%区間77〜151%）
#   その下の帯（1.1〜1.45）は全部買うのと変わらない（約80%）ので印を付けない。
#   大穴はどの方法・閾値でも複勝72〜80%（全部買うと60%）。買いの印は付けず「△参考」として出す。
# 区分 → [(印, 期待値の列, 閾値, 券種, 買いか)]
MARK_RULES = {
    "中穴": [("★", "ev_win", 1.45, "単勝", True)],
    "大穴": [("△", "ev_place", 1.0, "複勝", False)],
}
# 賭け金は均等（ケリー基準は検証で均等より大幅に悪かった：4年で資金0.36倍 vs 1.11倍）
STAKE_NOTE = "1点の金額は均等に（目安：資金の0.5%以下）。過去の検証では、回収率100%前後でも資金が一時5割近く減る時期があった"


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
