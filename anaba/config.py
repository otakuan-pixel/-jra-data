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

# 「大穴」の定義。ユーザーの指定どおり10番人気以下を基本にする
LONGSHOT_POP = 10
# 出走30分前の時点で候補に入れる人気（締切までに人気が動くので少し広めに取る）
CANDIDATE_POP = 8

# 「奇跡の血量」= 3×4（または4×3）のクロス = 1/8 + 1/16 = 18.75%
MIRACLE_PCT = 18.75

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
