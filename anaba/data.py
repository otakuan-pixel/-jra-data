"""成績データ（otakuan-pixel/jra2 の data/results）を読み込んで1つの表にする"""
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE, JRA2_REPO

KEY = ["ymd", "jo", "kai", "nichi", "rno"]


def _read(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("成績データが見つかりません: %s （JRA2_REPO を確認）" % pattern)
    return pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)


def load_results(repo=JRA2_REPO, use_cache=True):
    """(races, runs) を返す。runs は1頭1行で、レースの情報（芝ダ・距離・馬場など）も付けてある"""
    repo = Path(repo)
    idx = repo / "data" / "results" / "index.csv"
    stamp = idx.stat().st_mtime if idx.exists() else 0
    cpath = CACHE / "results.pkl"
    if use_cache and cpath.exists():
        with cpath.open("rb") as f:
            c = pickle.load(f)
        if c.get("stamp") == stamp and c.get("repo") == str(repo):
            return c["races"], c["runs"]
    races = _read(str(repo / "data" / "results" / "*" / "*_races.csv.gz"))
    runs = _read(str(repo / "data" / "results" / "*" / "*_runs.csv.gz"))
    for c in ("kai", "nichi", "rno"):
        races[c] = races[c].astype(int)
        runs[c] = runs[c].astype(int)
    for c in ("dist", "heads"):
        races[c] = pd.to_numeric(races[c], errors="coerce")
    for c in ("place", "waku", "num", "age", "wt", "hw", "hwd", "odds", "pop", "c1", "c2", "c3", "c4"):
        runs[c] = pd.to_numeric(runs[c], errors="coerce")
    runs["status"] = runs["status"].fillna("")
    runs["blinker"] = runs["blinker"].fillna("")
    runs["tsec"] = runs["time"].map(_sec)
    races = races.drop_duplicates(KEY)
    rc = ["surf", "dist", "inout", "turn", "weather", "cond", "heads", "name", "grade", "cls_txt", "last3", "first3"]
    runs = runs.merge(races[KEY + rc], on=KEY, how="left")
    runs["cls"] = [_cls(g, c, n) for g, c, n in zip(runs["grade"].fillna(""), runs["cls_txt"].fillna(""), runs["name"].fillna(""))]
    runs = runs.sort_values(KEY + ["place"]).reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with cpath.open("wb") as f:
        pickle.dump({"stamp": stamp, "repo": str(repo), "races": races, "runs": runs}, f)
    return races, runs


def _sec(t):
    if not isinstance(t, str) or not t:
        return np.nan
    try:
        if ":" in t:
            m, s = t.split(":")
            return int(m) * 60 + float(s)
        return float(t)
    except ValueError:
        return np.nan


CLS_RANK = {"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5, "G3": 6, "G2": 7, "G1": 8}


def _cls(grade, cls_txt, name):
    g = grade + " " + cls_txt + " " + name
    for k, v in (("GⅠ", "G1"), ("G1", "G1"), ("GⅡ", "G2"), ("G2", "G2"), ("GⅢ", "G3"), ("G3", "G3"),
                 ("新馬", "新馬"), ("未勝利", "未勝利"), ("1勝", "1勝"), ("2勝", "2勝"), ("3勝", "3勝"),
                 ("オープン", "OP"), ("L", "OP")):
        if k in g:
            return v
    return "OP"


# 成績PDFの「産地・生産牧場」欄は「浦河富田牧場」のように産地と牧場名がつながっている。出馬表は牧場名だけなので、
# 先頭の産地を外して出馬表の書き方にそろえる
SANCHI = sorted("""新ひだか 日高 浦河 新冠 平取 様似 えりも 安平 千歳 むかわ 鵡川 白老 登別 苫小牧 厚真 伊達 洞爺湖 豊浦 恵庭
北広島 札幌 当別 新十津川 浦臼 由仁 長沼 栗山 岩見沢 帯広 音更 幕別 新得 清水 鹿追 芽室 中札内 大樹 広尾 浜中 釧路 標茶
弟子屈 鶴居 別海 中標津 七飯 八雲 函館 北斗 石狩 小樽 余市 早来 追分 静内 三石 門別 美浦 青森 岩手 宮城 秋田 山形 福島
茨城 栃木 群馬 千葉 埼玉 熊本 宮崎 鹿児島 南ア 英 米 愛 仏 豪 独 加 伊 新 亜 智 伯""".split(), key=len, reverse=True)
_SANCHI_RE = None


def breeder_key(name):
    import re
    global _SANCHI_RE
    if not isinstance(name, str):
        return ""
    if _SANCHI_RE is None:
        _SANCHI_RE = re.compile("^(?:%s)(?=.{2,})" % "|".join(map(re.escape, SANCHI)))
    n = _SANCHI_RE.sub("", name.replace(" ", "").replace("\u3000", ""), count=1)
    return re.sub(r"^(?:\((?:有|株|同|善|合|資|一社|社)\)|有限会社|株式会社|合同会社)", "", n)


def dist_band(d):
    """距離帯（短距離/マイル/中距離/中長距離/長距離）"""
    if pd.isna(d):
        return ""
    d = int(d)
    if d <= 1300:
        return "S"
    if d <= 1600:
        return "M"
    if d <= 2000:
        return "I"
    if d <= 2400:
        return "L"
    return "E"


def going_group(cond):
    return "道悪" if cond in ("重", "不良") else ("良" if cond in ("良", "稍重") else "")
