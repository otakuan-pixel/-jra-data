"""馬の索引（馬名 → JRA/netkeibaの馬ID・父・母・母の父）

jra2 リポジトリの git 履歴に残っている全ての出馬表から作る。成績データ（PDF由来）には馬IDも血統も
無いので、馬名で突き合わせる（JRAの現役馬で同名はまず無い）。
"""
import csv
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import DATA, JRA2_REPO

HORSES_CSV = DATA / "horses.csv"
FIELDS = ["horse", "horse_id", "sire", "dam", "bms", "sex", "last_seen"]


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo)] + list(a), capture_output=True, check=True).stdout


def _entry_blobs(repo):
    """git履歴の全コミットに出てくる data/entries/<日付>/<場>_<R>.html の blob（重複なし）"""
    seen = {}
    commits = _git(repo, "log", "--format=%H", "--", "data/entries").decode().split()
    for c in commits:
        for line in _git(repo, "ls-tree", "-r", c, "data/entries").decode().splitlines():
            meta, path = line.split("\t", 1)
            sha = meta.split()[2]
            p = Path(path)
            if p.suffix == ".html" and p.parent.name.isdigit() and sha not in seen:
                seen[sha] = (p.parent.name, p.stem)
    # 作業ツリーにあってまだコミットされていないもの
    for p in (Path(repo) / "data" / "entries").glob("*/*.html"):
        if p.parent.name.isdigit():
            seen.setdefault("file:" + str(p), (p.parent.name, p.stem))
    return seen


def _parse_blob(args):
    repo, sha, ymd, stem = args
    from .entries import parse_entry
    try:
        b = Path(sha[5:]).read_bytes() if sha.startswith("file:") else _git(repo, "cat-file", "blob", sha)
        jo, r = stem.split("_")
        race = parse_entry(b, ymd, jo, int(r))
    except Exception:
        return []
    if not race:
        return []
    return [{"horse": h["horse"], "horse_id": h["horse_id"], "sire": h["sire"], "dam": h["dam"],
             "bms": h["bms"], "sex": h["sex"], "last_seen": ymd} for h in race["horses"] if h["horse"]]


def build(repo=JRA2_REPO, out=HORSES_CSV, workers=4):
    blobs = _entry_blobs(repo)
    jobs = [(str(repo), sha, ymd, stem) for sha, (ymd, stem) in blobs.items()]
    rows = {}
    with ProcessPoolExecutor(workers) as ex:
        for lst in ex.map(_parse_blob, jobs, chunksize=8):
            for h in lst:
                k = h["horse"]
                if k not in rows or h["last_seen"] >= rows[k]["last_seen"]:
                    # 古い出馬表で欠けていた欄は残す
                    old = rows.get(k, {})
                    rows[k] = {f: (h.get(f) or old.get(f, "")) for f in FIELDS}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 既存の索引（以前に作ったもの）とマージ：出馬表は3週間で消えるので、過去に作った行は残す
    if out.exists():
        for r in load(out).values():
            if r["horse"] not in rows:
                rows[r["horse"]] = r
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    return len(blobs), len(rows)


def load(path=HORSES_CSV):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["horse"]: r for r in csv.DictReader(f)}


def add_from_cards(cards, ymd, path=HORSES_CSV):
    """当日の出馬表から 馬名→馬ID・父・母・母父 を索引に足す（jra2 の履歴を読まなくても索引が育つ）"""
    rows = load(path)
    n0 = len(rows)
    for card in cards:
        for h in card["horses"]:
            if h["horse"]:
                rows[h["horse"]] = {"horse": h["horse"], "horse_id": h["horse_id"], "sire": h["sire"], "dam": h["dam"],
                                    "bms": h["bms"], "sex": h["sex"], "last_seen": ymd}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({k2: rows[k].get(k2, "") for k2 in FIELDS})
    return len(rows) - n0
