"""過去レースの血統・払戻をnetkeibaから少しずつ集める（症例対照の血統検証用）

  python -m anaba backfill --since 20230101 --controls 2 --max-requests 3000

- 「症例」= 10番人気以下で3着以内に来た馬、「対照」= 同じレースの10番人気以下で4着以下の馬（無作為に N 頭）
- 同じレースの中で比べるので、コース・クラス・頭数・馬場の違いが打ち消される
- 新しいレースから順に取る。取ったものは data/ に追記するので、途中で止めても続きから再開できる
"""
import json
import random
import time
from pathlib import Path

from .config import DATA, LONGSHOT_POP
from .data import KEY, load_results
from .pedigree import Fetcher, PedStore, RACE_URL, fetch_ped, netkeiba_race_id, parse_race_page

NK_DIR = DATA / "netkeiba"
CC_PATH = DATA / "pedigree" / "casecontrol.csv"


def _nk_path(race_id, root=NK_DIR):
    return Path(root) / ("races-%s.jsonl" % race_id[:4])


def load_nk_races(root=NK_DIR):
    """netkeiba のレースページの要約（年ごとのファイル）→ {race_id: {...}}"""
    out = {}
    root = Path(root)
    files = sorted(root.glob("races-*.jsonl"))
    legacy = root / "races.jsonl"
    for f in files + ([legacy] if legacy.exists() else []):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["race_id"]] = r
    # 旧形式（1ファイル）は年ごとに書き直す
    if legacy.exists():
        have = {}
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    have[json.loads(line)["race_id"]] = 1
        for rid, r in out.items():
            if rid not in have:
                with _nk_path(rid, root).open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        legacy.unlink()
    return out


def fetch_race(fetcher, cache, race_id, root=NK_DIR):
    if race_id in cache:
        return cache[race_id]
    b = fetcher.get(RACE_URL % race_id)
    if not b:
        return None
    horses, pays = parse_race_page(b)
    if not horses:
        return None
    keep = ("単勝", "複勝", "ワイド", "馬連", "三連複")
    r = {"race_id": race_id, "horses": horses, "pays": {k: v for k, v in pays.items() if k in keep}}
    if any(h["place"] for h in horses):     # 結果が出ているものだけ保存
        Path(root).mkdir(parents=True, exist_ok=True)
        with _nk_path(race_id, root).open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    cache[race_id] = r
    return r


def run(since="20230101", until="99999999", controls=2, max_requests=3000, interval=1.1, log=print):
    races, runs = load_results()
    flat = runs[(runs.surf != "障") & (runs.status == "") & (runs.ymd >= since) & (runs.ymd <= until)]
    ls = flat[flat["pop"] >= LONGSHOT_POP]
    hits = ls[ls.place <= 3]
    target = hits[KEY].drop_duplicates().sort_values("ymd", ascending=False)
    log("対象：10番人気以下が3着内に来たレース %d（%s〜%s）" % (len(target), since, until))
    store = PedStore()
    cache = load_nk_races()
    fetcher = Fetcher(interval=interval)
    CC_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CC_PATH.exists():
        for line in CC_PATH.read_text(encoding="utf-8").splitlines()[1:]:
            done.add(line.split(",")[0])
    new_file = not CC_PATH.exists()
    f = CC_PATH.open("a", encoding="utf-8")
    if new_file:
        f.write("race_id,ymd,jo,rno,horse,horse_id,role,pop,place,odds\n")
    t0 = time.time()
    n_races = n_ped = 0
    for _, k in target.iterrows():
        if fetcher.n >= max_requests:
            break
        rid = netkeiba_race_id(k.ymd, k.jo, k.kai, k.nichi, k.rno)
        if rid in done:
            continue
        r = fetch_race(fetcher, cache, rid)
        if r is None:
            continue
        ids = {h["horse"]: h["horse_id"] for h in r["horses"]}
        g = ls[(ls.ymd == k.ymd) & (ls.jo == k.jo) & (ls.rno == k.rno)]
        case = g[g.place <= 3]
        ctrl = g[g.place > 3]
        rnd = random.Random(int(rid))
        pick = ctrl.sample(n=min(controls, len(ctrl)), random_state=rnd.randint(0, 2 ** 31)) if len(ctrl) else ctrl
        rows = [(h, "case") for _, h in case.iterrows()] + [(h, "control") for _, h in pick.iterrows()]
        ok = True
        for h, role in rows:
            hid = ids.get(h.horse, "")
            if hid and hid not in store:
                if fetcher.n >= max_requests:
                    ok = False
                    break
                if fetch_ped(store, fetcher, hid):
                    n_ped += 1
        if not ok:
            break
        for h, role in rows:
            f.write("%s,%s,%s,%d,%s,%s,%s,%d,%d,%.1f\n" % (rid, k.ymd, k.jo, k.rno, h.horse, ids.get(h.horse, ""),
                                                          role, h["pop"], h.place, h.odds))
        f.flush()
        done.add(rid)
        n_races += 1
        if n_races % 50 == 0:
            log("  %d レース / 血統 %d 頭 / リクエスト %d / %.0f 秒" % (n_races, n_ped, fetcher.n, time.time() - t0))
    f.close()
    log("完了：%d レース / 血統 %d 頭 / リクエスト %d" % (n_races, n_ped, fetcher.n))
    return n_races, n_ped


def fetch_peds_for(horse_ids, max_requests=2000, interval=1.1, log=print):
    """指定した馬IDの血統をまとめて取る（出走予定馬など）"""
    store = PedStore()
    fetcher = Fetcher(interval=interval)
    todo = [h for h in dict.fromkeys(horse_ids) if h and h not in store]
    got = 0
    for hid in todo:
        if fetcher.n >= max_requests:
            break
        if fetch_ped(store, fetcher, hid):
            got += 1
    log("血統：対象 %d 頭（未取得 %d）→ 新たに %d 頭" % (len(horse_ids), len(todo), got))
    return got
