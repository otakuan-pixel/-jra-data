"""5代血統表の取得・解析と、インブリード（クロス）／「奇跡の血量」の判定

- 血統表は netkeiba の db.netkeiba.com/horse/ped/<馬ID>/（馬IDはJRA出馬表のリンクにある10桁と同じ）
- 血統表の <td rowspan> が 16/8/4/2/1 → 1〜5代。文書順に並んだ同じ世代のセルを上から数えた順番を
  2進数にすると、父(F)・母(M)の経路になる（例：3代目の3番目 = 0b010 = 父→母→父 = FMF）
- 祖先は netkeiba の祖先ID（/horse/<ID>/）で突き合わせる（カナ表記と英字表記の揺れを避けるため）

クロスの表記は「3×4」のように、その祖先が出てくる世代を、父側→母側の順に並べる。
血量 = Σ (1/2)^世代。3×4 = 12.5% + 6.25% = 18.75% を「奇跡の血量」と呼ぶ。
"""
import json
import re
import time
from fractions import Fraction
from pathlib import Path

from .config import MIRACLE_PCT, PED_DIR, USER_AGENT

PED_URL = "https://db.netkeiba.com/horse/ped/%s/"
RACE_URL = "https://db.netkeiba.com/race/%s/"
GEN_OF_ROWSPAN = {16: 1, 8: 2, 4: 3, 2: 4, 1: 5}


# ───────────────────────── 解析 ─────────────────────────
def _decode(b):
    if isinstance(b, str):
        return b
    for enc in ("utf-8", "euc_jp", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", "replace")


def _clean_name(txt):
    """'Frankel (英) 2008 鹿毛 [ 血統 ][ 産駒 ] 1-k' → 'Frankel'"""
    t = re.split(r"\s*[\(（]|\s+\d{4}\s|\s*\[", txt.strip(), maxsplit=1)[0]
    return re.sub(r"\s+\d+-[a-z]$", "", t).strip()


def parse_ped(html):
    """netkeiba の5代血統ページ → [{'path': 'FMF', 'gen': 3, 'id': '...', 'name': '...'}...]（62件）"""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(_decode(html), "lxml")
    t = s.select_one("table.blood_table")
    if t is None:
        return []
    idx = {g: 0 for g in range(1, 6)}
    out = []
    for td in t.select("td"):
        rs = int(td.get("rowspan") or 1)
        g = GEN_OF_ROWSPAN.get(rs)
        if g is None:
            continue
        i = idx[g]
        idx[g] += 1
        path = format(i, "0%db" % g).replace("0", "F").replace("1", "M")
        a = td.find("a", href=re.compile(r"/horse/[0-9a-z]+/?$"))
        aid = ""
        if a is not None:
            m = re.search(r"/horse/([0-9a-z]+)/?$", a.get("href", ""))
            aid = m.group(1) if m else ""
        name = _clean_name(a.get_text(" ", strip=True) if a is not None else td.get_text(" ", strip=True))
        out.append({"path": path, "gen": g, "id": aid, "name": name})
    return out


def crosses(anc, max_gen=5):
    """5代以内に2回以上出てくる祖先 → [{'name', 'id', 'gens': [3, 4], 'label': '3×4', 'pct': 18.75, 'via': [...]}]

    下の世代のクロスに含まれるだけの祖先（例：サンデーサイレンスが3×4なら、その父Haloは4×5で必ず出る）は
    'derived': True にする（表示や判定では使わない）。
    """
    by = {}
    for a in anc:
        if a["gen"] > max_gen or not (a["id"] or a["name"]):
            continue
        key = a["id"] or a["name"]
        by.setdefault(key, []).append(a)
    dup = {k: v for k, v in by.items() if len(v) >= 2}
    dup_paths = {a["path"] for v in dup.values() for a in v}
    out = []
    for k, v in dup.items():
        # 父側の出現 → 母側の出現 の順（表記の慣例）
        v = sorted(v, key=lambda a: (a["path"][0] != "F", a["gen"], a["path"]))
        gens = [a["gen"] for a in v]
        pct = float(sum(Fraction(1, 2 ** g) for g in gens) * 100)
        derived = all(any(a["path"][:j] in dup_paths for j in range(1, len(a["path"]))) for a in v)
        out.append({"name": v[0]["name"], "id": v[0]["id"] if v[0]["id"] else "", "gens": gens,
                    "label": "×".join(str(g) for g in gens), "pct": round(pct, 4),
                    "sides": "".join(a["path"][0] for a in v), "derived": derived})
    out.sort(key=lambda c: (-c["pct"], c["name"]))
    return out


def summarize(anc):
    """血統表 → 判定用の要約（奇跡の血量の有無・主なクロス・最大血量・父系/母父など）"""
    cs = [c for c in crosses(anc) if not c["derived"]]
    by_path = {a["path"]: a for a in anc}
    miracle = [c for c in cs if sorted(c["gens"]) == [3, 4]]
    miracle_eq = [c for c in cs if abs(c["pct"] - MIRACLE_PCT) < 1e-6]
    both = [c for c in cs if "F" in c["sides"] and "M" in c["sides"]]
    total = float(sum(Fraction(1, 2 ** g) for c in cs for g in c["gens"]) * 100) if cs else 0.0
    return {
        "sire": by_path.get("F", {}).get("name", ""), "dam": by_path.get("M", {}).get("name", ""),
        "sire_sire": by_path.get("FF", {}).get("name", ""), "bms": by_path.get("MF", {}).get("name", ""),
        "bms2": by_path.get("MMF", {}).get("name", ""),   # 母の母の父
        "crosses": [{k: c[k] for k in ("name", "label", "pct", "sides")} for c in cs],
        "miracle": bool(miracle),                      # 3×4 / 4×3 そのもの
        "miracle_names": [c["name"] + " " + c["label"] for c in miracle],
        "miracle_eq": bool(miracle_eq),                # 4×4×4 など、合計がちょうど18.75%
        "max_pct": max((c["pct"] for c in cs), default=0.0),
        "n_cross": len(cs),
        "n_cross_both": len(both),                     # 父側と母側の両方にまたがるクロス（いわゆる「父母クロス」）
        "inbreed_total": round(total, 4),
        "close": any(min(c["gens"]) <= 2 for c in cs),  # 2代以内の近親（2×3など）
    }


# ───────────────────────── 取得（キャッシュ付き） ─────────────────────────
# 5代血統表の62マス（1代：F,M／2代：FF,FM,MF,MM／…）を決まった順に並べる
PATHS = [format(i, "0%db" % g).replace("0", "F").replace("1", "M") for g in range(1, 6) for i in range(2 ** g)]


def _shard(hid):
    return hid[:4] if hid[:4].isdigit() else "other"


class PedStore:
    """血統のキャッシュ（GitHubの1ファイル上限に当たらないよう分割して保存）

    data/pedigree/names.tsv          祖先ID<TAB>名前
    data/pedigree/peds-<生年>.jsonl   1頭1行 {"h": 馬ID, "a": "祖先ID,祖先ID,…（PATHSの順に62個）", "f": 取得日}
    祖先IDが無いマス（古い海外馬など）は "n:名前" を使う。
    """

    def __init__(self, root=PED_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.names = {}
        self.data = {}
        npath = self.root / "names.tsv"
        if npath.exists():
            for line in npath.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    k, v = line.split("\t", 1)
                    self.names[k] = v
        for f in sorted(self.root.glob("peds-*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.data[r["h"]] = r
        # 旧形式（1ファイル）が残っていれば読み込んで新形式に移す
        legacy = self.root / "peds.jsonl"
        if legacy.exists():
            for line in legacy.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r["horse_id"] not in self.data:
                        anc = [{"path": p_, "gen": len(p_), "id": i, "name": n} for p_, i, n in r["anc"]]
                        self.add(r["horse_id"], anc, fetched=r.get("fetched"))
            legacy.unlink()

    def __contains__(self, hid):
        return hid in self.data

    def __len__(self):
        return len(self.data)

    def anc(self, hid):
        r = self.data.get(hid)
        if not r:
            return None
        out = []
        for p_, k in zip(PATHS, r["a"].split(",")):
            if not k:
                continue
            if k.startswith("n:"):
                out.append({"path": p_, "gen": len(p_), "id": "", "name": k[2:]})
            else:
                out.append({"path": p_, "gen": len(p_), "id": k, "name": self.names.get(k, k)})
        return out

    def get(self, hid):
        a = self.anc(hid)
        return {"horse_id": hid, "anc": a} if a else None

    def summary(self, hid):
        a = self.anc(hid)
        return summarize(a) if a else None

    def add(self, hid, anc, fetched=None):
        by = {a["path"]: a for a in anc}
        keys, new_names = [], []
        for p_ in PATHS:
            a = by.get(p_)
            if not a:
                keys.append("")
            elif a["id"]:
                keys.append(a["id"])
                if a["id"] not in self.names:
                    self.names[a["id"]] = a["name"]
                    new_names.append((a["id"], a["name"]))
            else:
                keys.append("n:" + a["name"].replace(",", " "))
        r = {"h": hid, "a": ",".join(keys), "f": fetched or time.strftime("%Y-%m-%d")}
        with (self.root / ("peds-%s.jsonl" % _shard(hid))).open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        if new_names:
            with (self.root / "names.tsv").open("a", encoding="utf-8") as f:
                for k, v in new_names:
                    f.write("%s\t%s\n" % (k, v.replace("\t", " ").replace("\n", " ")))
        self.data[hid] = r
        return self.get(hid)


class Fetcher:
    """netkeiba への丁寧なアクセス（1秒以上の間隔・リトライ・ブロック検知）"""

    def __init__(self, interval=1.1):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"})
        self.interval = interval
        self.last = 0.0
        self.n = 0

    def get(self, url):
        for i in range(3):
            wait = self.interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.s.get(url, timeout=30)
                self.n += 1
                if r.status_code == 200:
                    return r.content
                if r.status_code in (400, 403, 429):
                    # アクセス過多と判断されたら長めに待って、それでもだめなら諦める
                    time.sleep(30 * (i + 1))
                    continue
                if r.status_code == 404:
                    return None
            except Exception:
                time.sleep(5 * (i + 1))
        return None


def fetch_ped(store, fetcher, hid):
    if hid in store:
        return store.get(hid)
    b = fetcher.get(PED_URL % hid)
    if not b:
        return None
    anc = parse_ped(b)
    if len(anc) < 30:     # 血統表が読めない（ページ構造の変更など）
        return None
    return store.add(hid, anc)


# ───────────────────────── netkeiba のレース結果ページ ─────────────────────────
def parse_race_page(html):
    """db.netkeiba.com/race/<ID>/ → (馬の一覧[{place,num,horse,horse_id}], 払戻{券種: [(組番, 金額)]})"""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(_decode(html), "lxml")
    t = s.select_one("table.race_table_01")
    horses = []
    if t is not None:
        for tr in t.select("tr")[1:]:
            tds = tr.find_all("td")
            a = tr.select_one('a[href*="/horse/"]')
            if len(tds) < 4 or a is None:
                continue
            m = re.search(r"/horse/(\w+)/?", a.get("href", ""))
            pl = tds[0].get_text(strip=True)
            horses.append({"place": int(pl) if pl.isdigit() else None,
                           "num": int(tds[2].get_text(strip=True)) if tds[2].get_text(strip=True).isdigit() else None,
                           "horse": a.get_text(strip=True), "horse_id": m.group(1) if m else ""})
    pays = {}
    for p in s.select("table.pay_table_01"):
        for tr in p.select("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if th is None or len(tds) < 2:
                continue
            combos = tds[0].get_text("\n", strip=True).split("\n")
            amts = [int(x.replace(",", "")) for x in tds[1].get_text("\n", strip=True).split("\n") if x.replace(",", "").isdigit()]
            pays[th.get_text(strip=True)] = list(zip(combos, amts))
    return horses, pays


def netkeiba_race_id(ymd, jo, kai, nichi, rno):
    return "%s%s%02d%02d%02d" % (str(ymd)[:4], str(jo).zfill(2), int(kai), int(nichi), int(rno))
