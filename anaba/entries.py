"""JRA公式の出馬表HTML（Shift_JIS）を読む

1ページ = 1レース。馬ごとに 馬番・馬名・JRAの馬ID（= netkeibaの馬ID）・単勝オッズと人気（取得時点）・
父/母/母の父・騎手・斤量・調教師・馬体重（発表後のみ）・近4走 を取り出す。
"""
import re

from .config import JO_NAME


def _t(el):
    return el.get_text(" ", strip=True) if el else ""


def _num(s, cast=float):
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    if not m:
        return None
    try:
        return cast(m.group(0))
    except ValueError:
        return None


def decode(b):
    if isinstance(b, str):
        return b
    m = re.search(rb'charset=["\']?([\w-]+)', b[:3000], re.I)
    for enc in ([m.group(1).decode()] if m else []) + ["cp932", "utf-8", "euc_jp"]:
        try:
            return b.decode(enc)
        except (LookupError, UnicodeDecodeError):
            pass
    return b.decode("cp932", "replace")


def cls_of(text):
    t = text or ""
    for k, v in (("GⅠ", "G1"), ("GⅡ", "G2"), ("GⅢ", "G3"), ("JpnⅠ", "G1"), ("JpnⅡ", "G2"), ("JpnⅢ", "G3"),
                 ("リステッド", "OP"), ("オープン", "OP"), ("3勝", "3勝"), ("2勝", "2勝"), ("1勝", "1勝"),
                 ("1600万", "3勝"), ("1000万", "2勝"), ("500万", "1勝"), ("新馬", "新馬"), ("未勝利", "未勝利")):
        if k in t:
            return v
    return "他"


def _past(td):
    """近4走の1走分（出馬表の右側）"""
    if td is None or not td.select_one(".date"):
        return None
    dm = re.match(r"(\d{4})年(\d+)月(\d+)日", _t(td.select_one(".date")))
    dist_txt = _t(td.select_one(".dist"))
    md = re.match(r"(\d+)\s*(芝|ダ|障)?", dist_txt)
    return {
        "ymd": "%04d%02d%02d" % tuple(int(x) for x in dm.groups()) if dm else "",
        "rc": _t(td.select_one(".rc")),
        "place": _num(_t(td.select_one(".place_line .place")), int),
        "pop": _num(_t(td.select_one(".num .pop")), int),
        "heads": _num(_t(td.select_one(".num .max")), int),
        "surf": (md.group(2) or "") if md else "",
        "dist": int(md.group(1)) if md else None,
        "cond": _t(td.select_one(".condition")),
    }


def parse_entry(html, ymd=None, jo=None, rno=None):
    """出馬表1ページ → dict（race情報 + horses の list）。読めなければ None"""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(decode(html), "lxml")
    hd = s.select_one("div.race_header")
    if hd is None:
        return None
    gi = hd.select_one(".race_name .grade_icon img")
    grade = gi.get("alt", "") if gi else ""
    name = _t(hd.select_one(".race_name")).replace(grade, "").strip()
    course = _t(hd.select_one(".cell.course")).replace("コース：", "").strip()
    md = re.search(r"([\d,]+)", course)
    dist = int(md.group(1).replace(",", "")) if md else 0
    day = _t(hd.select_one(".cell.date"))
    if ymd is None:
        m = re.search(r"(\d{4})年(\d+)月(\d+)日", day)
        ymd = "%04d%02d%02d" % tuple(int(x) for x in m.groups()) if m else ""
    mk = re.search(r"(\d+)回(\S+?)(\d+)日", day)
    if jo is None and mk:
        jo = next((k for k, v in JO_NAME.items() if v == mk.group(2)), "")
    if rno is None:
        m = re.search(r"(\d+)レース", _t(s.select_one("title")) + " " + _t(s))
        rno = int(m.group(1)) if m else 0
    post = _t(hd.select_one(".cell.time")).replace("発走時刻：", "").strip()
    mp = re.search(r"(\d+)時(\d+)分", post)
    cat, klass = _t(hd.select_one(".cell.category")), _t(hd.select_one(".cell.class"))
    surf = "障" if ("障害" in course or "→" in course) else ("ダ" if "ダート" in course else "芝")
    race = {
        "ymd": ymd, "jo": jo, "venue": JO_NAME.get(jo, jo), "rno": int(rno), "name": name, "grade": grade,
        "kai": int(mk.group(1)) if mk else 0, "nichi": int(mk.group(3)) if mk else 0,
        "post": "%02d:%02d" % (int(mp.group(1)), int(mp.group(2))) if mp else "",
        "cat": cat, "cls_txt": klass, "cls": cls_of(" ".join([grade, klass, name])),
        "course": course, "dist": dist, "surf": surf,
        "inout": "外" if "外" in course else ("内" if "内" in course else ""),
        "turn": "左" if "左" in course else ("右" if "右" in course else "直"),
        "horses": [],
    }
    table = s.select_one("table.basic.narrow-xy.mt20") or s.select_one("table")
    if table is None:
        return None
    for tr in table.select("tbody tr"):
        td = tr.select_one("td.horse")
        if td is None:
            continue
        a = td.select_one(".name a")
        hname = _t(a) if a else _t(td.select_one(".name"))
        if not hname:
            continue
        hid = ""
        if a is not None:
            m = re.search(r"pw01dud\d{2}(\d{10})", a.get("href", ""))
            hid = m.group(1) if m else ""
        num_txt = _t(tr.select_one("td.num"))
        waku_img = tr.select_one("td.waku img")
        age = _t(tr.select_one("td.jockey p.age"))
        ma = re.match(r"(牡|牝|セ|せん)\s*(\d+)", age)
        wtxt = _t(td.select_one(".cell.weight"))
        mw = re.match(r"(\d+)kg\s*\(([+\-]?\d+|前計不|計不|初出走)?", wtxt.replace(" ", ""))
        hw = int(mw.group(1)) if mw else None
        hwd = None
        if mw and mw.group(2) and re.fullmatch(r"[+\-]?\d+", mw.group(2)):
            hwd = int(mw.group(2))
        mare = td.select_one("li.mare")
        bms = _t(td.select_one(".bloodmare")).replace("母の父：", "").strip("() ")
        dam = _t(mare).replace("母：", "").replace(_t(td.select_one(".bloodmare")), "").strip() if mare else ""
        blink = tr.select_one("td.jockey .blinker") or tr.select_one(".blinker")
        rec = re.search(r"\((\d+)\.(\d+)\.(\d+)\.(\d+)\)", _t(td.select_one(".cell.result")))
        status = ""
        rowtxt = _t(tr)
        for k in ("取消", "除外", "中止"):
            if k in _t(td.select_one(".name_line")) or (not num_txt.isdigit() and k in rowtxt):
                status = k
        race["horses"].append({
            "num": int(num_txt) if num_txt.isdigit() else None,
            "waku": _num(waku_img.get("alt", ""), int) if waku_img else None,
            "horse": hname, "horse_id": hid,
            "odds": _num(_t(td.select_one(".odds_line .num"))),
            "pop": _num(_t(td.select_one(".pop_rank")), int),
            "sex": {"せん": "セ"}.get(ma.group(1), ma.group(1)) if ma else "",
            "age": int(ma.group(2)) if ma else None,
            "wt": _num(_t(tr.select_one("td.jockey p.weight"))),
            "jockey": re.sub(r"^[▲△◇★☆]+", "", _t(tr.select_one("td.jockey p.jockey")).replace(" ", "")),
            "trainer": _t(td.select_one("p.trainer a")).replace(" ", ""),
            "base": _t(td.select_one("p.trainer .division")).strip("()"),
            "owner": _t(td.select_one("p.owner")), "breeder": _t(td.select_one("p.breeder")).replace(" ", ""),
            "sire": _t(td.select_one("li.sire")).replace("父：", "").strip(),
            "dam": dam, "bms": bms,
            "hw": hw, "hwd": hwd,
            "blinker": "B" if blink is not None else "",
            "rec": [int(x) for x in rec.groups()] if rec else None,
            "status": status,
            "past": [p for p in (_past(tr.select_one("td.past.p%d" % k)) for k in (1, 2, 3, 4)) if p],
        })
    return race
