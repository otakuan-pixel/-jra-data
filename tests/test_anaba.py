"""オフラインで動く検査（ネットワーク・成績データ不要）

  python -m pytest -q
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).parent / "fixtures"


# ───────── 出馬表 ─────────
def test_parse_entry_fixture():
    from anaba.entries import parse_entry
    r = parse_entry((FIX / "entry_20261122_05_11.html").read_bytes())
    assert r["ymd"] == "20261122" and r["jo"] == "05" and r["rno"] == 11
    assert r["surf"] == "ダ" and r["dist"] == 1400 and r["post"] == "15:20" and r["cls"] == "OP"
    hs = r["horses"]
    assert len(hs) == 16
    h1 = hs[0]
    assert h1["horse"] == "メリディアンスター" and h1["horse_id"] == "2022100905"
    assert h1["sire"] == "モズアスコット" and h1["dam"] == "ニシノケイト" and h1["bms"] == "ゴールドアリュール"
    assert h1["odds"] == 5.3 and h1["pop"] == 3 and h1["hw"] == 492 and h1["hwd"] == 0
    assert len(h1["past"]) == 4
    # ブリンカー
    assert {h["horse"] for h in hs if h["blinker"] == "B"} == {"サトノカルナバル", "マーゴットゲイン"}
    # 見習いの印（▲など）は外れている
    assert not any(h["jockey"][:1] in "▲△◇★☆" for h in hs)


def test_card_to_runs_types():
    from anaba.entries import parse_entry
    from anaba.live import card_to_runs
    card = parse_entry((FIX / "entry_20261122_05_11.html").read_bytes())
    df = card_to_runs(card, "良", {9: {"win": 25.8, "pop": 10, "place_lo": 4.8, "place_hi": 8.3}})
    assert len(df) == 16
    assert df.loc[df["num"] == 9, "odds"].iloc[0] == 25.8
    assert df["hw"].dtype.kind in "if" and df["odds"].dtype.kind == "f" and df["hwd"].dtype.kind in "if"
    assert set(df["status"]) == {""}


# ───────── 5代血統・クロス ─────────
def test_parse_ped_fixture():
    from anaba.pedigree import crosses, parse_ped, summarize
    anc = parse_ped((FIX / "ped_2022100905.html").read_bytes())
    assert len(anc) == 62
    by = {a["path"]: a for a in anc}
    assert by["F"]["name"] == "モズアスコット" and by["FF"]["name"] == "Frankel" and by["FFF"]["name"] == "Galileo"
    assert by["MF"]["name"] == "ゴールドアリュール"
    cs = {c["name"]: c for c in crosses(anc) if not c["derived"]}
    assert cs["Halo"]["label"] == "4×5" and cs["Northern Dancer"]["label"] == "5×5"
    s = summarize(anc)
    assert s["miracle"] is False and s["sire"] == "モズアスコット" and s["bms"] == "ゴールドアリュール"


def _synthetic(cross_paths, name="X"):
    """全マスに別々の祖先を置き、cross_paths のマスだけ同じ祖先にする"""
    from anaba.pedigree import PATHS
    anc = []
    for p in PATHS:
        if p in cross_paths:
            anc.append({"path": p, "gen": len(p), "id": "x", "name": name})
        else:
            anc.append({"path": p, "gen": len(p), "id": "id_" + p, "name": "A" + p})
    return anc


def test_miracle_3x4():
    from anaba.pedigree import summarize
    s = summarize(_synthetic({"FFF", "MFFF"}, "サンデーサイレンス"))
    assert s["miracle"] and s["miracle_names"] == ["サンデーサイレンス 3×4"]
    assert s["max_pct"] == pytest.approx(18.75)


def test_miracle_4x3_and_not_4x4():
    from anaba.pedigree import summarize
    assert summarize(_synthetic({"FFFF", "MFF"}))["miracle"]          # 4×3 も奇跡の血量
    s = summarize(_synthetic({"FFFF", "MFFF"}))                      # 4×4 = 12.5%
    assert not s["miracle"] and s["max_pct"] == pytest.approx(12.5)
    s = summarize(_synthetic({"FFFF", "FMFF", "MFFF"}))               # 4×4×4 = 18.75%（相当）
    assert not s["miracle"] and s["miracle_eq"]


def test_derived_cross_hidden():
    """X が 3×4 なら、X の父（FFFF と MFFFF の位置）も重なるが、派生なので表に出さない"""
    from anaba.pedigree import crosses
    anc = _synthetic({"FFF", "MFFF"}, "X")
    for a in anc:
        if a["path"] in ("FFFF", "MFFFF"):
            a["id"], a["name"] = "y", "Xの父"
    cs = crosses(anc)
    assert [c["name"] for c in cs if not c["derived"]] == ["X"]
    assert any(c["name"] == "Xの父" and c["derived"] for c in cs)


def test_pedstore_roundtrip(tmp_path):
    from anaba.pedigree import PedStore, parse_ped
    anc = parse_ped((FIX / "ped_2022100905.html").read_bytes())
    st = PedStore(tmp_path)
    st.add("2022100905", anc)
    st2 = PedStore(tmp_path)
    assert "2022100905" in st2
    a2 = {a["path"]: a for a in st2.anc("2022100905")}
    assert a2["FF"]["name"] == "Frankel" and len(a2) == 62
    assert st2.summary("2022100905")["crosses"][0]["name"] == "Halo"
    assert (tmp_path / "peds-2022.jsonl").exists()


def test_pedstore_migrates_legacy(tmp_path):
    from anaba.pedigree import PedStore, parse_ped
    anc = parse_ped((FIX / "ped_2022100905.html").read_bytes())
    legacy = {"horse_id": "2022100905", "anc": [[a["path"], a["id"], a["name"]] for a in anc], "fetched": "2026-11-26"}
    (tmp_path / "peds.jsonl").write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")
    st = PedStore(tmp_path)
    assert "2022100905" in st and not (tmp_path / "peds.jsonl").exists()
    assert st.summary("2022100905")["sire"] == "モズアスコット"


# ───────── 市場の確率・名前の正規化 ─────────
def test_harville_sums_to_three():
    from anaba.features import harville_top3
    p = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.04, 0.03])
    t = harville_top3(p / p.sum())
    assert t.sum() == pytest.approx(3.0, abs=1e-6)
    assert np.all(np.diff(t) <= 1e-12)          # 勝率の順と同じ並び


def test_breeder_key():
    from anaba.data import breeder_key
    assert breeder_key("安平ノーザンファーム") == "ノーザンファーム"
    assert breeder_key("新ひだか田湯牧場") == "田湯牧場"
    assert breeder_key("英Godolphin") == "Godolphin"
    assert breeder_key("(有)社台ファーム") == "社台ファーム"
    assert breeder_key("ノーザンファーム") == "ノーザンファーム"


# ───────── 発走30分前の判定 ─────────
def test_due_window(tmp_path):
    from anaba.due import JST, is_due
    d = tmp_path / "reports" / "20261129"
    d.mkdir(parents=True)
    st = {"races": {"05_11": {"post": "15:00", "reported": None, "surf": "芝"},
                    "05_12": {"post": "15:40", "reported": None, "surf": "芝"}}}
    (d / "state.json").write_text(json.dumps(st), encoding="utf-8")
    now = datetime(2026, 11, 29, 14, 30, tzinfo=JST)          # 11R の30分前
    assert is_due("20261129", now, root=tmp_path)[0]
    now = datetime(2026, 11, 29, 14, 0, tzinfo=JST)           # どれも60分以上前
    assert not is_due("20261129", now, root=tmp_path)[0]
    st["races"]["05_11"]["reported"] = "2026-11-29T14:30"
    (d / "state.json").write_text(json.dumps(st), encoding="utf-8")
    now = datetime(2026, 11, 29, 14, 32, tzinfo=JST)
    assert not is_due("20261129", now, root=tmp_path)[0]      # 報告済み
    assert is_due("20261130", now, root=tmp_path)[0]          # state.json が無い日は初回処理が必要


def test_netkeiba_race_id():
    from anaba.live import nk_race_id
    assert nk_race_id("20261122", "05", 5, 5, 11) == "202605050511"


def test_tier_of():
    from anaba.config import tier_of
    assert tier_of(15.0, 7) == "中穴"
    assert tier_of(29.9, 10) == "中穴"        # 10番人気でも30倍未満は中穴
    assert tier_of(30.0, 10) == "大穴"
    assert tier_of(45.0, 9) == "大穴"         # 人気とずれたらオッズで分ける
    assert tier_of(8.0, 6) == ""              # 10倍未満は対象外
    assert tier_of(50.0, 5) == ""             # 5番人気以内は対象外
    assert tier_of(None, 12) == ""


def test_mark_rules():
    from anaba.report import is_buy, mark_of
    base = {"ev_win": 0.0, "ev_place": 0.0}
    assert mark_of({**base, "odds": 20.0, "pop": 8, "ev_win": 1.5}) == ("★", "単勝")
    assert mark_of({**base, "odds": 20.0, "pop": 8, "ev_win": 1.2}) == ("", "")      # 1.1〜1.45 の帯は印なし
    assert mark_of({**base, "odds": 60.0, "pop": 12, "ev_place": 1.05}) == ("△", "複勝")
    assert mark_of({**base, "odds": 60.0, "pop": 12, "ev_win": 3.0}) == ("", "")      # 大穴の単勝期待値では印を付けない
    assert is_buy("★") and not is_buy("△") and not is_buy("")


def test_should_notify():
    from anaba.runner import should_notify
    star, ref, blood = {"mark": "★"}, {"mark": "△"}, {"mark": ""}
    assert should_notify([star, blood], "buy") and not should_notify([ref, blood], "buy")
    assert not should_notify([], "buy")
    assert should_notify([ref], "marked") and not should_notify([], "marked")
    assert should_notify([], "all")
