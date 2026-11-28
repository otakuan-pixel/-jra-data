"""発走30分前の自動レポート（GitHub Actions から10分おきに呼ぶ）

  python -m anaba live                 # 今日の、発走まで12〜45分のレースで、まだ報告していないもの
  python -m anaba live --race 05_11    # 指定レースを今すぐ
  python -m anaba replay --date 20261122 --race 05_11   # 過去のレースで動作確認（その日以降の成績は使わない）
"""
import json
import os
from pathlib import Path

from . import horses as H
from .backfill import fetch_peds_for
from .config import JRA2_REPO
from .data import load_results
from .entries import parse_entry
from .live import (Http, card_to_runs, jra_race_card, jra_race_tokens, nk_going, nk_odds, nk_race_id, now_jst,
                   post_datetime)
from .model import Bundle
from .report import day_dir, race_markdown, write_picks
from .scoring import score_cards


def _load_state(ymd):
    p = day_dir(ymd) / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_state(ymd, st):
    p = day_dir(ymd) / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def _update_index(ymd, st):
    rows = ["# 大穴レポート %s-%s-%s" % (ymd[:4], ymd[4:6], ymd[6:]), "",
            "| 発走 | レース | 印（馬番） | レポート |", "|---|---|---|---|"]
    for k, r in sorted(st.get("races", {}).items(), key=lambda kv: (kv[1].get("post", ""), kv[0])):
        if not r.get("reported"):
            continue
        rows.append("| %s | %s | %s | [%s.md](%s.md) |" % (r.get("post", ""), r.get("label", k), r.get("marks", ""), k, k))
    (day_dir(ymd) / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def snapshot_odds(http, ymd, st, now, horizon=120, log=print):
    """発走前2時間以内の全レースの単勝・複勝オッズを記録する（締切までのオッズの動きを、今後の検証に使うため）
    → reports/<日付>/odds_snap.csv（1行＝1頭×1回）"""
    import csv
    path = day_dir(ymd) / "odds_snap.csv"
    new = not path.exists()
    rows = []
    for key, r in sorted(st.get("races", {}).items()):
        pdt = post_datetime(ymd, r.get("post"))
        if not pdt or r.get("surf") == "障":
            continue
        mins = (pdt - now).total_seconds() / 60
        if not (0 < mins <= horizon):
            continue
        jo, rno = key.split("_")
        odds, ts = nk_odds(http, nk_race_id(ymd, jo, r["kai"], r["nichi"], rno))
        for num, o in sorted(odds.items()):
            rows.append([key, int(mins), ts or "", num, o.get("win"), o.get("pop"), o.get("place_lo"), o.get("place_hi")])
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["race", "mins_to_post", "odds_time", "num", "win", "pop", "place_lo", "place_hi"])
            w.writerows(rows)
        log("オッズ記録：%d 行" % len(rows))


def ensure_day(http, ymd, st, log=print, prefetch_peds=True, max_ped=600):
    """その日のレース一覧（cname・発走時刻）を作る。初回だけ全レースの出馬表を読む"""
    if st.get("races"):
        return st
    toks = jra_race_tokens(http, ymd, log=log)
    st["races"] = {}
    ids, cards_seen = [], []
    for (jo, rno), t in sorted(toks.items()):
        card = jra_race_card(http, t["cname"], ymd, jo, rno)
        if not card:
            continue
        key = "%s_%02d" % (jo, rno)
        st["races"][key] = {"cname": t["cname"], "kai": t["kai"], "nichi": t["nichi"], "post": card["post"],
                            "label": "%s%dR %s" % (card["venue"], rno, card["name"]), "surf": card["surf"],
                            "reported": None}
        ids += [h["horse_id"] for h in card["horses"] if h["horse_id"]]
        cards_seen.append(card)
    st["checked"] = now_jst().isoformat()
    H.add_from_cards(cards_seen, ymd)
    log("レース一覧：%d レース" % len(st["races"]))
    if prefetch_peds and ids:
        fetch_peds_for(ids, max_requests=max_ped, log=log)
    return st


_MEMO = {}


def _loaded():
    """成績データ・モデル・馬の索引は、常駐ループの中では1回だけ読む"""
    if "hist" not in _MEMO:
        _, _MEMO["hist"] = load_results(JRA2_REPO)
        _MEMO["bundle"] = Bundle.load()
        _MEMO["horses"] = H.load()
    return _MEMO["hist"], _MEMO["bundle"], _MEMO["horses"]


def run_live(ymd=None, lo=25, hi=35, only=None, notify=True, prefetch=False, snapshot=True, log=print):
    from .notify import send
    now = now_jst()
    ymd = ymd or now.strftime("%Y%m%d")
    st = _load_state(ymd)
    http = Http(interval=1.0)
    st = ensure_day(http, ymd, st, log=log, prefetch_peds=prefetch)
    _save_state(ymd, st)
    if not st.get("races"):
        log("%s は開催がありません" % ymd)
        return []
    targets = []
    for key, r in st["races"].items():
        if r.get("surf") == "障":
            continue
        if only:
            if key in only:
                targets.append(key)
            continue
        pd_ = post_datetime(ymd, r["post"])
        mins = (pd_ - now).total_seconds() / 60 if pd_ else None
        if r.get("reported") or mins is None:
            continue
        # 予定どおりなら 25〜35分前（10分おきの実行で1回は入る）。Actions の遅れで過ぎても、発走3分前までは出す
        if lo <= mins <= hi or (3 <= mins < lo):
            targets.append(key)
    if snapshot and not only:
        snapshot_odds(http, ymd, st, now, log=log)
    if not targets:
        log("報告するレースはありません（%s）" % now.strftime("%H:%M"))
        return []
    hist, bundle, horses = _loaded()
    cards, runs_list, meta = [], [], {}
    for key in sorted(targets):
        r = st["races"][key]
        jo, rno = key.split("_")
        card = jra_race_card(http, r["cname"], ymd, jo, int(rno))
        if not card:
            log("出馬表を取得できません：%s" % key)
            continue
        rid = nk_race_id(ymd, jo, r["kai"], r["nichi"], rno)
        odds, ts = nk_odds(http, rid)
        _, cond = nk_going(http, rid)
        cr = card_to_runs(card, cond, odds)
        if cr.empty:
            log("馬番が未確定のため判定できません：%s" % key)
            continue
        cr = cr[cr["status"] == ""]
        if cr["odds"].isna().mean() > 0.3:
            log("オッズ未発売のため判定できません：%s（次の回で再試行）" % key)
            continue
        # 発走前オッズが付いていない馬は取消・除外とみなして外す（netkeiba は取消馬にオッズを付けない）
        miss = cr["odds"].isna()
        if odds:
            miss = miss | ~cr["num"].isin(list(odds))
        gone = cr[miss]
        if len(gone):
            log("取消・除外とみなす：%s %s" % (key, "、".join(gone["horse"])))
            cr = cr.drop(gone.index)
            cr["pop"] = cr["odds"].rank(method="first")
        fetch_peds_for(list(cr["horse_id"]), max_requests=40, log=log)
        cards.append(card)
        runs_list.append(cr)
        meta[key] = (ts or now_jst().strftime("%H:%M"), rid)
    if not cards:
        return []
    scored = score_cards(runs_list, hist=hist, bundle=bundle, horses=horses)
    out = []
    for card in cards:
        key = "%s_%02d" % (card["jo"], card["rno"])
        d = scored[(scored["jo"] == card["jo"]) & (scored["rno"] == card["rno"])]
        pdt = post_datetime(ymd, card["post"])
        mins = int((pdt - now_jst()).total_seconds() // 60) if pdt else None
        md, picks = race_markdown(d, card, odds_ts=meta[key][0], minutes_to_post=mins)
        p = day_dir(ymd) / (key + ".md")
        p.write_text(md + "\n", encoding="utf-8")
        write_picks(day_dir(ymd) / "picks.csv", d, card, now_jst().isoformat(timespec="minutes"))
        st["races"][key]["reported"] = now_jst().isoformat(timespec="minutes")
        st["races"][key]["marks"] = " ".join("%s%d" % (x["mark"] or "血", x["num"]) for x in picks)
        if notify and (picks or os.environ.get("ANABA_NOTIFY", "marked") == "all"):
            send(ymd, md, log=log)
        out.append((key, md))
        log("報告：%s %s" % (key, st["races"][key]["marks"] or "（印なし）"))
    _save_state(ymd, st)
    _update_index(ymd, st)
    return out


def git_sync(msg, log=print):
    """レポートをコミットして push（GitHub Actions の中だけ）。常駐ループで報告のたびに呼ぶ"""
    import subprocess
    if not os.environ.get("GITHUB_ACTIONS"):
        return False
    run = lambda *a: subprocess.run(list(a), capture_output=True, text=True)
    run("git", "add", "reports", "data/pedigree", "data/horses.csv")
    if run("git", "diff", "--cached", "--quiet").returncode == 0:
        return False
    run("git", "commit", "-q", "-m", msg)
    for _ in range(3):
        if run("git", "pull", "--rebase", "-q").returncode == 0 and run("git", "push", "-q").returncode == 0:
            return True
    log("push に失敗しました（次の回で再試行）")
    return False


def run_loop(ymd=None, until="17:10", max_minutes=345, interval=60, lo=28, hi=32, snap_every=600, log=print):
    """常駐ループ：1分ごとに発走30分前（28〜32分前）のレースを判定し、報告したらすぐコミット・通知する。
    GitHub の定時起動は数時間遅れることがあるので、朝に1回起動すればその日の残りを取りこぼさないようにする。
    終わる条件：その日の残りのレースがすべて報告済み／until を過ぎた／max_minutes を超えた"""
    import time as _t
    t0 = _t.time()
    now = now_jst()
    ymd = ymd or now.strftime("%Y%m%d")
    h, m = until.split(":")
    end = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
    last_snap = 0.0
    log("常駐開始：%s（%s まで、最長 %d 分）" % (ymd, until, max_minutes))
    while True:
        now = now_jst()
        snap = _t.time() - last_snap >= snap_every
        try:
            out = run_live(ymd, lo=lo, hi=hi, snapshot=snap, log=log)
        except Exception as e:      # 1回の失敗でループ全体を止めない
            import traceback
            log("エラー（続行）：%r\n%s" % (e, traceback.format_exc()[-800:]))
            out = []
        if snap:
            last_snap = _t.time()
        if out or snap:
            git_sync("大穴レポート %s" % now.strftime("%Y-%m-%dT%H:%M"), log=log)
        st = _load_state(ymd)
        races = st.get("races") or {}
        if not races:
            log("開催なし → 終了")
            break
        left = [k for k, r in races.items() if r.get("surf") != "障" and not r.get("reported")
                and post_datetime(ymd, r.get("post")) and post_datetime(ymd, r["post"]) > now_jst()]
        if not left:
            log("全レース報告済み → 終了")
            break
        if now_jst() >= end or (_t.time() - t0) / 60 >= max_minutes:
            log("時間切れ → 終了（残り %d レースは次の起動で）" % len(left))
            break
        _t.sleep(max(5, interval - (_t.time() - now.timestamp())))
    git_sync("大穴レポート %s" % now_jst().strftime("%Y-%m-%dT%H:%M"), log=log)


def replay(ymd, race_key, log=print):
    """過去のレースを、jra2 に残っている出馬表と確定オッズで再現する（その日以降の成績は使わない）"""
    jo, rno = race_key.split("_")
    path = Path(JRA2_REPO) / "data" / "entries" / ymd / ("%s_%s.html" % (jo, rno))
    if not path.exists():
        raise FileNotFoundError(path)
    card = parse_entry(path.read_bytes(), ymd, jo, int(rno))
    _, hist = load_results(JRA2_REPO)
    res = hist[(hist["ymd"] == ymd) & (hist["jo"] == jo) & (hist["rno"] == int(rno))]
    odds = {int(r.num): {"win": r.odds, "pop": int(r["pop"])} for _, r in res.iterrows() if r.odds == r.odds}
    cond = res["cond"].iloc[0] if len(res) else ""
    cr = card_to_runs(card, cond, odds)
    cr = cr[cr["status"] == ""]
    scored = score_cards([cr], hist=hist, horses=H.load(), cutoff=ymd)
    md, picks = race_markdown(scored, card, odds_ts="確定オッズ（再現）")
    if len(res):
        top = res.sort_values("place").head(3)
        md += "\n\n実際の結果：" + " → ".join("%d着 %d %s（%d番人気）" % (r.place, r.num, r.horse, r["pop"])
                                             for _, r in top.iterrows() if r.place == r.place)
    return md, scored


def replay_days(dates, out_dir=None, log=print):
    """過去の開催日をまるごと再現する（最初の日より前のデータだけで学習したモデル・確定オッズ）
    → reports/replay/<最初の日>-<最後の日>/ にレースごとのレポートと、印ごとの成績のまとめ"""
    import pandas as pd

    from . import payout
    from .config import MARK_RULES, TIERS, tier_series
    from .data import KEY
    from .report import mark_of
    dates = sorted(dates)
    out_dir = Path(out_dir or (day_dir(dates[0]).parent / "replay" / ("%s-%s" % (dates[0], dates[-1]))))
    out_dir.mkdir(parents=True, exist_ok=True)
    races, hist = load_results(JRA2_REPO)
    cards, runs_list, results = [], [], {}
    for ymd in dates:
        for path in sorted((Path(JRA2_REPO) / "data" / "entries" / ymd).glob("*.html")):
            jo, rno = path.stem.split("_")
            card = parse_entry(path.read_bytes(), ymd, jo, int(rno))
            if not card or card["surf"] == "障":
                continue
            res = hist[(hist["ymd"] == ymd) & (hist["jo"] == jo) & (hist["rno"] == int(rno))]
            ok = res[res["status"].isin(["", "中止", "降着"]) & res["odds"].notna()]
            if ok.empty:
                continue
            odds = {int(r.num): {"win": r.odds, "pop": int(r["pop"])} for _, r in ok.iterrows()}
            cr = card_to_runs(card, res["cond"].iloc[0], odds)
            cr = cr[(cr["status"] == "") & cr["num"].isin(list(odds))]
            cards.append(card)
            runs_list.append(cr)
            results[(ymd, jo, int(rno))] = res
    log("再現：%d 日・%d レース" % (len(dates), len(cards)))
    scored = score_cards(runs_list, hist=hist, horses=H.load(), cutoff=dates[0], log=log)
    pp = payout.place_payouts(races, hist)
    scored = scored.merge(pp, on=KEY + ["num"], how="left")
    scored["place_pay"] = scored["place_pay"].fillna(0.0)
    res_all = pd.concat(results.values())[KEY + ["num", "place", "odds"]].rename(columns={"place": "fin", "odds": "fodds"})
    scored = scored.merge(res_all, on=KEY + ["num"], how="left")
    mb = scored.apply(mark_of, axis=1)
    scored["mark"] = [m for m, _ in mb]
    scored["bet"] = [b for _, b in mb]
    scored["tier"] = tier_series(scored)
    index = ["| 日付 | レース | 印 | 1〜3着（人気） |", "|---|---|---|---|"]
    for card in cards:
        d = scored[(scored["ymd"] == card["ymd"]) & (scored["jo"] == card["jo"]) & (scored["rno"] == card["rno"])]
        md, picks = race_markdown(d, card, odds_ts="確定オッズ（再現）")
        res = results[(card["ymd"], card["jo"], card["rno"])]
        top = res[res["place"] <= 3].sort_values("place")
        line = " → ".join("%d %s（%d）" % (r.num, r.horse, r["pop"]) for _, r in top.iterrows())
        md += "\n\n実際の結果：" + line + "\n"
        name = "%s_%s_%02d.md" % (card["ymd"], card["jo"], card["rno"])
        (out_dir / name).write_text(md, encoding="utf-8")
        marks = " ".join("%s%d" % (x["mark"] or "血", x["num"]) for x in picks)
        index.append("| %s | [%s](%s) | %s | %s |" % (card["ymd"], race_label_short(card), name, marks, line))
    rows = ["| 区分 | 選び方 | 頭数 | 3着内 | 複勝回収率 | 単勝回収率 |", "|---|---|--:|--:|--:|--:|"]
    for tier in TIERS:
        c = scored[scored["tier"] == tier]
        sels = [("印%s（%s・%s）" % (mk, bet, "買い候補" if buy else "参考"), c["mark"] == mk) for mk, _, _, bet, buy in MARK_RULES[tier]]
        for nm, m in sels + [("全部", c["tier"] == tier)]:
            g = c[m]
            if len(g):
                rows.append("| %s | %s | %d | %d | %d%% | %d%% |" % (
                    tier, nm, len(g), int((g["fin"] <= 3).sum()), round(g["place_pay"].mean()),
                    round(100 * ((g["fin"] == 1) * g["fodds"]).mean())))
    head = ["# 再現：%s〜%s（%dレース）" % (dates[0], dates[-1], len(cards)), "",
            "モデルは %s より前のデータだけで学習し、確定オッズで判定（本番は発走30分前のオッズなので少し条件が違う）。"
            "件数が少ないので成績の証拠にはならない。出力の見本として見ること。" % dates[0], "",
            "## 区分・印ごとの成績", ""] + rows + ["", "## レース一覧", ""] + index
    (out_dir / "README.md").write_text("\n".join(head) + "\n", encoding="utf-8")
    log("保存：%s" % out_dir)
    return scored


def race_label_short(card):
    from .config import JO_NAME
    return "%s%dR %s" % (JO_NAME.get(card["jo"], card["jo"]), card["rno"], card["name"])
