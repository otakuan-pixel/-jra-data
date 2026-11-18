#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""features.py — 第4版エンジンの中身を1頭ずつ分解して書き出す（2段目のモデル用）

validate と同じ「その週より前のデータだけでモデルを作る」通し方式で、各レースの各馬について
  ・SIM（第4版）の勝率・複勝率（1000回）
  ・能力値 mu と、その内訳（第4版の各補正がいくら足されたか）
  ・オッズ込みの期待値（MK）で使う項目
  ・新しい候補の特徴（ブリンカー、馬体重の増減、指数のばらつき、馬主・生産者の勝率など）
  ・確定単勝オッズ、着順
を1行にして CSV に書き出す。

使い方: python3 -I features.py --db db --engine keiba_sim_v4.py --since 20230101 --out feats.csv.gz [--procs 4]
"""
import argparse, collections, csv, datetime as dt, gzip, importlib.util, math, os, re, sys

K = None   # エンジン（keiba_sim.py）モジュール


def load_engine(path):
    spec = importlib.util.spec_from_file_location("keiba_sim_engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def days_between(a, b):
    return (dt.date(int(a[:4]), int(a[4:6]), int(a[6:])) - dt.date(int(b[:4]), int(b[4:6]), int(b[6:]))).days


def make_fmodel(K):
    _i, fl, WET, P = K._i, K.fl, K.WET, K.P

    class FModel(K.Model):
        """rate() の各補正を comp に記録する版（計算そのものは第4版と同じ）"""

        def rate(self, h, race, jk):
            Pm = self.P
            date, surf, dist = race["date"], race["surf"], race["dist"]
            d0 = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
            seen, rs = set(), []
            for r in list(self.by_horse.get(h["name"], [])) + [self._adhoc(p, h["name"]) for p in h.get("runs", [])]:
                if r is None or r["ymd"] >= date or r["ymd"] in seen:
                    continue
                seen.add(r["ymd"])
                rs.append(r)
            rs.sort(key=lambda r: r["ymd"], reverse=True)
            prior = self.clv.get(race["cls"], 0.0) - Pm["prior_gap"]
            ws, fs, jw, e_w, e_v, wet, dry = [], [], [], [], [], [], []
            same_surf = near_dist = 0
            for i, r in enumerate(rs[:12]):
                if r["F"] is None or (r["surf"] == "障") != (surf == "障"):
                    continue
                days = (d0 - dt.date(int(r["ymd"][:4]), int(r["ymd"][4:6]), int(r["ymd"][6:]))).days
                w = 0.5 ** (days / Pm["recency_days"]) * (0.92 ** i) * r.get("Fq", 1.0)
                if r["surf"] == surf:
                    same_surf += 1
                    if abs(r["d"] - dist) <= 300:
                        near_dist += 1
                else:
                    w *= 0.2
                w *= math.exp(-abs(r["d"] - dist) / 900.0)
                ws.append(w); fs.append(r["F"]); jw.append(jk.get(r["jockey"], 0.0))
                if r.get("rp1") is not None:
                    e_w.append(w + 0.05); e_v.append(0.65 * r["rp1"] + 0.35 * r["rp4"])
                if r["surf"] == surf and r["jra"]:
                    (wet if WET.get(r["cond"], 0) >= 0.45 else dry).append(r["F"])
            c = collections.OrderedDict()

            def add(k, v):
                c[k] = c.get(k, 0.0) + v

            sw = sum(ws)
            if sw > 0:
                avg = sum(w * f for w, f in zip(ws, fs)) / sw
                good = sorted((f for w, f in zip(ws, fs) if w >= 0.2 * max(ws)), reverse=True)[:2]
                top = sum(good) / len(good)
                mu0 = (1 - Pm["top_blend"]) * avg + Pm["top_blend"] * top
                mu0 = (sw * mu0 + Pm["prior_n"] * prior) / (sw + Pm["prior_n"])
                jpast = sum(w * j for w, j in zip(ws, jw)) / sw
            else:
                avg = top = None
                mu0, jpast = prior - Pm["n0_gap"], 0.0
            mu = mu0
            if h.get("rec") and sum(h["rec"]) >= 3:
                n_st = sum(h["rec"])
                add("rec", Pm["k_rec"] * ((h["rec"][0] + h["rec"][1] + h["rec"][2] + 1.0) / (n_st + 3.3) - 0.3))
            if h.get("base") == "栗東":
                add("west", Pm["k_west"])
            add("wt_now", -Pm["k_wt_now"] * ((h.get("wt") or 55.0) - 55.0))
            field = [x for x in race.get("horses", []) if not x.get("cancel")]
            if len({x.get("age") for x in field if x.get("age")}) > 1 and h.get("age"):
                add("age", Pm["age_adj"].get(min(int(h["age"]), 7), 0.0))
            if h.get("sex") == "牝" and any(x.get("sex") and x.get("sex") != "牝" for x in field):
                add("fem", Pm["fem_adj"])
            jn = re.sub(r"^[▲△☆★◇◆]", "", h.get("jockey") or "")
            jd = Pm["k_jockey"] * (jk.get(jn, 0.0) - jpast)
            add("jd", jd)
            lay = None
            if rs:
                lay = (d0 - dt.date(int(rs[0]["ymd"][:4]), int(rs[0]["ymd"][4:6]), int(rs[0]["ymd"][6:]))).days
                if lay > 120:
                    add("layoff", -Pm["k_layoff"] * min(1.0, (lay - 120) / 180.0))
            mkf = {}
            fp = [r for r in rs if r["F"] is not None and r["surf"] != "障"]
            if surf != "障":
                js = self.jockey_stats(date).get(jn, (0, 0, 0.0))
                mkf["jk_wr"] = math.log(((js[1] + 3.5) / (js[0] + 50.0)) / 0.07)
                mkf["closer"] = 1.0 if len(e_w) >= 2 and sum(w * v for w, v in zip(e_w, e_v)) / sum(e_w) >= 0.8 else 0.0
                mkf["n1"] = 1.0 if len(fs) == 1 else 0.0
                mkf["a3"] = 1.0 if _i(h.get("age"), 0) == 3 and len({x.get("age") for x in field if x.get("age")}) > 1 else 0.0
                t_wr0 = self.trainer_stats(date)[0]
                if h.get("trainer"):
                    tk0 = self._name_key(h["trainer"], t_wr0)
                    if tk0 in t_wr0 and t_wr0[tk0][0] >= 1:
                        mkf["tr_wr"] = math.log(((t_wr0[tk0][1] + 3.5) / (t_wr0[tk0][0] + 50.0)) / 0.07)
                if fp:
                    lp0 = fp[0]["pl"] or 0
                    mkf["l10"] = 1.0 if lp0 >= 10 else 0.0
                    mkf["l6_9"] = 1.0 if 6 <= lp0 <= 9 else 0.0
                    mkf["drop"] = 1.0 if self.clv.get(race["cls"], 0.0) < self.clv.get(fp[0]["cls"], 0.0) - 0.05 else 0.0
                    mkf["quick"] = 1.0 if days_between(date, fp[0]["ymd"]) <= 15 else 0.0
                add("jk_wr", Pm["k_jk_wr"] * mkf["jk_wr"])
                add("jk_vm", Pm["k_jk_vm"] * 50.0 * (js[1] - js[2]) / (js[0] + 100.0))
                m_f = len(field)
                g_rel = (_i(h.get("num"), 0) - 1) / (m_f - 1) if m_f >= 12 and _i(h.get("num"), 0) else None
                if g_rel is not None and surf in ("芝", "ダ"):
                    add("gate", (Pm["gate_d"] if surf == "ダ" else Pm["gate_t"]) * (g_rel - 0.5))
                if m_f >= 2 and _i(h.get("num"), 0):
                    add("draw", Pm["k_draw"] * (0.5 - (_i(h["num"]) - 1) / (m_f - 1)) * self.draw.get((race.get("jo"), surf, dist), 0.0) * 10.0)
                if len(fs) == 2:
                    add("n2", Pm["n2_adj"])
                if len(fs) <= 2 and h.get("trainer"):
                    t_wr, t_y = self.trainer_stats(date)
                    tk = self._name_key(h["trainer"], t_wr)
                    if tk in t_wr:
                        add("tr_young", Pm["k_tr_young"] * math.log(((t_wr[tk][1] + 3.5) / (t_wr[tk][0] + 50.0)) / 0.07))
                    add("tr_debut", Pm["k_tr_debut"] * t_y.get(self._name_key(h["trainer"], t_y), 0.0))
                if h.get("rec") and sum(h["rec"]) >= 3:
                    add("winrate", Pm["k_winrate"] * max(-0.1, min(0.4, (h["rec"][0] + 0.3) / (sum(h["rec"]) + 3.0) - 0.1)))
                if len(e_w) >= 2:
                    add("front", Pm["k_front"] * (0.5 - sum(w * v for w, v in zip(e_w, e_v)) / sum(e_w)))
                if fp:
                    l = fp[0]
                    lp = l["pl"] or 0
                    add("last", Pm["last_adj"].get(2 if lp == 2 else 3 if lp == 3 else 6 if 6 <= lp <= 9 else 10 if lp >= 10 else 0, 0.0))
                    if len(fs) >= 3 and sw > 0:
                        add("trend", Pm["k_trend"] * max(-1.0, min(1.0, l["F"] - sum(w * f for w, f in zip(ws, fs)) / sw)))
                    c_now = self.clv.get(race["cls"], 0.0)
                    if c_now < self.clv.get(l["cls"], 0.0) - 0.05:
                        add("drop", Pm["drop_adj"])
                    add("maxcls", Pm["k_maxcls"] * max(0.0, min(1.0, max(self.clv.get(r["cls"], 0.0) for r in fp) - c_now)))
                    if days_between(date, l["ymd"]) <= 15:
                        add("quick", Pm["quick_adj"])
                    if l["surf"] == surf and dist - l["d"] <= -400:
                        add("short", Pm["short_adj"])
                    hws = [_i(r.get("hw")) for r in fp[:6] if _i(r.get("hw"))]
                    if hws:
                        add("hw", Pm["k_hw"] * max(-2.5, min(2.5, (sum(hws) / len(hws) - 470.0) / 30.0)))
                    here = [r["F"] for r in fp[:12] if r["jo"] == race.get("jo") and r["surf"] == surf]
                    if here:
                        f_all = sum(r["F"] for r in fp[:12]) / len(fp[:12])
                        add("course", Pm["k_course"] * max(-0.5, min(0.5, (sum(here) / len(here) - f_all) * len(here) / (len(here) + 3.0))))
            if sw > 0 and same_surf == 0:
                add("new_surf", -0.25)
            elif sw > 0 and near_dist == 0:
                add("new_dist", -Pm["k_dist"])
            mu += sum(c.values())
            apt = 0.0
            if wet and dry:
                apt = max(-0.5, min(0.5, len(wet) / (len(wet) + 2.5) * (sum(wet) / len(wet) - sum(dry) / len(dry))))
            sig = Pm["sigma0"] * Pm["sigma_scale"]
            e = sum(w * v for w, v in zip(e_w, e_v)) / sum(e_w) if e_w else 0.5
            out = {"mu": mu, "sig": sig, "apt": apt, "e": e, "e_n": len(e_w), "n": len(fs), "sw": sw, "lay": lay,
                   "jd": jd, "notes": [], "n_wet": len(wet), "best": max(fs) if fs else None, "avg": avg,
                   "last_f3": None, "mkf": mkf,
                   # ここから記録用
                   "mu0": mu0, "comp": dict(c), "top": top, "prior": prior,
                   "f_sd": (sum((f - sum(fs[:8]) / len(fs[:8])) ** 2 for f in fs[:8]) / len(fs[:8])) ** 0.5 if len(fs[:8]) >= 3 else None,
                   "f_last": fp[0]["F"] if fp else None,
                   "rs": rs}
            self._log.append((h["name"], out))
            return out

    return FModel


COMP_KEYS = ["rec", "west", "wt_now", "age", "fem", "jd", "layoff", "jk_wr", "jk_vm", "gate", "draw", "n2", "tr_young",
             "tr_debut", "winrate", "front", "last", "trend", "drop", "maxcls", "quick", "short", "hw", "course", "new_surf", "new_dist"]
MKF_KEYS = ["jk_wr", "closer", "n1", "a3", "tr_wr", "l10", "l6_9", "drop", "quick"]

_W = {}


def group_stats(runs, key, asof, years=2):
    """key（馬主・生産者・種別など）ごとの直近 years 年の（出走数, 勝ち数, 単勝オッズから見た勝ちの見込み）"""
    cut = "%04d%s" % (int(asof[:4]) - years, asof[4:])
    acc = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in runs:
        if cut <= r["ymd"] < asof and r["place"].isdigit() and r["surf"] != "障":
            od = K.fl(r.get("odds"))
            k = r.get(key) or ""
            if od and k:
                a = acc[k]
                a[0] += 1; a[1] += r["place"] == "1"; a[2] += 0.8 / od
    return acc


def _init(db, engine, since, until, n):
    global K
    K = load_engine(engine)
    _, runs = K.load_db(db)
    byrace = collections.defaultdict(list)
    for r in runs:
        if r.get("odds") and since <= r["ymd"] <= until:
            byrace[(r["ymd"], r["jo"], int(r["rno"]))].append(r)
    _W.update(runs=runs, byrace=byrace, n=n, FModel=make_fmodel(K))


def _work(group):
    asof, dates = group
    FModel = _W["FModel"]
    M = FModel(_W["runs"], asof=asof)
    own = group_stats(_W["runs"], "owner", asof)
    brd = group_stats(_W["runs"], "breeder", asof)
    rows = []
    for k in sorted(_W["byrace"]):
        if k[0] not in dates:
            continue
        rs = _W["byrace"][k]
        r0 = rs[0]
        cls = K.cls_of(r0["grade"], r0["race"])
        if r0["surf"] == "障" or cls == "新馬" or len(rs) < 5 or not r0["dist"] or not any(r["place"] == "1" for r in rs):
            continue
        race = {"date": k[0], "jo": k[1], "r": k[2], "surf": r0["surf"], "dist": int(r0["dist"]), "cls": cls, "name": r0["race"], "horses": []}
        for r in sorted(rs, key=lambda r: int(r["gate"] or 0)):
            past = [p for p in M.by_horse.get(r["horse"], []) if p["ymd"] < k[0] and p["pl"]]
            rec = [sum(1 for p in past if p["pl"] == i) for i in (1, 2, 3)] + [sum(1 for p in past if p["pl"] > 3)]
            race["horses"].append({"name": r["horse"], "num": r["gate"], "waku": "", "odds": K.fl(r["odds"]), "pop": K._i(r["pop"]),
                                   "sex": r["sex"], "age": K._i(r["age"], 0), "wt": K.fl(r["wt"]), "jockey": r["jockey"],
                                   "rec": rec if sum(rec) else None, "base": "", "runs": [], "place": int(r["place"]) if r["place"].isdigit() else 99,
                                   "trainer": r.get("trainer") or "", "_row": r})
        cond = r0["cond"] if r0["cond"] in K.WET else "良"
        M._log = []
        o = K.simulate(M, race, going={cond: 1.0}, n=_W["n"])
        rts = dict(M._log)
        mk = K.market_ev(M, race) or []
        mkp = {x["num"]: x["p"] for x in mk}
        n, m = _W["n"], len(race["horses"])
        fidx = M.front_index(k[1], r0["surf"], int(r0["dist"]), cond)
        for h, oh in zip(race["horses"], o["horses"]):
            rt = rts[h["name"]]
            row = h["_row"]
            prev = rt["rs"][0] if rt["rs"] else None
            hw_now = K._i(row.get("hw"))
            hw_prev = K._i(prev.get("hw")) if prev else None
            past_b = [p.get("blinker") == "B" for p in rt["rs"][:5]]
            b_now = (row.get("blinker") or "").strip() == "B"
            ow = own.get(row.get("owner") or "", [0, 0, 0.0])
            br = brd.get(row.get("breeder") or "", [0, 0, 0.0])
            jn_prev = prev.get("jockey") if prev else None
            out = {"date": k[0], "jo": k[1], "rno": k[2], "surf": r0["surf"], "dist": r0["dist"], "cls": cls, "cond": cond, "m": m,
                   "num": h["num"], "horse": h["name"], "place": h["place"], "odds": h["odds"],
                   "p_sim": (oh["win"] * n + 0.5) / (n + 0.5 * m), "p3_sim": oh["top3"], "p_mk": mkp.get(h["num"]),
                   "mu": rt["mu"], "mu0": rt["mu0"], "mu_t": rt.get("mu_t"), "k_brd_vm": rt["mkf"].get("brd_vm"), "apt": rt["apt"], "e": rt["e"], "e_n": rt["e_n"], "n_runs": rt["n"],
                   "sw": rt["sw"], "lay": rt["lay"], "f_sd": rt["f_sd"], "f_last": rt["f_last"], "f_top": rt["top"], "f_avg": rt["avg"],
                   "prior": rt["prior"], "fidx": fidx, "age": h["age"], "sex": h["sex"], "wt": h["wt"],
                   "b_now": int(b_now), "b_first": int(b_now and not any(past_b)), "b_off": int((not b_now) and bool(past_b) and past_b[0]),
                   "hw_now": hw_now, "hw_diff": (hw_now - hw_prev) if hw_now and hw_prev else None,
                   "jk_change": int(bool(jn_prev) and jn_prev != h["jockey"]),
                   "dist_prev": prev["d"] if prev else None, "surf_prev": prev["surf"] if prev else None,
                   "own_n": ow[0], "own_w": ow[1], "own_x": ow[2], "brd_n": br[0], "brd_w": br[1], "brd_x": br[2]}
            for kk in COMP_KEYS:
                out["c_" + kk] = rt["comp"].get(kk, 0.0)
            for kk in MKF_KEYS:
                out["k_" + kk] = rt["mkf"].get(kk, 0.0)
            rows.append(out)
    print("  %s の週 %d 頭" % (asof, len(rows)), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--engine", required=True)
    ap.add_argument("--since", default="20230101"); ap.add_argument("--until", default="99999999")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from multiprocessing import Pool
    _init(a.db, a.engine, a.since, a.until, a.n)
    days = sorted({k[0] for k in _W["byrace"]})
    groups = []
    for d in days:
        dd = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
        if groups and (dd - groups[-1][2]).days <= 2:
            groups[-1][1].add(d); groups[-1][2] = dd
        else:
            groups.append([d, {d}, dd])
    print("%s〜%s：%d 週" % (days[0], days[-1], len(groups)), flush=True)
    _W.clear()   # 子プロセスは自分で読む
    w = None
    with gzip.open(a.out, "wt", encoding="utf-8", newline="") as f, \
            Pool(a.procs, initializer=_init, initargs=(a.db, a.engine, a.since, a.until, a.n)) as p:
        for rows in p.imap(_work, [(g[0], g[1]) for g in groups]):
            if not rows:
                continue
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
            w.writerows(rows)
            f.flush()


if __name__ == "__main__":
    main()
