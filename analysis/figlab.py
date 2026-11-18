#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figlab.py — 指数（能力値 mu）の作り方を変えたときの当たり具合を、短時間で比べる

各週「その週より前のデータだけ」でモデルを作り、その週の全レース（平地・新馬以外）で各馬の mu を出す。
採点：
  mu_only   勝率 = softmax(β·mu)（β だけ合わせる）の対数損失  … 指数そのものの強さ
  mkt_mu    勝率 = softmax(a·log市場 + β·mu) の対数損失        … 市場に足せる情報の量
  hit       mu 1位の勝率
変える方法：--var '{"recency_days": 120, ...}'（P の値、または下の追加パラメータ）

追加パラメータ（第4版の固定値を外に出したもの）
  mg_cap(2.0)   負けた馬の着差（秒/1000m）の上限
  mg_soft(9)    着差がこれを超えた分は mg_soft_k 倍に縮める（大敗の扱い）
  mg_soft_k(1)
  win_k(0.5)    勝ち馬の「2着との差」の反映率
  decay_i(0.92) 何走前かによる重みの減り方
  n_use(12)     使う過去走の数
  dist_scale(900) 距離の違いによる重みの減り方（m）
  other_w(0.2)  芝ダが違う走りの重み
"""
import argparse, collections, datetime as dt, importlib.util, json, math, os, re, sys
import numpy as np

K = None
_W = {}
XP = {"mg_cap": 2.0, "mg_soft": 9.0, "mg_soft_k": 1.0, "win_k": 0.5, "decay_i": 0.92, "n_use": 12, "dist_scale": 900.0, "other_w": 0.2}


def load_engine(path):
    spec = importlib.util.spec_from_file_location("keiba_sim_engine", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_model(K):
    _i, fl, WET = K._i, K.fl, K.WET

    class FigModel(K.Model):
        def __init__(self, runs, params=None, asof=None, xp=None):
            self.X = dict(XP, **(xp or {}))
            super().__init__(runs, params=params, asof=asof)

        def _figures(self):
            Pm, X = self.P, self.X
            elig = []
            for r in self.runs:
                r["F"] = None
                d = r["R"]
                if d is None or r["pl"] is None or not r["t"]:
                    continue
                if r["pl"] == 1:
                    r["mg"] = -X["win_k"] * min((r["dv"] or 0.0) / r["d"] * 1000, 0.5)
                else:
                    g = max(r["t"] - d["win"], 0.0) / r["d"] * 1000
                    if g > X["mg_soft"]:
                        g = X["mg_soft"] + (g - X["mg_soft"]) * X["mg_soft_k"]
                    r["mg"] = min(g, X["mg_cap"])
                r["key"] = (r["ymd"], r["jo"], r["rno"])
                elig.append(r)
            clv = {c: -v for c, v in self.cadj.items()}
            L = {k: clv.get(d["cls"], 0.0) for k, d in self.R.items()}
            for _ in range(6):
                acc = collections.defaultdict(list)
                for r in elig:
                    acc[(r["horse"], r["surf"])].append((r["key"], L[r["key"]] - r["mg"], r["mg"]))
                vote = collections.defaultdict(list)
                for lst in acc.values():
                    n = len(lst)
                    if n < 2:
                        continue
                    tot = sum(v for _, v, _ in lst)
                    for key, v, mg in lst:
                        vote[key].append((tot - v) / (n - 1) + mg)
                raw = collections.defaultdict(list)
                for k, d in self.R.items():
                    xs = vote.get(k, [])
                    xs = [max(min(x, clv.get(d["cls"], 0.0) + 1.5), clv.get(d["cls"], 0.0) - 1.5) for x in xs]
                    L[k] = (sum(xs) + Pm["k_level"] * clv.get(d["cls"], 0.0)) / (len(xs) + Pm["k_level"])
                    if xs:
                        raw[d["cls"]].append(sum(xs) / len(xs))
                for c, v in raw.items():
                    if len(v) >= 8:
                        clv[c] = 0.5 * clv.get(c, 0.0) + 0.5 * (sum(v) / len(v))
                z = clv.get("1勝", 0.0)
                clv = {c: v - z for c, v in clv.items()}
                L = {k: v - z for k, v in L.items()}
            self.clv, self.L = clv, L
            cnt = collections.Counter(d["cls"] for d in self.R.values())
            pts = [(clv[c], self.cadj[c], cnt[c]) for c in clv if c in self.cadj and cnt[c] >= 15]
            sw_ = sum(w for _, _, w in pts) or 1.0
            mx, my = sum(x * w for x, _, w in pts) / sw_, sum(y * w for _, y, w in pts) / sw_
            vx = sum(w * (x - mx) ** 2 for x, _, w in pts) or 1.0
            s_ = -sum(w * (x - mx) * (y - my) for x, y, w in pts) / vx
            self.f2t = (s_, my + s_ * mx)
            for r in elig:
                d = r["R"]
                res = d["res"]
                if d.get("pz") is not None:
                    res -= Pm["k_pace"] * self.pace_slope * d["pz"]
                F = L[r["key"]] - r["mg"] - Pm["w_resid"] * res
                if r["wtv"]:
                    F += Pm["k_wt"] * (r["wtv"] - 55.0)
                rp = (r["c4"] - 1) / max(r["hd"] - 1, 1) if r["c4"] and r["hd"] > 1 else None
                if rp is not None and d.get("pz") is not None:
                    F += Pm["k_style_run"] * d["pz"] * (0.5 - rp) * 2
                pp = _i(r.get("pop"))
                if pp and r["hd"] > 1:
                    F += Pm["k_pop"] * (0.5 - (pp - 1) / (r["hd"] - 1))
                r["F"], r["Fq"] = F, 1.0
                r["rp1"] = (r["c1"] - 1) / max(r["hd"] - 1, 1) if r["c1"] and r["hd"] > 1 else None
                r["rp4"] = rp
            for r in self.runs:
                if r["F"] is None and not r["jra"] and r["pl"] and r["cls"] in ("G1", "G2", "G3") and r["dv"] is not None and r["d"]:
                    mg = (0 if r["pl"] == 1 else r["dv"]) / r["d"] * 1000
                    r["F"], r["Fq"] = clv.get(r["cls"], 0.8) - 0.15 - min(mg, 2.0), 0.5

    return FigModel


def patch_rate(K, X):
    """rate() の中の固定値（0.92、12走、900m、0.2）を X の値に置き換えた版を作る"""
    import inspect, textwrap
    src = textwrap.dedent(inspect.getsource(K.Model.rate))
    rep = [("rs[:12]", "rs[:int(_X['n_use'])]"), ("(0.92 ** i)", "(_X['decay_i'] ** i)"),
           ("/ 900.0)", "/ _X['dist_scale'])"), ("w *= 0.2", "w *= _X['other_w']")]
    for a, b in rep:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    ns = dict(vars(K)); ns["_X"] = X
    exec(src, ns)
    return ns["rate"]


def _init(db, engine, since, until, var, every, var2=None):
    global K
    K = load_engine(engine)
    _, runs = K.load_db(db)
    byrace = collections.defaultdict(list)
    for r in runs:
        if r.get("odds") and since <= r["ymd"] <= until:
            byrace[(r["ymd"], r["jo"], int(r["rno"]))].append(r)
    P = {k: v for k, v in var.items() if k in K.P}
    X = dict(XP, **{k: v for k, v in var.items() if k in XP})
    unknown = [k for k in var if k not in K.P and k not in XP]
    assert not unknown, unknown
    FM = make_model(K)
    FM.rate = patch_rate(K, X)
    _W.update(runs=runs, byrace=byrace, P=P, X=X, FM=FM, var2=None)
    if var2 is not None:
        X2 = dict(XP, **{k: v for k, v in var2.items() if k in XP})
        FM2 = make_model(K)
        FM2.rate = patch_rate(K, X2)
        _W.update(var2=({k: v for k, v in var2.items() if k in K.P}, X2, FM2))


def _work(group):
    asof, dates = group
    M = _W["FM"](_W["runs"], params=_W["P"], asof=asof, xp=_W["X"])
    jk = M.jockeys(asof)
    M2 = jk2 = None
    if _W.get("var2"):
        P2, X2, FM2 = _W["var2"]
        M2 = FM2(_W["runs"], params=P2, asof=asof, xp=X2)
        jk2 = M2.jockeys(asof)
    out = []
    for k in sorted(_W["byrace"]):
        if k[0] not in dates:
            continue
        rs = _W["byrace"][k]
        r0 = rs[0]
        cls = K.cls_of(r0["grade"], r0["race"])
        if r0["surf"] == "障" or cls == "新馬" or len(rs) < 5 or not r0["dist"] or sum(r["place"] == "1" for r in rs) != 1:
            continue
        race = {"date": k[0], "jo": k[1], "r": k[2], "surf": r0["surf"], "dist": int(r0["dist"]), "cls": cls, "horses": []}
        for r in sorted(rs, key=lambda r: int(r["gate"] or 0)):
            past = [p for p in M.by_horse.get(r["horse"], []) if p["ymd"] < k[0] and p["pl"]]
            rec = [sum(1 for p in past if p["pl"] == i) for i in (1, 2, 3)] + [sum(1 for p in past if p["pl"] > 3)]
            race["horses"].append({"name": r["horse"], "num": r["gate"], "sex": r["sex"], "age": K._i(r["age"], 0), "wt": K.fl(r["wt"]),
                                   "jockey": r["jockey"], "rec": rec if sum(rec) else None, "base": "", "runs": [], "trainer": r.get("trainer") or ""})
        inv = [1.0 / K.fl(r["odds"]) for r in sorted(rs, key=lambda r: int(r["gate"] or 0))]
        rid = "%s:%s:%d" % k
        for h, r, iv in zip(race["horses"], sorted(rs, key=lambda r: int(r["gate"] or 0)), inv):
            x = M.rate(h, race, jk)
            mu2 = M2.rate(h, race, jk2)["mu"] if M2 else 0.0
            out.append((rid, x["mu"], iv / sum(inv), int(r["place"] == "1"), mu2))
    return out


def fit_eval(rows):
    """rows=(rid, mu, pm, win) → mu_only・mkt_mu の対数損失（係数はこの rows に合わせる＝変種どうしの比較用）"""
    from scipy.optimize import minimize
    rid = np.array([r[0] for r in rows]); mu = np.array([r[1] for r in rows]); pm = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows])
    _, g = np.unique(rid, return_inverse=True)
    order = np.argsort(g, kind="stable"); g, mu, pm, y = g[order], mu[order], pm[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    nr = len(starts)
    mu = mu - np.add.reduceat(mu, starts)[g] / np.bincount(g)[g]
    lpm = np.log(pm)

    def ll(w, X):
        s = X @ w
        mx = np.maximum.reduceat(s, starts)
        lse = np.log(np.add.reduceat(np.exp(s - mx[g]), starts)) + mx
        return -(s[y == 1].sum() - lse.sum()) / nr

    r1 = minimize(lambda w: ll(w, mu[:, None]), [1.5], method="Nelder-Mead")
    r2 = minimize(lambda w: ll(w, np.column_stack([lpm, mu])), [1.0, 0.2], method="Nelder-Mead")
    r0 = ll(np.array([1.0]), lpm[:, None])
    if len(rows[0]) > 4 and any(r[4] for r in rows):
        mu2 = np.array([r[4] for r in rows])[order]
        mu2 = mu2 - np.add.reduceat(mu2, starts)[g] / np.bincount(g)[g]
        r3 = minimize(lambda w: ll(w, np.column_stack([lpm, mu, mu2])), [1.0, 0.2, 0.1], method="Nelder-Mead")
        print("    市場＋mu＋mu2: %.4f（市場比 %+.4f）係数 %s" % (r3.fun, r0 - r3.fun, np.round(r3.x, 3)))
    top = np.maximum.reduceat(mu, starts)
    hit = (y[mu == top[g]]).sum() / nr
    return {"races": nr, "mu_only": r1.fun, "beta": float(r1.x[0]), "mkt_mu": r2.fun, "market": r0, "gain_vs_mkt": r0 - r2.fun, "hit": hit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="db"); ap.add_argument("--engine", default="keiba_sim_v4.py")
    ap.add_argument("--since", default="20240101"); ap.add_argument("--until", default="20241231")
    ap.add_argument("--var", action="append", default=[], help='JSON。複数指定すると順に比べる（"{}" が第4版そのまま）')
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--log", default="figlab_results.jsonl")
    ap.add_argument("--var2", help="2つ目の指数（市場との合わせ方で mu と並べて使う）")
    a = ap.parse_args()
    from multiprocessing import Pool
    for vs in a.var or ["{}"]:
        var = json.loads(vs)
        v2 = json.loads(a.var2) if a.var2 else None
        _init(a.db, a.engine, a.since, a.until, var, a.every, v2)
        days = sorted({k[0] for k in _W["byrace"]})
        groups = []
        for d in days:
            dd = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
            if groups and (dd - groups[-1][2]).days <= 2:
                groups[-1][1].add(d); groups[-1][2] = dd
            else:
                groups.append([d, {d}, dd])
        groups = groups[::a.every]
        _W.clear()
        rows = []
        with Pool(a.procs, initializer=_init, initargs=(a.db, a.engine, a.since, a.until, var, a.every, v2)) as p:
            for out in p.imap_unordered(_work, [(g[0], g[1]) for g in groups]):
                rows += out
        res = fit_eval(rows)
        res["var"] = var; res["period"] = "%s-%s/%d" % (a.since, a.until, a.every)
        print("%-60s races %d  mu_only %.4f (β %.2f)  mkt_mu %.4f  市場比 %+.4f  hit %.3f" % (
            json.dumps(var, ensure_ascii=False)[:60], res["races"], res["mu_only"], res["beta"], res["mkt_mu"], res["gain_vs_mkt"], res["hit"]), flush=True)
        with open(a.log, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
