"""ベンチマークの評価

1. 確率の精度：対数損失（3着内・勝ち）を市場と比べた「市場超え度」（1 − LL/LL市場）、AUC
2. 運用ルール：方法 × 対象（中穴/大穴/中穴＋大穴/1〜5番人気/全頭）× 選び方（複勝期待値・単勝期待値・
   1レース1頭・エッジ比）× 閾値 を総当たりし、前半（2023〜24年）で最良を選び、後半（2025〜26年）で確かめる
3. 券種：選んだ馬を軸に、単勝・複勝・ワイド・馬連・3連複（相手は人気馬）を確定払戻で比べる
4. 賭け金：均等 vs ケリー（4分の1）
5. 絞り込み：芝ダ・場・距離・クラス・頭数・馬場ごとに、前半と後半の両方で良いか
"""
import numpy as np
import pandas as pd

from .bench import TEST_YEARS, VAL_YEARS, _ll

BANDS = {
    "中穴": lambda d: d["tier"] == "中穴",
    "大穴": lambda d: d["tier"] == "大穴",
    "中穴＋大穴": lambda d: d["tier"] != "",
    "1〜5番人気": lambda d: d["pop"] <= 5,
    "全頭": lambda d: d["pop"] > 0,
}
THR = np.round(np.arange(0.70, 1.41, 0.05), 2)


def attach(df, pred, pay):
    d = df[["race_id", "year", "ymd", "jo", "rno", "num", "horse", "odds", "pop", "tier", "top3", "win", "place_pay",
            "heads", "surf", "dist", "cls_r", "cond", "mkt_top3"]].loc[pred.index].copy()
    d["p_top3"] = pred["p_top3"].values
    d["p_win"] = pred["p_win"].values if "p_win" in pred else np.nan
    d["exp_pay"] = pay.loc[pred.index].values
    d["ev_place"] = d["p_top3"] * d["exp_pay"] / 100
    d["ev_win"] = d["p_win"] * d["odds"]
    d["ret_place"] = d["place_pay"] / 100
    d["ret_win"] = d["win"] * d["odds"]
    return d


# ───────────────────────── 1. 確率の精度 ─────────────────────────
def prob_quality(preds, df, mkt="mkt"):
    from sklearn.metrics import roc_auc_score
    rows = []
    m = preds[mkt]
    for name, p in preds.items():
        d = df.loc[p.index]
        for band, f in BANDS.items():
            sel = f(d).values
            if sel.sum() < 1000:
                continue
            y3, yw = d["top3"].values[sel], d["win"].values[sel]
            ll3 = _ll(y3, p["p_top3"].values[sel])
            ll3m = _ll(y3, m.loc[p.index, "p_top3"].values[sel])
            llw = _ll(yw, p["p_win"].values[sel])
            llwm = _ll(yw, m.loc[p.index, "p_win"].values[sel])
            rows.append({"方法": name, "対象": band, "頭数": int(sel.sum()),
                         "3着内 市場超え度(%)": 100 * (1 - ll3 / ll3m), "勝ち 市場超え度(%)": 100 * (1 - llw / llwm),
                         "AUC(3着内)": roc_auc_score(y3, p["p_top3"].values[sel]),
                         "予想平均/実際": p["p_top3"].values[sel].mean() / y3.mean()})
    return pd.DataFrame(rows)


# ───────────────────────── 2. 運用ルール ─────────────────────────
def _roi(r):
    return r.mean() if len(r) else np.nan


def boot_ci(d, ret_col, n=1000, seed=0):
    """レース単位で再標本化した回収率の95%区間"""
    if d.empty:
        return (np.nan, np.nan)
    g = d.groupby("race_id")[ret_col].agg(["sum", "size"])
    s, c = g["sum"].values, g["size"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    vals = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def select(d, family, band, t):
    base = d[BANDS[band](d)]
    if family == "複勝期待値":
        return base[base["ev_place"] >= t], "ret_place"
    if family == "単勝期待値":
        return base[base["ev_win"] >= t], "ret_win"
    if family == "1レース1頭(複勝)":
        b = base[base["ev_place"] >= t]
        return b.loc[b.groupby("race_id")["ev_place"].idxmax()], "ret_place"
    if family == "1レース1頭(単勝)":
        b = base[base["ev_win"] >= t]
        return b.loc[b.groupby("race_id")["ev_win"].idxmax()], "ret_win"
    if family == "エッジ比(複勝)":   # 市場の3着内率の何倍か（tを倍率として 1.0〜1.7 に読み替え）
        r = 1.0 + (t - 0.7)
        return base[(base["p_top3"] / base["mkt_top3"] >= r) & (base["p_top3"] >= 0.05)], "ret_place"
    raise ValueError(family)


FAMILIES = ["複勝期待値", "単勝期待値", "1レース1頭(複勝)", "1レース1頭(単勝)", "エッジ比(複勝)"]


def policy_grid(d):
    """全ての (選び方, 対象, 閾値) について、前半・後半・年ごとの回収率"""
    rows = []
    val = d["year"].isin(VAL_YEARS)
    for fam in FAMILIES:
        for band in BANDS:
            for t in THR:
                s, rc = select(d, fam, band, t)
                if len(s) < 50:
                    continue
                v, te = s[val.loc[s.index]], s[~val.loc[s.index]]
                row = {"選び方": fam, "対象": band, "閾値": t, "券": "単勝" if rc == "ret_win" else "複勝",
                       "前半頭数": len(v), "前半回収": _roi(v[rc]), "後半頭数": len(te), "後半回収": _roi(te[rc])}
                for y, g in s.groupby("year"):
                    row["y%d" % y] = _roi(g[rc])
                # 前半の回収率の下側（標準誤差1つ分）：件数が少なく偶然に高いものを選ばないため
                se = v[rc].std(ddof=1) / np.sqrt(max(len(v), 1)) if len(v) > 1 else np.nan
                row["前半下限"] = row["前半回収"] - se
                rows.append(row)
    return pd.DataFrame(rows)


def choose(grid, min_val=400, key="前半下限"):
    """前半だけを見て選ぶ（後半は見ない）"""
    g = grid[grid["前半頭数"] >= min_val]
    return g.sort_values(key, ascending=False)


def base_rates(d):
    rows = []
    for band, f in BANDS.items():
        b = d[f(d)]
        for yrs, lab in ((VAL_YEARS, "前半"), (TEST_YEARS, "後半")):
            x = b[b["year"].isin(yrs)]
            rows.append({"対象": band, "期間": lab, "頭数": len(x), "複勝回収": _roi(x["ret_place"]),
                         "単勝回収": _roi(x["ret_win"])})
    return pd.DataFrame(rows)


# ───────────────────────── 3. 券種 ─────────────────────────
def _pay(pays, kind, combo):
    """netkeiba の払戻 → 組番（順不同の券種はソートして比較）に一致する金額"""
    tot = 0
    for c, amt in pays.get(kind, []):
        nums = [int(x) for x in c.replace("→", "-").split("-") if x.strip().isdigit()]
        if kind in ("ワイド", "馬連", "三連複", "枠連"):
            if sorted(nums) == sorted(combo):
                tot += amt
        elif nums == list(combo):
            tot += amt
    return tot


BET_KINDS = ("単勝", "複勝", "ワイド 1番人気へ", "ワイド 1・2番人気へ（2点）", "ワイド 1〜3番人気へ（3点）",
             "馬連 1〜3番人気へ（3点）", "3連複 軸＋1〜3番人気から2頭（3点）", "3連複 軸＋1〜5番人気から2頭（10点）")


def bet_returns(sel, df_all, nk, race_id_map):
    """sel：軸の馬（1頭1行）。相手は同じレースの人気上位 → 1軸1行で、券種ごとの（点数, 払戻）"""
    import itertools
    by_race = df_all.groupby("race_id")
    rows = []
    for _, r in sel.iterrows():
        rid = race_id_map.get(r["race_id"])
        info = nk.get(rid) if rid else None
        if not info or not info.get("pays"):
            continue
        pays = info["pays"]
        g = by_race.get_group(r["race_id"]).sort_values("pop")
        favs = [int(x) for x in g["num"].values if int(x) != int(r["num"])][:5]
        a = int(r["num"])
        tickets = {
            "単勝": ([(a,)], "単勝"), "複勝": ([(a,)], "複勝"),
            "ワイド 1番人気へ": ([(a, favs[0])], "ワイド"),
            "ワイド 1・2番人気へ（2点）": ([(a, f) for f in favs[:2]], "ワイド"),
            "ワイド 1〜3番人気へ（3点）": ([(a, f) for f in favs[:3]], "ワイド"),
            "馬連 1〜3番人気へ（3点）": ([(a, f) for f in favs[:3]], "馬連"),
            "3連複 軸＋1〜3番人気から2頭（3点）": ([(a,) + c for c in itertools.combinations(favs[:3], 2)], "三連複"),
            "3連複 軸＋1〜5番人気から2頭（10点）": ([(a,) + c for c in itertools.combinations(favs[:5], 2)], "三連複"),
        }
        row = {"race_id": r["race_id"], "year": r["year"]}
        for k, (ts, kind) in tickets.items():
            row[k + "|n"] = len(ts)
            row[k + "|ret"] = sum(_pay(pays, kind, t) for t in ts)
        rows.append(row)
    return pd.DataFrame(rows)


def bet_types(sel, df_all, nk, race_id_map, n_boot=1000, seed=0):
    br = bet_returns(sel, df_all, nk, race_id_map)
    if br.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(br), size=(n_boot, len(br)))
    out = []
    for k in BET_KINDS:
        n, ret = br[k + "|n"].values, br[k + "|ret"].values
        roi = ret.sum() / (100 * n.sum())
        boots = ret[idx].sum(axis=1) / (100 * n[idx].sum(axis=1))
        row = {"券種": k, "軸の頭数": len(br), "点数": int(n.sum()), "的中": int((ret > 0).sum()), "回収率": roi,
               "95%区間": "%.0f〜%.0f%%" % (100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5))}
        for y, g in br.groupby("year"):
            row[str(y)] = g[k + "|ret"].sum() / (100 * g[k + "|n"].sum())
        out.append(row)
    return pd.DataFrame(out)


# ───────────────────────── 4. 賭け金 ─────────────────────────
def staking(sel, ret_col, prob_col, odds_col, frac=0.25, cap=0.02):
    """時系列で資金を動かす。均等＝毎回資金の0.5%、ケリー＝期待値に応じて（frac倍、上限cap）"""
    s = sel.sort_values(["ymd", "race_id"])
    res = {}
    for mode in ("均等", "ケリー"):
        bank, peak, mdd = 1.0, 1.0, 0.0
        stake_sum = ret_sum = 0.0
        for ymd, g in s.groupby("ymd", sort=True):
            day_bets = []
            for _, r in g.iterrows():
                o = r[odds_col]
                p = r[prob_col]
                if mode == "均等":
                    f = 0.005
                else:
                    k = (p * o - 1) / (o - 1) if o > 1 else 0
                    f = min(max(k * frac, 0.0), cap)
                if f > 0:
                    day_bets.append((f * bank, r[ret_col]))
            for st, ret in day_bets:   # 同じ日の賭けは、その日の朝の資金で決める
                bank += st * (ret - 1)
                stake_sum += st
                ret_sum += st * ret
            peak = max(peak, bank)
            mdd = max(mdd, 1 - bank / peak)
        res[mode] = {"最終資金（初期1）": bank, "最大下落": mdd, "回収率": ret_sum / stake_sum if stake_sum else np.nan}
    return res


# ───────────────────────── 5. 絞り込み ─────────────────────────
def segments(sel, ret_col):
    s = sel.copy()
    s["距離帯"] = pd.cut(s["dist"], [0, 1300, 1600, 2000, 2400, 4000], labels=["〜1300", "〜1600", "〜2000", "〜2400", "2400超"])
    s["クラス"] = s["cls_r"].map({0: "新馬", 1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝"}).fillna("OP以上")
    s["頭数帯"] = pd.cut(s["heads"], [0, 11, 14, 16, 18], labels=["〜11", "12〜14", "15〜16", "17〜18"])
    s["馬場"] = s["cond"].map(lambda c: "重・不良" if c in ("重", "不良") else "良・稍重")
    s["芝ダ"] = s["surf"]
    from .config import JO_NAME
    s["場"] = s["jo"].map(JO_NAME)
    rows = []
    val = s["year"].isin(VAL_YEARS)
    for col in ("芝ダ", "場", "距離帯", "クラス", "頭数帯", "馬場"):
        for k, g in s.groupby(col, observed=True):
            v, t = g[val.loc[g.index]], g[~val.loc[g.index]]
            rows.append({"軸": col, "区分": k, "前半頭数": len(v), "前半回収": _roi(v[ret_col]),
                         "後半頭数": len(t), "後半回収": _roi(t[ret_col])})
    return pd.DataFrame(rows)
