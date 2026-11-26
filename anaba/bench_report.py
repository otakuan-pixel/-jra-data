"""ベンチマークの結果をまとめて reports/research/benchmark.md に書く

  python -m anaba bench          # 予測の総当たり（.cache に保存、続きから再開可）
  python -m anaba bench-report   # 評価してレポートを書く
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import bench, bench_eval as E
from .backfill import load_nk_races
from .config import MODEL_DIR, REPORT_DIR


def _md(t, fmt=None, index=False):
    fmt = fmt or {}
    t = t.reset_index() if index else t
    cols = list(t.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in t.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                cells.append("—")
            elif f and isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(f % v)
            else:
                cells.append(str(v))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def nk_id(race_id):
    return race_id[:4] + race_id[8:]


def evaluate(log=print):
    df, preds = bench.run(log=log)
    pay = bench.pay_model_by_year(df)
    out = {}
    # 1. 確率の精度
    pq = E.prob_quality(preds, df)
    out["prob"] = pq
    # 2. 運用ルール（方法ごとに前半で選んで後半で確かめる）
    grids, best = {}, []
    for name, p in preds.items():
        d = E.attach(df, p, pay)
        g = E.policy_grid(d)
        g.insert(0, "方法", name)
        grids[name] = g
        c = E.choose(g)
        if len(c):
            top = c.iloc[0]
            s, rc = E.select(d, top["選び方"], top["対象"], top["閾値"])
            te = s[s["year"].isin(bench.TEST_YEARS)]
            lo, hi = E.boot_ci(te, rc)
            best.append({**top.to_dict(), "後半95%区間": "%.0f〜%.0f%%" % (100 * lo, 100 * hi)})
        log("  評価：%s" % name)
    out["grid"] = pd.concat(grids.values(), ignore_index=True)
    out["best"] = pd.DataFrame(best).sort_values("後半回収", ascending=False)
    d0 = E.attach(df, preds["mkt"], pay)
    out["base"] = E.base_rates(d0)
    return df, preds, pay, out


def detail(df, preds, pay, method, family, band, t, log=print):
    """採用候補の深掘り：券種・賭け金・絞り込み・年ごと"""
    d = E.attach(df, preds[method], pay)
    s, rc = E.select(d, family, band, t)
    res = {}
    res["by_year"] = s.groupby("year").agg(頭数=(rc, "size"), 回収率=(rc, "mean"), 三着内率=("top3", "mean"),
                                           平均オッズ=("odds", "mean"))
    te = s[s["year"].isin(bench.TEST_YEARS)]
    res["ci"] = E.boot_ci(te, rc)
    nk = load_nk_races()
    have = {rid for rid in nk}
    te_nk = te[te["race_id"].map(nk_id).isin(have)]
    rmap = {r: nk_id(r) for r in te_nk["race_id"].unique()}
    res["bet_cov"] = (len(te_nk), len(te))
    res["bets"] = E.bet_types(te_nk, df[df["race_id"].isin(rmap)], nk, rmap)
    prob = "p_win" if rc == "ret_win" else "p_top3"
    odds = "odds" if rc == "ret_win" else "exp_pay_x"
    s = s.assign(exp_pay_x=s["exp_pay"] / 100)
    res["stake"] = E.staking(s[s["year"].isin(bench.TEST_YEARS)], rc, prob, odds)
    res["seg"] = E.segments(s, rc)
    return res


def write(out, details, chosen, path=None, log=print):
    path = path or REPORT_DIR / "research" / "benchmark.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    pf = {"3着内 市場超え度(%)": "%.2f", "勝ち 市場超え度(%)": "%.2f", "AUC(3着内)": "%.4f", "予想平均/実際": "%.3f"}
    rf = {"前半回収": "%.3f", "後半回収": "%.3f", "前半下限": "%.3f", "閾値": "%.2f", "y2023": "%.3f", "y2024": "%.3f",
          "y2025": "%.3f", "y2026": "%.3f"}
    L = ["# 予測方法・運用ルールの総当たり検証", "",
         "更新：%s。JRA平地 2021年〜。2023〜26年を毎年「その年より前のデータだけ」で学習して予想（ウォークフォワード）。"
         % datetime.now().strftime("%Y-%m-%d"),
         "運用ルール（閾値など）は **前半（2023〜24年）だけで選び、後半（2025〜26年）で確かめる**。後半の数字が本当の実力の目安。", "",
         "## 採用した運用", ""] + chosen["text"] + ["",
         "## 1. 確率の精度（市場超え度：対数損失が市場モデルより何%小さいか。＋なら市場より正確）", "",
         "### 中穴＋大穴", "", _md(out["prob"][out["prob"]["対象"] == "中穴＋大穴"].drop(columns="対象")
                                .sort_values("3着内 市場超え度(%)", ascending=False), pf), "",
         "### 全頭", "", _md(out["prob"][out["prob"]["対象"] == "全頭"].drop(columns="対象")
                           .sort_values("3着内 市場超え度(%)", ascending=False), pf), "",
         "方法の説明は `anaba/bench.py` の先頭。`-speed` などは現行（gbm6）からその特徴の群を抜いたもの（市場超え度が下がれば、その群は効いている）。", "",
         "## 2. 何もしない場合（全部買う）", "", _md(out["base"], {"複勝回収": "%.3f", "単勝回収": "%.3f"}), "",
         "## 3. 方法ごとの最良ルール（前半の「回収率−標準誤差」が最大のもの → 後半で確認）", "",
         _md(out["best"][["方法", "選び方", "対象", "閾値", "券", "前半頭数", "前半回収", "後半頭数", "後半回収", "後半95%区間",
                          "y2023", "y2024", "y2025", "y2026"]], rf), "",
         "## 4. 採用候補の深掘り", ""]
    for name, res in details.items():
        lo, hi = res["ci"]
        L += ["### " + name, "", "年ごと", "", _md(res["by_year"], {"回収率": "%.3f", "三着内率": "%.3f", "平均オッズ": "%.1f"}, index=True),
              "", "後半の回収率の95%%区間（レース単位のブートストラップ）：%.0f〜%.0f%%" % (100 * lo, 100 * hi), "",
              "券種（後半・確定払戻があるレース %d / %d 頭。相手は同じレースの人気上位）" % res["bet_cov"], "",
              _md(res["bets"], {"回収率": "%.3f"}), "", "賭け金（後半、時系列で資金を動かした場合）", "",
              _md(pd.DataFrame(res["stake"]).T, {"最終資金（初期1）": "%.3f", "最大下落": "%.3f", "回収率": "%.3f"}, index=True), "",
              "絞り込み（前半と後半の両方で、全体より良いか）", "",
              _md(res["seg"], {"前半回収": "%.3f", "後半回収": "%.3f"}), ""]
    L += ["## 5. 全ての組み合わせ（前半の頭数400以上、前半の下限順に上位40）", "",
          _md(E.choose(out["grid"]).head(40)[["方法", "選び方", "対象", "閾値", "券", "前半頭数", "前半回収", "前半下限", "後半頭数", "後半回収"]], rf), ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log("保存：%s" % path)
    return path


def save_policy(chosen):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "policy.json").write_text(json.dumps(chosen["policy"], ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────── 最終レポート（採用ルールの根拠をすべて載せる） ─────────────────────────
def _quantile_table(df, preds, pay, methods, band, col, rc, qs=(0.05, 0.1, 0.2)):
    rows = []
    for name in methods:
        d = E.attach(df, preds[name], pay)
        b = d[E.BANDS[band](d)].copy()
        b["rk"] = b.groupby("year")[col].rank(pct=True, ascending=False)
        for q in qs:
            s = b[b["rk"] <= q]
            r = {"方法": name, "上位": "%d%%" % round(100 * q), "頭数": len(s), "通算": s[rc].mean()}
            for y, g in s.groupby("year"):
                r[str(y)] = g[rc].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def _threshold_table(d, band, col, rc, ts):
    b = d[E.BANDS[band](d)]
    rows = []
    for t in ts:
        s = b[b[col] >= t]
        v, te = s[s["year"].isin(bench.VAL_YEARS)], s[s["year"].isin(bench.TEST_YEARS)]
        se = v[rc].std() / np.sqrt(len(v)) if len(v) > 1 else np.nan
        lo, hi = E.boot_ci(te, rc)
        r = {"閾値": t, "前半頭数": len(v), "前半回収": v[rc].mean(), "前半下限": v[rc].mean() - se,
             "後半頭数": len(te), "後半回収": te[rc].mean(), "後半95%区間": "%.0f〜%.0f%%" % (100 * lo, 100 * hi)}
        for y, g in s.groupby("year"):
            r[str(y)] = g[rc].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def complete_since(nk, races):
    """netkeiba の払戻が「全レース」そろっている期間の始まり（新しい方から連続している範囲）"""
    from .pedigree import netkeiba_race_id
    r = races[(races["surf"] != "障") & (races["ymd"] >= "20250101")]
    have = pd.Series([netkeiba_race_id(x.ymd, x.jo, x.kai, x.nichi, x.rno) in nk for x in r.itertuples()], index=r.index)
    by = have.groupby(r["ymd"]).all().sort_index(ascending=False)
    since = None
    for ymd, ok in by.items():
        if not ok:
            break
        since = ymd
    return since


def final(log=print):
    from .data import load_results
    df, preds = bench.run(log=log)
    # 追加のアンサンブル
    def avg(*names, win=None):
        o = pd.DataFrame(index=preds[names[0]].index)
        o["p_top3"] = np.mean([preds[n]["p_top3"] for n in names], axis=0)
        o["p_win"] = np.mean([preds[n]["p_win"] for n in (win or names)], axis=0)
        return o
    preds.setdefault("ens_a", avg("gbm_all", "benter"))
    preds.setdefault("ens_b", avg("gbm_all", "logit", "benter"))
    preds.setdefault("ens_c", avg("gbm_all", "benter", win=["benter", "rank"]))
    pay = bench.pay_model_by_year(df)
    T = {}
    T["prob"] = E.prob_quality(preds, df)
    T["base"] = E.base_rates(E.attach(df, preds["mkt"], pay))
    grids, best = [], []
    for name, p in preds.items():
        d = E.attach(df, p, pay)
        g = E.policy_grid(d)
        g.insert(0, "方法", name)
        grids.append(g)
        c = E.choose(g)
        if len(c):
            top = c.iloc[0]
            s, rc = E.select(d, top["選び方"], top["対象"], top["閾値"])
            lo, hi = E.boot_ci(s[s["year"].isin(bench.TEST_YEARS)], rc)
            best.append({**top.to_dict(), "後半95%区間": "%.0f〜%.0f%%" % (100 * lo, 100 * hi)})
    T["grid"] = pd.concat(grids, ignore_index=True)
    T["best"] = pd.DataFrame(best).sort_values("後半回収", ascending=False)
    ms = ["mkt", "gbm6", "gbm_all", "gbm_small", "logit", "benter", "rank", "-speed", "-course", "-people", "ens",
          "ens_a", "ens_b", "ens_c"]
    T["q_chu_win"] = _quantile_table(df, preds, pay, ms, "中穴", "ev_win", "ret_win")
    T["q_oo_place"] = _quantile_table(df, preds, pay, ms, "大穴", "ev_place", "ret_place")
    T["q_fav_win"] = _quantile_table(df, preds, pay, ["mkt", "gbm_all", "benter", "ens_a"], "1〜5番人気", "ev_win", "ret_win")
    d6 = E.attach(df, preds["gbm6"], pay)
    T["thr_chu"] = _threshold_table(d6, "中穴", "ev_win", "ret_win", [1.1, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6])
    T["thr_oo"] = _threshold_table(d6, "大穴", "ev_place", "ret_place", [0.8, 0.9, 1.0, 1.1, 1.2])
    chu = d6[(d6["tier"] == "中穴") & (d6["ev_win"] >= CHU_T)]
    oo = d6[(d6["tier"] == "大穴") & (d6["ev_place"] >= OO_T)]
    T["seg_chu"] = E.segments(chu, "ret_win")
    T["stake_chu"] = {"後半": E.staking(chu[chu["year"].isin(bench.TEST_YEARS)], "ret_win", "p_win", "odds"),
                      "4年": E.staking(chu, "ret_win", "p_win", "odds")}
    # 券種：払戻が全レースそろっている期間だけ（偏りを避ける）
    races, _ = load_results()
    nk = load_nk_races()
    since = complete_since(nk, races) or "99999999"
    T["bet_since"] = since
    bets = {}
    for lab, sel in (("中穴★（単勝期待値%.2f以上）" % CHU_T, chu), ("大穴（複勝期待値%.1f以上）" % OO_T, oo),
                     ("中穴 全部", d6[d6["tier"] == "中穴"]), ("大穴 全部", d6[d6["tier"] == "大穴"])):
        s = sel[sel["ymd"] >= since]
        rmap = {r: nk_id(r) for r in s["race_id"].unique()}
        bets[lab] = E.bet_types(s, df[df["race_id"].isin(rmap)], nk, rmap)
    T["bets"] = bets
    return T


def _summary(T):
    """冒頭の「採用した運用」（数字は T から）"""
    thr = T["thr_chu"].set_index("閾値")
    r = thr.loc[CHU_T]
    yrs = "・".join("%d%%" % round(100 * r[y]) for y in ("2023", "2024", "2025", "2026") if y in r and r[y] == r[y])
    b = T["bets"]
    chu = next(v for k, v in b.items() if k.startswith("中穴★")).set_index("券種")
    oo = next(v for k, v in b.items() if k.startswith("大穴（")).set_index("券種")
    oo_best = oo["回収率"].idxmax()
    st = T["stake_chu"]["4年"]
    return ["## 採用した運用（この検証の結論）", "",
            "| | ルール | 根拠 |", "|---|---|---|",
            "| ★ 買い | 中穴の単勝期待値 %.2f 以上 → **単勝** | 年ごと %s。前半で閾値を選び、後半 %d%%（95%%区間 %s）。券種では単勝が最良（複勝 %d%%・馬連 %d%%・ワイド %d%%） |"
            % (CHU_T, yrs, round(100 * r["後半回収"]), r["後半95%区間"], round(100 * chu.loc["複勝", "回収率"]),
               round(100 * chu.loc["馬連 1〜3番人気へ（3点）", "回収率"]), round(100 * chu.loc["ワイド 1〜3番人気へ（3点）", "回収率"])),
            "| △ 参考 | 大穴の複勝期待値 %.1f 以上 | どの券種も100%%未満（最も損が小さいのは「%s」%d%%）。買いの印にはしない |"
            % (OO_T, oo_best, round(100 * oo.loc[oo_best, "回収率"])),
            "| 賭け金 | 均等 | ★を4年：均等 資金%.2f倍（最大下落%d%%）、ケリー %.2f倍（最大下落%d%%） |"
            % (st["均等"]["最終資金（初期1）"], round(100 * st["均等"]["最大下落"]), st["ケリー"]["最終資金（初期1）"],
               round(100 * st["ケリー"]["最大下落"])),
            "| 絞り込み | しない | 場・芝ダ・距離・クラス・頭数・馬場のどれも前半と後半で向きがそろわない（§10） |", ""]


CHU_T = 1.45   # 中穴：単勝期待値の閾値（前半2023〜24年の「回収率−標準誤差」が最大）
OO_T = 1.0     # 大穴：表示用（買いの基準ではない）


def write_final(T, path=None, log=print):
    path = path or REPORT_DIR / "research" / "benchmark.md"
    pf = {"3着内 市場超え度(%)": "%.2f", "勝ち 市場超え度(%)": "%.2f", "AUC(3着内)": "%.4f", "予想平均/実際": "%.3f"}
    yf = {k: "%.3f" for k in ("通算", "2023", "2024", "2025", "2026", "前半回収", "後半回収", "前半下限",
                              "y2023", "y2024", "y2025", "y2026", "回収率", "複勝回収", "単勝回収")}
    yf["閾値"] = "%.2f"
    L = ["# 予測方法・運用ルールの総当たり検証", "",
         "更新：%s。JRA平地2021年〜。**2023〜26年を毎年「その年より前のデータだけ」で学習して予想**（ウォークフォワード）。"
         "運用ルールは **前半（2023〜24年）だけで選び、後半（2025〜26年）で確かめる**。" % datetime.now().strftime("%Y-%m-%d"), "",
         "区分：中穴＝6番人気以下・単勝10〜30倍、大穴＝6番人気以下・単勝30倍以上。回収率は1点100円の平均（1.000＝100%）。", ""]
    L += _summary(T)
    L += ["## 1. 何もしない場合（全部買う）", "", _md(T["base"], yf), "",
          "## 2. 確率の精度（市場超え度＝対数損失が市場モデルより何%小さいか）", ""]
    for band in ("中穴", "大穴", "全頭"):
        t = T["prob"][T["prob"]["対象"] == band].drop(columns="対象").sort_values("3着内 市場超え度(%)", ascending=False)
        L += ["### " + band, "", _md(t, pf), ""]
    L += ["方法：mkt＝オッズだけ、gbm6＝現行（6番人気以下で学習）、gbm_all＝全頭で学習、gbm_tier＝区分別、gbm_small/big＝正則化の強弱、"
          "logit＝線形、benter＝能力モデル＋市場の2段階（条件付きロジット）、rank＝ランキング学習＋市場、recency＝直近重視、"
          "day＝当日の馬場傾向を追加、-speed 等＝その特徴群を抜いたもの、shrink＝市場へ縮小、ens＝gbm6とbenterの平均、"
          "ens_a/b/c＝gbm_all・logit・benter・rank の平均。", "",
          "## 3. 中穴・単勝：期待値の上位q%（年ごとの順位。閾値の調整が要らない比べ方）", "", _md(T["q_chu_win"], yf), "",
          "## 4. 大穴・複勝：期待値の上位q%", "", _md(T["q_oo_place"], yf), "",
          "## 5. 1〜5番人気・単勝：期待値の上位q%（穴に絞るのが最適かの確認）", "", _md(T["q_fav_win"], yf), "",
          "## 6. 採用：中穴の単勝期待値の閾値（現行モデル gbm6）", "",
          "前半の「回収率−標準誤差」が最大の閾値を採用（後半は見ずに決める）。", "", _md(T["thr_chu"], yf), "",
          "## 7. 大穴：複勝期待値の閾値（現行モデル）", "", _md(T["thr_oo"], yf), "",
          "## 8. 賭け金（中穴★、時系列で資金を動かした場合。均等＝毎回資金の0.5%、ケリー＝4分の1ケリー・上限2%）", ""]
    for k, v in T["stake_chu"].items():
        st = pd.DataFrame(v).T.rename_axis("賭け方")
        L += ["### " + k, "", _md(st, {"最終資金（初期1）": "%.3f", "最大下落": "%.3f", "回収率": "%.3f"}, index=True), ""]
    L += ["## 9. 券種（払戻が全レースそろっている %s 以降。相手は同じレースの人気上位）" % T["bet_since"], ""]
    for k, v in T["bets"].items():
        L += ["### " + k, "", _md(v, {"回収率": "%.3f"}), ""]
    L += ["## 10. 絞り込み（中穴★。前半と後半の両方で100%超えか）", "", _md(T["seg_chu"], yf), "",
          "## 11. 方法ごとの最良ルール（自動選択：前半の頭数400以上で「回収率−標準誤差」最大）", "",
          _md(T["best"][["方法", "選び方", "対象", "閾値", "券", "前半頭数", "前半回収", "後半頭数", "後半回収", "後半95%区間",
                         "y2023", "y2024", "y2025", "y2026"]], yf), ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    log("保存：%s" % path)
    return path
