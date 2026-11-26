"""1頭1行の特徴量（すべて「そのレースより前」に分かっていた情報だけで作る）

大きく4種類：
  1. 市場：単勝オッズ・人気・Harville式で直した「オッズから見た3着内確率」
  2. その馬の過去：前走・近走・同じ競馬場/コース/距離帯/馬場/回りでの好走歴・過去の激走歴（人気薄での好走）
  3. 臨戦過程：間隔・叩き何戦目・距離/芝ダ/クラスの変化・乗り替わり・ブリンカー初装着・斤量・馬体重
  4. 人と血統：騎手・調教師・生産者・父・母の父の「その日より前」の成績（人気薄の好走率）
"""
import warnings

import numpy as np
import pandas as pd

from .config import LONGSHOT_POP
from .data import CLS_RANK, breeder_key, dist_band, going_group

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

SMOOTH = 30.0   # 率の平滑化（件数が少ないときは全体の率に寄せる）


# ───────────────────────── 市場 ─────────────────────────
def harville_top3(p):
    """勝率ベクトル → 各馬の3着内確率（Harville）。p は合計1"""
    p = np.asarray(p, float)
    n = len(p)
    if n <= 3:
        return np.ones(n)
    q = np.clip(1.0 - p, 1e-9, None)
    # 2着：P(j 1着, i 2着) = p_j * p_i / (1 - p_j)
    m2 = np.outer(p / q, p)               # [j, i]
    np.fill_diagonal(m2, 0.0)
    second = m2.sum(axis=0)
    # 3着：P(j,k,i) = p_j * p_k/(1-p_j) * p_i/(1-p_j-p_k)
    third = np.zeros(n)
    for j in range(n):
        pk = p * p[j] / q[j]               # P(j 1着, k 2着)
        pk[j] = 0.0
        rest = np.clip(1.0 - p[j] - p, 1e-9, None)   # 1 - p_j - p_k（kごと）
        w = pk / rest                      # k ごとの重み
        w[j] = 0.0
        # i ≠ j, k
        tot = (w.sum() - w) * p            # Σ_{k≠i} w_k * p_i
        tot[j] = 0.0
        third += tot
    return np.clip(p + second + third, 0.0, 1.0)


def market_features(df):
    """df: 1レース分ずつ race_id でまとまった表。odds から勝率・3着内確率を付ける"""
    inv = 1.0 / df["odds"].clip(lower=1.0)
    s = inv.groupby(df["race_id"]).transform("sum")
    df["mkt_win"] = inv / s
    out = np.empty(len(df))
    for _, idx in df.groupby("race_id").indices.items():
        out[idx] = harville_top3(df["mkt_win"].values[idx])
    df["mkt_top3"] = out
    df["log_odds"] = np.log(df["odds"].clip(lower=1.0))
    df["pop_pct"] = df["pop"] / df["heads"]
    df["fav_odds"] = df.groupby("race_id")["odds"].transform("min")
    df["n_under10"] = df.groupby("race_id")["odds"].transform(lambda x: (x < 10).sum())
    df["overround_top"] = df.groupby("race_id")["mkt_win"].transform(lambda x: np.sort(x.values)[-3:].sum())
    return df


# ───────────────────────── 時点つき集計（人・血統） ─────────────────────────
def asof_rates(df, keys, name):
    """keys ごとに「その日より前」の 出走数・3着内率・人気薄（10番人気以下）の3着内率 を返す"""
    k = df[keys + ["ymd", "top3", "is_ls", "done"]].copy()
    k = k[k["done"] == 1]
    k["ls_top3"] = k["top3"] * k["is_ls"]
    day = k.groupby(keys + ["ymd"], sort=False).agg(n=("top3", "size"), t3=("top3", "sum"),
                                                      ln=("is_ls", "sum"), lt3=("ls_top3", "sum")).reset_index()
    day = day.sort_values(keys + ["ymd"])
    g = day.groupby(keys, sort=False)
    for c in ("n", "t3", "ln", "lt3"):
        day["c_" + c] = g[c].cumsum() - day[c]     # 当日を除いた累計
    # 予測したい日（当日に成績が無い行）にも、直前までの累計を付けるため merge_asof を使う
    day["ymd_i"] = day["ymd"].astype(int)
    day["tot_n"] = day["c_n"] + day["n"]
    day["tot_t3"] = day["c_t3"] + day["t3"]
    day["tot_ln"] = day["c_ln"] + day["ln"]
    day["tot_lt3"] = day["c_lt3"] + day["lt3"]
    left = df[keys + ["ymd"]].copy()
    left["ymd_i"] = left["ymd"].astype(int)
    left["_row"] = np.arange(len(df))
    left = left.sort_values("ymd_i")
    right = day[keys + ["ymd_i", "tot_n", "tot_t3", "tot_ln", "tot_lt3"]].sort_values("ymd_i")
    # 前日以前の最新の累計（当日を含めないよう、1を引いた日付で突き合わせる）
    left["ymd_q"] = left["ymd_i"] - 1
    m = pd.merge_asof(left.sort_values("ymd_q"), right.rename(columns={"ymd_i": "ymd_q"}),
                      on="ymd_q", by=keys, direction="backward")
    m = m.sort_values("_row")
    n = m["tot_n"].fillna(0).values
    t3 = m["tot_t3"].fillna(0).values
    ln = m["tot_ln"].fillna(0).values
    lt3 = m["tot_lt3"].fillna(0).values
    base_all = df.loc[df["done"] == 1, "top3"].mean()
    base_ls = df.loc[(df["done"] == 1) & (df["is_ls"] == 1), "top3"].mean()
    ls_t3 = (lt3 + SMOOTH * base_ls) / (ln + SMOOTH)
    return pd.DataFrame({
        name + "_n": n,
        name + "_t3": (t3 + SMOOTH * base_all) / (n + SMOOTH),
        name + "_ls_n": ln,
        name + "_ls_t3": ls_t3,
        name + "_ls_lift": ls_t3 / base_ls,   # 人気薄の好走率が全体の何倍か
    }, index=df.index)


# ───────────────────────── スピード指数・ペース ─────────────────────────
def speed_and_pace(df, win_t):
    """スピード指数 = (コース・馬場ごとの標準の勝ちタイム − 走破タイム) を1000mあたりに直したもの（秒）
    ペース = レースの前半3F − 後半3F をコースの中央値と比べたもの（マイナス = 前が速い = 先行馬に厳しい）"""
    cg = df["jo"] + df["surf"] + df["dist"].fillna(0).astype(int).astype(str)
    gg = cg + df["going"]
    w = pd.DataFrame({"race_id": win_t.index, "wt_": win_t.values})
    w["cg"] = w["race_id"].map(gg.groupby(df["race_id"]).first())
    std = w.groupby("cg")["wt_"].median()
    cnt = w.groupby("cg").size()
    std = std[cnt >= 5]
    df["std_time"] = gg.map(std)
    df["fig"] = ((df["std_time"] - df["tsec"]) / df["dist"] * 1000.0).clip(-6.0, 3.0)
    f3 = pd.to_numeric(df["first3"], errors="coerce")
    l3 = pd.to_numeric(df["last3"], errors="coerce")
    pace = f3 - l3
    med = pace.groupby(cg).transform("median")
    df["pace_dev"] = pace - med
    # 先行してハイペースに巻き込まれた度合い：前にいた（early_pctが小さい）× 前が速かった
    early = df[["c1", "c2", "c3", "c4"]].bfill(axis=1).iloc[:, 0] / df["heads"]
    df["front_cost"] = (0.5 - early).clip(lower=0) * (-df["pace_dev"]).clip(lower=0)
    return df


def race_relative(df):
    """同じレースの中での相対的な位置（市場が見落としやすい「メンバーの中での能力」）"""
    g = df.groupby("race_id")
    df["fig_rel"] = df["best3_fig"] - g["best3_fig"].transform("mean")
    df["fig_rank"] = g["best3_fig"].rank(ascending=False) / df["heads"]
    df["avgfig_rank"] = g["avg3_fig"].rank(ascending=False) / df["heads"]
    df["form_rank"] = g["avg3_place_pct"].rank(ascending=True) / df["heads"]
    df["early_rank"] = g["avg3_early"].rank(ascending=True) / df["heads"]   # メンバー内で何番目に前に行くか
    df["n_front"] = g["avg3_early"].transform(lambda x: (x <= 0.25).sum())  # 先行タイプの頭数（ペースの見込み）
    return df


def day_bias(df):
    """当日・同じ競馬場・同じ芝ダの、それより前のレースの傾向（前残り/差し、内/外、人気薄の好走）"""
    plc = df[(df["top3"] == 1)].copy()
    plc["early"] = plc[["c1", "c2", "c3", "c4"]].bfill(axis=1).iloc[:, 0] / plc["heads"]
    plc["ls"] = (plc["pop"] >= LONGSHOT_POP).astype(float)
    plc["num_pct_"] = plc["num"] / plc["heads"]
    pr = plc.groupby("race_id").agg(e_sum=("early", "sum"), e_n=("early", "count"), i_sum=("num_pct_", "sum"),
                                    ls=("ls", "sum"))
    races = df.drop_duplicates("race_id")[["race_id", "ymd", "jo", "surf", "rno"]].set_index("race_id")
    races = races.join(pr).fillna({"e_sum": 0, "e_n": 0, "i_sum": 0, "ls": 0}).sort_values(["ymd", "jo", "surf", "rno"])
    gk = races.groupby(["ymd", "jo", "surf"], sort=False)
    for c in ("e_sum", "e_n", "i_sum", "ls"):
        races["d_" + c] = gk[c].cumsum() - races[c]
    races["day_n"] = gk.cumcount()
    races["day_front"] = np.where(races["d_e_n"] > 0, races["d_e_sum"] / races["d_e_n"].clip(lower=1), np.nan)
    races["day_inner"] = np.where(races["d_e_n"] > 0, races["d_i_sum"] / races["d_e_n"].clip(lower=1), np.nan)
    races["day_ls"] = races["d_ls"]
    m = races[["day_n", "day_front", "day_inner", "day_ls"]]
    df = df.join(m, on="race_id")
    df["day_front_fit"] = (0.5 - df["avg3_early"]) * (0.5 - df["day_front"])
    df["day_inner_fit"] = (0.5 - df["num"] / df["heads"]) * (0.5 - df["day_inner"])
    return df


# ───────────────────────── その馬の過去 ─────────────────────────
def horse_history(df):
    df = df.sort_values(["horse", "ymd", "rno"]).copy()
    g = df.groupby("horse", sort=False)
    df["n_prev"] = g.cumcount()
    dt = pd.to_datetime(df["ymd"], format="%Y%m%d")
    df["_dt"] = dt
    prev_dt = g["_dt"].shift(1)
    df["interval"] = (dt - prev_dt).dt.days
    # 叩き何戦目（90日以上空いたら1戦目）
    lay = (df["interval"].isna() | (df["interval"] >= 90)).astype(int)
    seg = lay.groupby(df["horse"]).cumsum()
    df["since_layoff"] = df.groupby([df["horse"], seg]).cumcount() + 1
    df.loc[df["n_prev"] == 0, "since_layoff"] = 0
    # 前走の情報
    df["c_first"] = df[["c1", "c2", "c3", "c4"]].bfill(axis=1).iloc[:, 0]
    df["early_pct"] = df["c_first"] / df["heads"]
    df["c4_pct"] = df["c4"] / df["heads"]
    df["gain"] = (df["c4"] - df["place"]) / df["heads"]      # 4角→ゴールで上げた割合
    df["place_pct"] = df["place"] / df["heads"]
    df["beat_mkt"] = (df["pop"] - df["place"]) / df["heads"]  # 人気より上の着順なら＋
    shift_cols = {"place": "p1_place", "pop": "p1_pop", "odds": "p1_odds", "heads": "p1_heads", "dist": "p1_dist",
                  "surf": "p1_surf", "jo": "p1_jo", "cls_r": "p1_cls_r", "tdiff": "p1_tdiff", "early_pct": "p1_early",
                  "c4_pct": "p1_c4", "gain": "p1_gain", "wt": "p1_wt", "jockey": "p1_jockey", "place_pct": "p1_place_pct",
                  "beat_mkt": "p1_beat_mkt", "cond": "p1_cond", "fig": "p1_fig", "pace_dev": "p1_pace_dev",
                  "front_cost": "p1_front_cost"}
    for c, nm in shift_cols.items():
        df[nm] = g[c].shift(1)
    df["p2_place_pct"] = g["place_pct"].shift(2)
    df["p3_place_pct"] = g["place_pct"].shift(3)
    df["p2_tdiff"] = g["tdiff"].shift(2)
    # 近走の平均（直近3走・5走）
    for c, w, nm in (("place_pct", 3, "avg3_place_pct"), ("tdiff", 3, "avg3_tdiff"), ("early_pct", 3, "avg3_early"),
                     ("gain", 3, "avg3_gain"), ("place_pct", 5, "best5_place_pct"), ("beat_mkt", 5, "avg5_beat_mkt")):
        sh = g[c].shift(1)
        r = sh.groupby(df["horse"]).rolling(w, min_periods=1)
        df[nm] = (r.min() if nm.startswith("best") else r.mean()).reset_index(level=0, drop=True)
    shf = g["fig"].shift(1).groupby(df["horse"])
    df["best3_fig"] = shf.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    df["avg3_fig"] = shf.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["best5_fig"] = shf.rolling(5, min_periods=1).max().reset_index(level=0, drop=True)
    # 変化
    df["d_dist"] = df["dist"] - df["p1_dist"]
    df["surf_change"] = ((df["p1_surf"].notna()) & (df["surf"] != df["p1_surf"])).astype(float)
    df["d_cls"] = df["cls_r"] - df["p1_cls_r"]
    df["jockey_change"] = ((df["p1_jockey"].notna()) & (df["jockey"] != df["p1_jockey"])).astype(float)
    df["d_wt"] = df["wt"] - df["p1_wt"]
    df["same_jo_prev"] = (df["jo"] == df["p1_jo"]).astype(float)
    b = (df["blinker"] == "B").astype(int)
    prev_b = b.groupby(df["horse"]).cumsum() - b
    df["blinker_on"] = b.astype(float)
    df["blinker_first"] = ((b == 1) & (prev_b == 0) & (df["n_prev"] > 0)).astype(float)
    # 条件別の過去成績（同じ競馬場・コース・距離帯・馬場・回り）
    df["course"] = df["jo"] + df["surf"] + df["dist"].fillna(0).astype(int).astype(str)
    df["jo_surf"] = df["jo"] + df["surf"]
    df["surf_band"] = df["surf"] + df["band"]
    df["surf_going"] = df["surf"] + df["going"]
    df["surf_turn"] = df["surf"] + df["turn"].fillna("")
    df["_t3f"] = df["top3"].fillna(0.0)
    df["ls_top3_flag"] = ((df["pop"] >= 8) & (df["top3"] == 1)).astype(float)  # 人気薄（8番人気以下）で3着内
    for key, nm in (("horse", "all"), ("jo", "jo"), ("jo_surf", "josurf"), ("course", "course"),
                    ("surf_band", "band"), ("surf_going", "going"), ("surf_turn", "turn")):
        keys = ["horse"] if key == "horse" else ["horse", key]
        gg = df.groupby(keys, sort=False)
        n = gg.cumcount()
        t3 = df["_t3f"].groupby([df[k] for k in keys], sort=False).cumsum() - df["_t3f"]
        lsh = gg["ls_top3_flag"].cumsum() - df["ls_top3_flag"]
        df["h_%s_n" % nm] = n
        df["h_%s_t3" % nm] = t3
        df["h_%s_t3r" % nm] = np.where(n > 0, t3 / n.clip(lower=1), np.nan)
        df["h_%s_lshit" % nm] = lsh
    # 条件の「得意度」= その条件での3着内率 − 全体の3着内率
    for nm in ("jo", "josurf", "course", "band", "going", "turn"):
        df["h_%s_edge" % nm] = df["h_%s_t3r" % nm] - df["h_all_t3r"]
    return df.drop(columns=["_dt", "_t3f"]).copy()


# ───────────────────────── まとめ ─────────────────────────
def prepare(runs, horses=None):
    """成績表（＋予測したいレースの行。place が空）→ 特徴量つきの表"""
    df = runs.copy()
    df = df[(df["surf"] != "障") & (~df["status"].isin(["取消", "除外"]))].copy()
    df["race_id"] = df["ymd"] + df["jo"] + df["kai"].astype(int).map("{:02d}".format) + \
        df["nichi"].astype(int).map("{:02d}".format) + df["rno"].astype(int).map("{:02d}".format)
    df["done"] = df["place"].notna().astype(int) | (df["status"] == "中止").astype(int)
    df["top3"] = np.where(df["done"] == 1, (df["place"] <= 3).astype(float), np.nan)
    df.loc[df["status"] == "中止", "top3"] = 0.0
    df["win"] = np.where(df["done"] == 1, (df["place"] == 1).astype(float), np.nan)
    df["heads"] = df.groupby("race_id")["horse"].transform("size").astype(float)
    df["cls_r"] = df["cls"].map(CLS_RANK).astype(float)
    df["band"] = df["dist"].map(dist_band)
    df["going"] = df["cond"].map(going_group).fillna("")
    win_t = df[df["place"] == 1].groupby("race_id")["tsec"].min()
    df["tdiff"] = (df["tsec"] - df["race_id"].map(win_t)).clip(upper=5.0)
    df = speed_and_pace(df, win_t)
    df = market_features(df)
    df["is_ls"] = (df["pop"] >= LONGSHOT_POP).astype(int)
    df = horse_history(df)
    df = race_relative(df)
    df = day_bias(df)
    # 人・血統
    if horses is not None and len(horses):
        hmap = pd.DataFrame(horses.values()) if isinstance(horses, dict) else horses
        hmap = hmap[["horse", "sire", "bms"]].drop_duplicates("horse")
        df = df.drop(columns=[c for c in ("sire", "bms") if c in df.columns]).merge(hmap, on="horse", how="left")
    for c in ("sire", "bms"):
        if c not in df.columns:
            df[c] = np.nan
    df["breeder_k"] = df["breeder"].map(breeder_key)
    df["trainer_surf"] = df["trainer"].fillna("") + df["surf"]
    df["jockey_jo"] = df["jockey"].fillna("") + df["jo"]
    df["sire_surf_band"] = df["sire"].fillna("?") + df["surf_band"]
    df["sire_course"] = df["sire"].fillna("?") + df["course"]
    df["bms_surf"] = df["bms"].fillna("?") + df["surf"]
    parts = [df]
    for keys, nm in ((["jockey"], "jk"), (["trainer"], "tr"), (["trainer_surf"], "trs"), (["jockey_jo"], "jkjo"),
                     (["breeder_k"], "br"), (["sire_surf_band"], "sire_sb"), (["sire_course"], "sire_c"),
                     (["bms_surf"], "bms_s")):
        part = asof_rates(df, keys, nm)
        if nm.startswith(("sire", "bms")):
            part.loc[df["sire" if nm.startswith("sire") else "bms"].isna()] = np.nan
        parts.append(part)
    extra = pd.DataFrame({
        "month": df["ymd"].str[4:6].astype(int),
        "sex_c": df["sex"].map({"牡": 0, "牝": 1, "セ": 2}),
        "num_pct": df["num"] / df["heads"],
        "year": df["ymd"].str[:4].astype(int),
    }, index=df.index)
    df = pd.concat(parts + [extra], axis=1)
    # 騎手の格上げ：今回の騎手の3着内率 − 前走の騎手の3着内率（どちらもその時点の値）
    df = df.sort_values(["horse", "ymd", "rno"])
    df["jk_up"] = df["jk_t3"] - df.groupby("horse", sort=False)["jk_t3"].shift(1)
    df["jk_rank"] = df.groupby("race_id")["jk_t3"].rank(ascending=False) / df["heads"]
    df["tr_rank"] = df.groupby("race_id")["tr_t3"].rank(ascending=False) / df["heads"]
    return df.sort_values(["ymd", "race_id", "num"]).reset_index(drop=True)


NUMERIC = [
    # 市場
    "log_odds", "pop", "pop_pct", "mkt_win", "mkt_top3", "fav_odds", "n_under10", "overround_top", "heads",
    # レース
    "dist", "cls_r", "month",
    # 馬
    "age", "sex_c", "wt", "hw", "hwd", "num_pct", "waku",
    "n_prev", "interval", "since_layoff",
    "p1_place", "p1_pop", "p1_odds", "p1_place_pct", "p1_tdiff", "p1_early", "p1_c4", "p1_gain", "p1_beat_mkt",
    "p2_place_pct", "p3_place_pct", "p2_tdiff",
    "avg3_place_pct", "avg3_tdiff", "avg3_early", "avg3_gain", "best5_place_pct", "avg5_beat_mkt",
    "d_dist", "surf_change", "d_cls", "jockey_change", "d_wt", "same_jo_prev", "blinker_on", "blinker_first",
    # スピード指数・ペース・メンバー内の相対位置・当日の傾向
    "p1_fig", "best3_fig", "avg3_fig", "best5_fig", "p1_pace_dev", "p1_front_cost",
    "fig_rel", "fig_rank", "avgfig_rank", "form_rank", "early_rank", "n_front",
    "day_n", "day_front", "day_inner", "day_ls", "day_front_fit", "day_inner_fit",
    "h_all_n", "h_all_t3r", "h_all_lshit",
    "h_jo_n", "h_jo_t3", "h_jo_t3r", "h_jo_lshit", "h_jo_edge",
    "h_josurf_n", "h_josurf_t3r", "h_josurf_lshit", "h_josurf_edge",
    "h_course_n", "h_course_t3", "h_course_t3r", "h_course_lshit", "h_course_edge",
    "h_band_n", "h_band_t3r", "h_band_edge", "h_going_n", "h_going_t3r", "h_going_edge",
    "h_turn_n", "h_turn_t3r", "h_turn_edge",
    # 人・血統（当日より前）
    "jk_n", "jk_t3", "jk_ls_t3", "jk_ls_lift", "tr_n", "tr_t3", "tr_ls_t3", "tr_ls_lift", "jk_up", "jk_rank", "tr_rank",
    "trs_ls_t3", "trs_ls_lift", "jkjo_ls_t3", "jkjo_ls_lift", "br_ls_t3", "br_ls_lift",
    "sire_sb_n", "sire_sb_t3", "sire_sb_ls_t3", "sire_sb_ls_lift", "sire_c_n", "sire_c_t3", "sire_c_ls_lift",
    "bms_s_t3", "bms_s_ls_lift",
]
CATEGORICAL = ["jo", "surf"]
