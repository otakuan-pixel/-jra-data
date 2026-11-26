"""中穴・大穴の好走条件の洗い出し（区分は config.tier_of：6番人気以下のうち 10〜30倍＝中穴、30倍以上＝大穴）

好走した馬だけを眺めると「よくある条件」を拾ってしまう（生存者バイアス）。そこで、
  - 同じ区分（中穴／大穴）の全馬の中で、その条件に当てはまる馬の3着内率を、当てはまらない馬と比べる（リフト）
  - オッズから見て期待される3着内率と比べる（実績/期待 = A/E。1.0超えなら市場が過小評価）
  - 単勝・複勝の回収率（複勝は実測＋推定払戻）
  - 前半（2021〜23年）と後半（2024〜26年）の両方で同じ向きか（偶然の当たりを除くため）
を1つの表にする。
"""
import numpy as np
import pandas as pd

from .config import tier_series


def expected_top3(df):
    """オッズ帯 × 頭数 ごとの実際の3着内率（全期間）= 市場から見た期待値（Harvilleの偏りを補正済み）"""
    d = df[df["done"] == 1]
    ob = np.floor(np.log(df["odds"].clip(lower=1.0)) / 0.15)
    hb = pd.cut(df["heads"], [0, 11, 14, 16, 18], labels=False)
    key = ob.astype(int).astype(str) + "_" + hb.astype(str)
    rate = d.groupby(key[d.index])["top3"].mean()
    cnt = d.groupby(key[d.index]).size()
    rate = rate[cnt >= 30]
    e = key.map(rate)
    # 件数が少ない帯は Harville の値で埋める（極端な大穴）
    return e.fillna(df["mkt_top3"] * 0.8)


def factors(df):
    """名前 → 真偽の Series（すべて発走前に分かること）"""
    f = {}
    ge = lambda c, v: df[c].fillna(-1e9) >= v
    le = lambda c, v: df[c].fillna(1e9) <= v
    # ── 過去に好走したことのある条件 ──
    f["同コース（場・芝ダ・距離）で3着内あり"] = ge("h_course_t3", 1)
    f["同競馬場で3着内あり"] = ge("h_jo_t3", 1)
    f["同競馬場で3着内2回以上"] = ge("h_jo_t3", 2)
    f["同場・同芝ダの3着内率が全体より高い"] = ge("h_josurf_edge", 0.10) & ge("h_josurf_n", 2)
    f["同距離帯の3着内率が全体より高い"] = ge("h_band_edge", 0.10) & ge("h_band_n", 2)
    f["今回道悪 × 道悪で3着内率が高い"] = (df["going"] == "道悪") & ge("h_going_edge", 0.10) & ge("h_going_n", 2)
    f["同じ回り（右/左）の3着内率が高い"] = ge("h_turn_edge", 0.10) & ge("h_turn_n", 2)
    f["過去に8番人気以下で3着内（激走歴）"] = ge("h_all_lshit", 1)
    f["激走歴2回以上"] = ge("h_all_lshit", 2)
    f["同競馬場で激走歴"] = ge("h_jo_lshit", 1)
    f["同コースで激走歴"] = ge("h_course_lshit", 1)
    # ── 近走 ──
    f["前走3着内"] = le("p1_place", 3)
    f["前走 勝ち馬から0.5秒以内"] = le("p1_tdiff", 0.5)
    f["前走 1秒以内"] = le("p1_tdiff", 1.0)
    f["前走 人気より大きく好走（頭数の25%以上）"] = ge("p1_beat_mkt", 0.25)
    f["前走 4角から大きく追い上げ（頭数の25%以上）"] = ge("p1_gain", 0.25)
    f["前走 2秒以上の大敗"] = ge("p1_tdiff", 2.0)
    f["近3走の平均着順が上位40%以内"] = le("avg3_place_pct", 0.4)
    f["近5走に上位20%の着順あり"] = le("best5_place_pct", 0.2)
    f["先行タイプ（近3走の最初のコーナーが前20%）"] = le("avg3_early", 0.2)
    f["追い込みタイプ（近3走の最初のコーナーが後ろ30%）"] = ge("avg3_early", 0.7)
    # ── 臨戦過程（ローテーション）──
    f["休み明け初戦（90日以上）"] = (df["since_layoff"] == 1)
    f["叩き2戦目"] = (df["since_layoff"] == 2)
    f["叩き3戦目"] = (df["since_layoff"] == 3)
    f["中1週以内（連闘含む）"] = le("interval", 14) & ge("interval", 1)
    f["距離短縮（200m以上）"] = le("d_dist", -200)
    f["距離延長（200m以上）"] = ge("d_dist", 200)
    f["芝→ダート替わり"] = (df["surf_change"] == 1) & (df["surf"] == "ダ")
    f["ダート→芝替わり"] = (df["surf_change"] == 1) & (df["surf"] == "芝")
    f["クラス降級（前走より下）"] = le("d_cls", -1)
    f["昇級初戦"] = ge("d_cls", 1)
    f["乗り替わり"] = df["jockey_change"] == 1
    f["ブリンカー初装着"] = df["blinker_first"] == 1
    f["ブリンカー着用"] = df["blinker_on"] == 1
    f["斤量2kg以上減"] = le("d_wt", -2)
    f["馬体重+10kg以上"] = ge("hwd", 10)
    f["馬体重-10kg以下"] = le("hwd", -10)
    # ── 人（その日より前の成績）──
    f["騎手：人気薄の好走率が平均の1.3倍以上"] = ge("jk_ls_lift", 1.3)
    f["調教師：人気薄の好走率が平均の1.3倍以上"] = ge("tr_ls_lift", 1.3)
    f["調教師×芝ダ：人気薄の好走率が1.3倍以上"] = ge("trs_ls_lift", 1.3)
    f["騎手×競馬場：人気薄の好走率が1.3倍以上"] = ge("jkjo_ls_lift", 1.3)
    f["生産者：人気薄の好走率が1.3倍以上"] = ge("br_ls_lift", 1.3)
    f["騎手：3着内率25%以上（上位騎手）"] = ge("jk_t3", 0.25)
    # ── 馬・枠・レース ──
    f["内枠（馬番が前25%）"] = le("num_pct", 0.25)
    f["外枠（馬番が後ろ25%）"] = ge("num_pct", 0.76)
    f["牝馬"] = df["sex"] == "牝"
    f["3歳"] = df["age"] == 3
    f["7歳以上"] = ge("age", 7)
    f["16頭以上"] = ge("heads", 16)
    f["道悪（重・不良）"] = df["going"] == "道悪"
    f["ダート"] = df["surf"] == "ダ"
    f["新馬・未勝利"] = le("cls_r", 1)
    f["2〜3勝クラス"] = df["cls_r"].isin([3, 4])
    f["オープン・重賞"] = ge("cls_r", 5)
    f["キャリア3戦以内"] = le("n_prev", 3) & ge("n_prev", 1)
    return {k: v.fillna(False).astype(bool) for k, v in f.items()}


def _stats(d):
    n = len(d)
    if n == 0:
        return dict(n=0, top3=np.nan, ae=np.nan, win_roi=np.nan, place_roi=np.nan)
    return dict(n=n, top3=d["top3"].mean(), ae=d["top3"].sum() / d["exp_top3"].sum(),
                win_roi=(d["win"] * d["odds"]).mean(), place_roi=d["place_pay"].mean() / 100.0)


def _tier(df, tier):
    return df[(df["done"] == 1) & (tier_series(df) == tier)]


def factor_table(df, tier="大穴", split="2024"):
    d = _tier(df, tier).copy()
    base = _stats(d)
    fs = factors(d)
    rows = []
    a = d["ymd"] < split + "0101"
    for name, m in fs.items():
        s_in, s_out = _stats(d[m]), _stats(d[~m])
        sa, sb = _stats(d[m & a]), _stats(d[m & ~a])
        rows.append({"条件": name, "該当数": s_in["n"], "該当率": m.mean(),
                     "3着内率": s_in["top3"], "非該当の3着内率": s_out["top3"],
                     "リフト": s_in["top3"] / s_out["top3"] if s_out["top3"] else np.nan,
                     "A/E": s_in["ae"], "A/E前半": sa["ae"], "A/E後半": sb["ae"],
                     "単勝回収": s_in["win_roi"], "複勝回収": s_in["place_roi"],
                     "複勝前半": sa["place_roi"], "複勝後半": sb["place_roi"]})
    t = pd.DataFrame(rows)
    t["安定"] = ((t["A/E前半"] > 1.05) & (t["A/E後半"] > 1.05)) | ((t["A/E前半"] < 0.95) & (t["A/E後半"] < 0.95))
    return base, t.sort_values("A/E", ascending=False).reset_index(drop=True)


def profile_table(df, tier="大穴"):
    """好走馬（3着内）と凡走馬の平均の比較（標準化差）"""
    d = _tier(df, tier)
    cols = {"log_odds": "単勝オッズ(log)", "p1_place_pct": "前走着順(頭数比)", "p1_tdiff": "前走 勝ち馬との差(秒)",
            "avg3_place_pct": "近3走の平均着順(頭数比)", "avg3_early": "近3走の位置取り(前ほど小)",
            "h_all_t3r": "通算3着内率", "h_course_t3r": "同コース3着内率", "h_jo_t3r": "同競馬場3着内率",
            "h_all_lshit": "激走歴の回数", "interval": "間隔(日)", "n_prev": "キャリア", "age": "年齢",
            "jk_t3": "騎手の3着内率", "tr_t3": "調教師の3着内率", "jk_ls_lift": "騎手の穴好走率(倍)",
            "tr_ls_lift": "調教師の穴好走率(倍)", "num_pct": "馬番(頭数比)", "hwd": "馬体重増減"}
    rows = []
    for c, nm in cols.items():
        x1, x0 = d.loc[d["top3"] == 1, c], d.loc[d["top3"] == 0, c]
        sd = d[c].std()
        rows.append({"項目": nm, "好走馬の平均": x1.mean(), "凡走馬の平均": x0.mean(),
                     "差（標準偏差単位）": (x1.mean() - x0.mean()) / sd if sd else np.nan})
    return pd.DataFrame(rows).sort_values("差（標準偏差単位）", key=lambda s: -s.abs()).reset_index(drop=True)


def course_table(df, tier="大穴", min_n=300):
    """コース（場・芝ダ・距離）ごとの中穴・大穴の出やすさ"""
    d = _tier(df, tier)
    g = d.groupby(["jo", "surf", "dist"])
    t = g.apply(lambda x: pd.Series(_stats(x)), include_groups=False).reset_index()
    t = t[t["n"] >= min_n]
    return t.sort_values("ae", ascending=False).reset_index(drop=True)
