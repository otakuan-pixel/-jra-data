"""python -m anaba <コマンド>

  live      発走30分前レポート（Actions から10分おき。--race 05_11 で指定レースを今すぐ）
  prefetch  その日の出走馬の5代血統を先に取っておく
  replay    過去のレースで再現（その日より前のデータだけで学習し直す）
  train     学習一式（複勝払戻モデル・条件表・検証・本番モデル）
  horses    jra2 の出馬表の履歴から 馬名→馬ID・父・母父 の索引を更新
  backfill  過去レースの血統を netkeiba から少しずつ集める（症例対照）
  pedstudy  血統（奇跡の血量など）の検証
  settle    レポートで挙げた馬の答え合わせ
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="anaba", description="大穴（10番人気以下）分析システム")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("live")
    b.add_argument("--date")
    b.add_argument("--race", action="append", help="例 05_11（複数可）")
    b.add_argument("--lo", type=int, default=25)
    b.add_argument("--hi", type=int, default=35)
    b.add_argument("--no-notify", action="store_true")
    b = sub.add_parser("prefetch")
    b.add_argument("--date")
    b.add_argument("--max", type=int, default=800)
    b = sub.add_parser("replay")
    b.add_argument("--date", required=True, help="YYYYMMDD（カンマ区切りで複数日。--race を省くとその日の全レース）")
    b.add_argument("--race")
    b = sub.add_parser("train")
    b.add_argument("--no-validate", action="store_true")
    sub.add_parser("horses")
    b = sub.add_parser("backfill")
    b.add_argument("--since", default="20230101")
    b.add_argument("--until", default="99999999")
    b.add_argument("--controls", type=int, default=2)
    b.add_argument("--max-requests", type=int, default=3000)
    b = sub.add_parser("pedstudy")
    b.add_argument("--min-n", type=int, default=30)
    b.add_argument("--boot", type=int, default=400)
    sub.add_parser("settle")
    a = ap.parse_args(argv)

    if a.cmd == "live":
        from .runner import run_live
        run_live(a.date, lo=a.lo, hi=a.hi, only=a.race, notify=not a.no_notify)
    elif a.cmd == "prefetch":
        from .live import Http, now_jst
        from .runner import _load_state, _save_state, ensure_day
        ymd = a.date or now_jst().strftime("%Y%m%d")
        st = _load_state(ymd)
        st = ensure_day(Http(), ymd, st, max_ped=a.max)
        _save_state(ymd, st)
    elif a.cmd == "replay":
        from .runner import replay, replay_days
        if a.race:
            md, _ = replay(a.date, a.race)
            print(md)
        else:
            replay_days(a.date.split(","))
    elif a.cmd == "train":
        from .train import train
        train(do_validate=not a.no_validate)
    elif a.cmd == "horses":
        from .horses import build
        print("出馬表 %d 件 → 馬 %d 頭" % build())
    elif a.cmd == "backfill":
        from .backfill import run
        run(since=a.since, until=a.until, controls=a.controls, max_requests=a.max_requests)
    elif a.cmd == "pedstudy":
        from .pedstudy import study
        study(min_n=a.min_n, boot=a.boot)
    elif a.cmd == "settle":
        from .settle import settle
        settle()
    return 0


if __name__ == "__main__":
    sys.exit(main())
