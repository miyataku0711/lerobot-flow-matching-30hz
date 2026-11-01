"""成功率の記録表（CSV）を方策ごとに集計する。

    python so101/scripts/success_tally.py trials.csv

CSV の列（1 行が 1 試行）:
    trial, policy, color, position, placed, returned_home, regrasps, empty_reach, notes
    - placed:        ブロックが正しい箱に入ったら 1（赤は黄の箱、青は透明の箱）
    - returned_home: 置いたあと home の姿勢に戻ったら 1
    - regrasps:      つかみ直した回数
    - empty_reach:   home に戻ったあと、空の机へ手を伸ばしかけたら 1
成功率には Wilson の 95% 信頼区間を付ける（10 回程度では幅が広いことを示すため）。
"""

from __future__ import annotations

import argparse
import math

import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    args = ap.parse_args()
    try:
        df = pd.read_csv(args.csv)
    except UnicodeDecodeError:  # Excel（日本語版）で保存すると Shift-JIS になる
        df = pd.read_csv(args.csv, encoding="cp932")
    rows = []
    for policy, g in df.groupby("policy", sort=False):
        n, k = len(g), int(g.placed.sum())
        lo, hi = wilson(k, n)
        rows.append({
            "policy": policy,
            "成功": f"{k}/{n}",
            "成功率": f"{k / n * 100:.0f}%",
            "95%区間": f"{lo * 100:.0f}–{hi * 100:.0f}%",
            "home に戻った": f"{int(g.returned_home.sum())}/{n}",
            "つかみ直し(平均)": f"{g.regrasps.mean():.1f}",
            "空振り": f"{int(g.empty_reach.sum())}/{n}",
            "赤・青の成功": "・".join(
                f"{n} {int(g[g.color == c].placed.sum())}/{int((g.color == c).sum())}"
                for c, n in (("red", "赤"), ("blue", "青"))
            ),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
