"""実機の指令の「後戻り」（軌跡を少し戻ってまた進む）を数え、chunk の切り替えとの重なりを見る。

    python so101/scripts/backtrack.py <記録のディレクトリ>...

後戻り = ある関節が、前後それぞれ数周期の向き（W 周期で `THR` 度以上進む）と逆に、`THR` 度以上戻る区間。
1周期ごとの細かな行き戻り（rollout_summary.py の「行き戻り」）より長い、数周期にわたる戻りを見る。
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
W, THR = 4, 1.0  # 前後の向きを見る周期の数、戻り・進みとみなす量（度）


def backtracks(a: np.ndarray) -> list[tuple[int, int, float]]:
    """(周期, 関節, 戻った量) の一覧。周期 k から k+W の動きが、前（k-W..k）と後（k+W..k+2W）の動きと逆向き。"""
    out = []
    for j in range(a.shape[1] - 1):  # gripper は除く
        x = a[:, j]
        for k in range(W, len(x) - 2 * W):
            before, mid, after = x[k] - x[k - W], x[k + W] - x[k], x[k + 2 * W] - x[k + W]
            if abs(before) >= THR and abs(after) >= THR and np.sign(before) == np.sign(after) == -np.sign(mid) \
                    and abs(mid) >= THR:
                if not out or out[-1][1] != j or k - out[-1][0] > W:
                    out.append((k, j, float(mid)))
    return out


if __name__ == "__main__":
    for d in map(Path, sys.argv[1:]):
        t = pd.read_csv(sorted(glob.glob(f"{d}/async_ticks*.csv"))[-1])
        c = pd.read_csv(sorted(glob.glob(f"{d}/async_chunks*.csv"))[-1])
        t, c = t[t.epoch == t.epoch.max()].reset_index(drop=True), c[c.epoch == c.epoch.max()]
        a = t[[f"a{i}" for i in range(6)]].to_numpy()
        sw = np.where(np.diff(t.chunk_id.to_numpy()) != 0)[0] + 1  # 新しい chunk に切り替わった周期
        bt = backtracks(a)
        near = [b for b in bt if np.min(np.abs(sw - b[0] - W // 2)) <= W] if len(sw) else []
        secs = len(t) / 30
        print(f"== {d.name}（{secs:.0f} 秒、chunk の切り替え {len(sw)} 回、遅れ {sorted(c.delay_ticks.unique())}）")
        print(f"  後戻り（{W} 周期で {THR} 度以上戻ってまた進む）: {len(bt)} 回（1 分あたり {len(bt) / secs * 60:.1f}）、"
              f"うち切り替えの前後 {W} 周期以内 {len(near)} 回")
        if bt:
            per = {n: sum(1 for b in bt if b[1] == i) for i, n in enumerate(NAMES[:-1])}
            print("  関節別:", per, " 戻った量 中央 / 最大:",
                  f"{np.median([abs(b[2]) for b in bt]):.2f} / {max(abs(b[2]) for b in bt):.2f} 度")
            print("  例（周期, 関節, 戻った量）:", [(k, NAMES[j], round(m, 2)) for k, j, m in bt[:8]])
        jumps = np.abs(np.diff(a[:, :5], axis=0)).max(1)
        big = np.where(jumps > 4.0)[0] + 1
        big_sw = [k for k in big if np.min(np.abs(sw - k)) <= 1] if len(sw) else []
        print(f"  1 周期で 4 度を超える跳び: {len(big)} 回、うち切り替えの周期（±1）{len(big_sw)} 回")
