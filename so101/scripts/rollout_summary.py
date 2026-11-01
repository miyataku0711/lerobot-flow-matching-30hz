"""実機の rollout の記録（`--strategy.timestamp_log_dir` の parquet と `--inference.log_dir` の CSV）をまとめる。

    python so101/scripts/rollout_summary.py <記録のディレクトリ>... [--json summary.json]

ディレクトリには timestamp_logs の parquet と、async_chunk ならエンジンの CSV（async_ticks / async_chunks）がある。
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def rev(d, thr):
    return ((np.sign(d[1:]) * np.sign(d[:-1]) < 0) & (np.abs(d[1:]) > thr) & (np.abs(d[:-1]) > thr)).any(1)


def summarize(d: Path) -> dict:
    p = pd.read_parquet(sorted(glob.glob(f"{d}/*.parquet"))[-1])
    loop = np.diff(p.loop_start_timestamp.to_numpy()) * 1e3
    ctrl = (p.loop_control_end_timestamp - p.loop_start_timestamp).to_numpy() * 1e3
    r = {
        "記録した周期": len(p),
        "実効 fps": (len(p) - 1) / (p.loop_start_timestamp.iloc[-1] - p.loop_start_timestamp.iloc[0]),
        "周期 中央/p99/最大 ms": f"{np.median(loop):.2f}/{np.percentile(loop, 99):.2f}/{loop.max():.1f}",
        "33.4ms 超 / 35ms 超": f"{int((loop > 33.4).sum())} / {int((loop > 35).sum())}",
        "制御部 中央/p99 ms": f"{np.median(ctrl):.2f}/{np.percentile(ctrl, 99):.2f}",
    }
    ticks = sorted(glob.glob(f"{d}/async_ticks*.csv"))
    if not ticks:
        return r
    t = pd.read_csv(ticks[-1])
    c = pd.read_csv(sorted(glob.glob(f"{d}/async_chunks*.csv"))[-1])
    t, c = t[t.epoch == t.epoch.max()], c[c.epoch == c.epoch.max()]
    a = t[[f"a{i}" for i in range(6)]].to_numpy()
    dd = np.diff(a, axis=0)
    j = np.abs(dd).max(1)
    sw = np.diff(t.chunk_id.to_numpy()) != 0
    tk = t.tick.to_numpy()
    first = (t.t.to_numpy()[2:] - t.t.to_numpy()[0]) < 2.0
    r.update({
        "保持": int(t.held.sum()),
        "番号 重なり/飛び": f"{int((np.diff(tk) == 0).sum())}/{int((np.diff(tk) > 1).sum())}",
        "chunk 数 / 誘導": f"{len(c)} / {int(c.guided.sum())}",
        "計算 中央/最大 ms": f"{c.latency_ms.median():.1f}/{c.latency_ms.max():.1f}",
        "遅れ（周期）": "/".join(str(x) for x in sorted(c.delay_ticks.unique())),
        "行き戻り 0.3/1/2 度 %": "/".join(f"{rev(dd, x).mean() * 100:.1f}" for x in (0.3, 1.0, 2.0)),
        "行き戻り 0.3 度 最初2秒/以降 %": f"{rev(dd, 0.3)[first].mean() * 100:.0f}/{rev(dd, 0.3)[~first].mean() * 100:.0f}",
        "chunk 内の跳び p99/最大": f"{np.percentile(j[~sw], 99):.2f}/{j[~sw].max():.2f}",
        "切替の跳び 中央/p99/最大": f"{np.median(j[sw]):.2f}/{np.percentile(j[sw], 99):.2f}/{j[sw].max():.2f}",
        "隣どうしの差の平均": round(float(np.abs(dd).mean()), 3),
    })
    return r


args = sys.argv[1:]
out_json = None
if "--json" in args:
    i = args.index("--json")
    out_json = args[i + 1]
    args = args[:i] + args[i + 2:]
out = {}
for d in map(Path, args):
    out[d.name] = summarize(d)
    print("==", d.name)
    for k, x in out[d.name].items():
        print(f"  {k}: {x:.3f}" if isinstance(x, float) else f"  {k}: {x}")
if out_json:
    json.dump(out, open(out_json, "w"), ensure_ascii=False, indent=1, default=float)
