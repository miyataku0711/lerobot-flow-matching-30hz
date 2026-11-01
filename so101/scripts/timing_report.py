"""timestamp_logs から制御周期とカメラのフレーム間隔を集計する。

lerobot の独自改造（`utils/timestamp_logger.py`）が収録ごとに
`<dataset>/timestamp_logs/episode_NNNNN.parquet` を書く。1行が制御ループの1周で、
時刻はすべて同じプロセスの `time.perf_counter()`（秒）なので差を取れば比べられる。

1ファイルにはエピソード本体の後ろにリセット中の周回も入っている。
本体は `phase` 列があればそれで、無ければ `meta/episodes` の length ぶんの先頭行とみなす。

列が増えても古いデータセットを読めるよう、無い列の指標は出さない（None にする）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTAMP_LOGS_DIR = "timestamp_logs"
TS = "_timestamp"

# カメラではない時刻列（`<名前>_timestamp` の <名前> 部分）
LOOP_KEYS = {
    "loop_start",
    "loop_control_end",
    "loop_end",
    "observation",
    "observation_end",  # モーター読み取りの終了。local/timing 以降
    "action",
    "target_start",  # 絶対時刻スケジューラが予定した周回の開始時刻。local/timing 以降
}
# カメラ層が付ける補助の時刻（local/timing 以降。どれも perf_counter の時計に揃えてある）
#   `<cam>_arrival_timestamp`: 読み取りスレッドがフレームを受け取った時刻（後処理の前）
#   `<cam>_device_timestamp`:  RealSense はセンサ側の時刻、OpenCV(V4L2) はドライバのバッファ時刻
ARRIVAL_SUFFIX = "_arrival"
DEVICE_SUFFIX = "_device"

# カメラ間隔の判定。周期に対する比で決めるので fps が違っても同じ意味になる
DUP_RATIO = 0.15  # これ未満の間隔 = 同じフレームを2回読んだ（30fps で 5ms）
GAP_RATIO = 1.5  # これ超の間隔 = 途中のフレームを読み飛ばした（30fps で 50ms）
LATE_RATIO = 1.2  # これ超の周期 = 予算超え（30fps で 40ms）

EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.parquet$")


@dataclass
class Segment:
    """1エピソードの本体部分。"""

    episode_index: int
    df: pd.DataFrame
    total_rows: int  # リセットを含むファイルの行数


@dataclass
class Report:
    path: Path
    fps: float
    episodes: list[Segment] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    has_phase: bool = False
    error: str | None = None


# --------------------------------------------------------------------------
# 読み込み
# --------------------------------------------------------------------------


def camera_names(columns: list[str]) -> list[str]:
    names = []
    for c in columns:
        if not c.endswith(TS):
            continue
        key = c[: -len(TS)]
        if key in LOOP_KEYS or key.endswith((DEVICE_SUFFIX, ARRIVAL_SUFFIX)):
            continue
        names.append(key)
    return names


def _episode_lengths(ds: Path) -> dict[int, int]:
    files = sorted((ds / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        return {}
    ep = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "length"]) for f in files],
        ignore_index=True,
    )
    return dict(zip(ep.episode_index.astype(int), ep.length.astype(int)))


def load(ds: Path, *, include_reset: bool = False) -> Report:
    info_path = ds / "meta" / "info.json"
    try:
        fps = float(json.loads(info_path.read_text())["fps"])
    except (OSError, KeyError, ValueError) as e:
        return Report(path=ds, fps=float("nan"), error=f"meta/info.json を読めません: {e}")

    rep = Report(path=ds, fps=fps)
    ts_dir = ds / TIMESTAMP_LOGS_DIR
    files = sorted(ts_dir.glob("episode_*.parquet"))
    if not files:
        rep.error = "timestamp_logs がありません"
        return rep

    lengths = _episode_lengths(ds)
    for f in files:
        m = EPISODE_FILE_RE.search(f.name)
        if not m:
            continue
        idx = int(m.group(1))
        df = pd.read_parquet(f)
        total = len(df)
        rep.has_phase = rep.has_phase or "phase" in df.columns
        if include_reset:
            body = df
        elif "phase" in df.columns:
            body = df[df.phase == "episode"]
        elif idx in lengths:
            body = df.iloc[: lengths[idx]]
        else:
            body = df  # length が分からなければ全行（警告は表示側で出す）
        rep.episodes.append(Segment(episode_index=idx, df=body.reset_index(drop=True), total_rows=total))
        if not rep.cameras:
            rep.cameras = camera_names(list(df.columns))
    return rep


def load_raw(target: Path, fps: float) -> Report:
    """データセットでない時刻ログ（rollout の base 戦略が書く rollout_*.parquet）を読む。

    ファイルなら1本、ディレクトリなら中の rollout_*.parquet をそれぞれ1エピソードとして扱う。
    リセットの区別は無いので、全行を使う。
    """
    files = [target] if target.is_file() else sorted(target.glob("rollout_*.parquet"))
    rep = Report(path=target, fps=fps)
    if not files:
        rep.error = "rollout_*.parquet が見つかりません"
        return rep
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        rep.episodes.append(Segment(episode_index=i, df=df, total_rows=len(df)))
        if not rep.cameras:
            rep.cameras = camera_names(list(df.columns))
    return rep


# --------------------------------------------------------------------------
# 集計
# --------------------------------------------------------------------------


def _q(x: np.ndarray, q: float) -> float | None:
    return float(np.quantile(x, q)) if len(x) else None


def _ms(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 1e3


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    parts = [p[~np.isnan(p)] for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.array([])


def summarize(rep: Report, segments: list[Segment] | None = None) -> dict[str, float | int | None]:
    """指標を順序付き dict で返す。差分はエピソードの境目をまたがない。"""
    segs = rep.episodes if segments is None else segments
    period_ms = 1e3 / rep.fps
    out: dict[str, float | int | None] = {}

    periods, controls, reads, lates = [], [], [], []
    frames = 0
    duration = 0.0
    for s in segs:
        d = s.df
        frames += len(d)
        if len(d) < 2:
            continue
        start = d["loop_start" + TS].to_numpy(dtype=float)
        periods.append(np.diff(_ms(start)))
        duration += start[-1] - start[0]
        controls.append(_ms(d["loop_control_end" + TS] - d["loop_start" + TS]))
        if "observation_end" + TS in d:
            reads.append(_ms(d["observation_end" + TS] - d["observation" + TS]))
        if "target_start" + TS in d:
            lates.append(_ms(d["loop_start" + TS] - d["target_start" + TS]))

    p = _concat(periods)
    out["episodes"] = len(segs)
    out["frames"] = frames
    out["period_p50_ms"] = _q(p, 0.5)
    out["period_p95_ms"] = _q(p, 0.95)
    out["period_p99_ms"] = _q(p, 0.99)
    out["period_max_ms"] = float(p.max()) if len(p) else None
    out["period_late_pct"] = float((p > LATE_RATIO * period_ms).mean() * 100) if len(p) else None
    n_intervals = sum(len(x) for x in periods)
    eff = n_intervals / duration if duration > 0 else None
    out["effective_fps"] = eff
    # データセットの時刻は frame_index / fps。実時間に対して何 % 遅れて進むか
    out["clock_slip_pct"] = (rep.fps / eff - 1) * 100 if eff else None

    c = _concat(controls)
    out["control_p50_ms"] = _q(c, 0.5)
    out["control_p95_ms"] = _q(c, 0.95)
    out["control_max_ms"] = float(c.max()) if len(c) else None

    r = _concat(reads)
    out["motor_read_p50_ms"] = _q(r, 0.5)
    out["motor_read_p99_ms"] = _q(r, 0.99)
    out["motor_read_max_ms"] = float(r.max()) if len(r) else None

    lt = _concat(lates)
    out["wake_late_p50_ms"] = _q(lt, 0.5)
    out["wake_late_p99_ms"] = _q(lt, 0.99)
    out["wake_late_max_ms"] = float(lt.max()) if len(lt) else None

    for cam in rep.cameras:
        col = cam + TS
        gaps, ages, dev, lat, post, offs = [], [], [], [], [], []
        acol = cam + ARRIVAL_SUFFIX + TS
        dcol = cam + DEVICE_SUFFIX + TS
        for s in segs:
            d = s.df
            if col not in d or len(d) < 2:
                continue
            gaps.append(np.diff(_ms(d[col])))
            ages.append(_ms(d["observation" + TS] - d[col]))
            if dcol in d:
                dev.append(np.diff(_ms(d[dcol])))
                if "observation_end" + TS in d:
                    # 撮影から関節角を読み終えるまで = 画像が関節角より何 ms 前の瞬間か
                    offs.append(_ms(d["observation_end" + TS] - d[dcol]))
            if acol in d:
                post.append(_ms(d[col] - d[acol]))
                if dcol in d:
                    lat.append(_ms(d[acol] - d[dcol]))
        g = _concat(gaps)
        a = _concat(ages)
        out[f"{cam}.dup_pct"] = float((g < DUP_RATIO * period_ms).mean() * 100) if len(g) else None
        out[f"{cam}.gap_pct"] = float((g > GAP_RATIO * period_ms).mean() * 100) if len(g) else None
        out[f"{cam}.age_p50_ms"] = _q(a, 0.5)
        out[f"{cam}.age_max_ms"] = float(a.max()) if len(a) else None
        if dev:
            dv = _concat(dev)
            # センサ側の時刻で見た重複と欠け。ホスト側と食い違えば、原因は転送か読み取りの側にある
            out[f"{cam}.src_dup_pct"] = float((dv < DUP_RATIO * period_ms).mean() * 100) if len(dv) else None
            out[f"{cam}.src_gap_pct"] = float((dv > GAP_RATIO * period_ms).mean() * 100) if len(dv) else None
        if post:
            pp = _concat(post)
            out[f"{cam}.postproc_p50_ms"] = _q(pp, 0.5)
            out[f"{cam}.postproc_p99_ms"] = _q(pp, 0.99)
        if offs:
            # 時間差の一定さ。エピソードの中の幅と、エピソードごとの中央値の違いを分けて見る
            # （位相合わせ phase_align_camera が効いていれば、後者が小さくなる）
            meds = [float(np.median(o)) for o in offs if len(o)]
            out[f"{cam}.offset_p50_ms"] = _q(_concat(offs), 0.5)
            out[f"{cam}.offset_within_ms"] = float(np.mean([_q(o, 0.95) - _q(o, 0.05) for o in offs if len(o)]))
            out[f"{cam}.offset_between_ms"] = max(meds) - min(meds) if len(meds) > 1 else None
        if lat:
            la = _concat(lat)
            # センサの時計がホストに写っていない（hardware_clock）と桁違いになる。そのときは出さない
            if len(la) and abs(np.median(la)) < 1000:
                out[f"{cam}.transfer_p50_ms"] = _q(la, 0.5)
                out[f"{cam}.transfer_p99_ms"] = _q(la, 0.99)
    return out


def per_episode(rep: Report) -> list[dict[str, float | int | None]]:
    rows = []
    for s in rep.episodes:
        row = {"episode": s.episode_index, "rows_in_file": s.total_rows}
        row.update(summarize(rep, [s]))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# 表示
# --------------------------------------------------------------------------

LABELS = {
    "episodes": "エピソード数",
    "frames": "フレーム数（本体）",
    "period_p50_ms": "周期 中央値 [ms]",
    "period_p95_ms": "周期 p95 [ms]",
    "period_p99_ms": "周期 p99 [ms]",
    "period_max_ms": "周期 最大 [ms]",
    "period_late_pct": "予算超え(>1.2周期) [%]",
    "effective_fps": "実効 fps",
    "clock_slip_pct": "データセット時刻の遅れ [%]",
    "control_p50_ms": "制御部 中央値 [ms]",
    "control_p95_ms": "制御部 p95 [ms]",
    "control_max_ms": "制御部 最大 [ms]",
    "motor_read_p50_ms": "モーター読取 中央値 [ms]",
    "motor_read_p99_ms": "モーター読取 p99 [ms]",
    "motor_read_max_ms": "モーター読取 最大 [ms]",
    "wake_late_p50_ms": "起床の遅れ 中央値 [ms]",
    "wake_late_p99_ms": "起床の遅れ p99 [ms]",
    "wake_late_max_ms": "起床の遅れ 最大 [ms]",
}
CAM_LABELS = {
    "dup_pct": "重複 [%]",
    "gap_pct": "欠け [%]",
    "age_p50_ms": "古さ 中央値 [ms]",
    "age_max_ms": "古さ 最大 [ms]",
    "src_dup_pct": "センサ側 重複 [%]",
    "src_gap_pct": "センサ側 欠け [%]",
    "postproc_p50_ms": "後処理 中央値 [ms]",
    "postproc_p99_ms": "後処理 p99 [ms]",
    "transfer_p50_ms": "転送遅延 中央値 [ms]",
    "transfer_p99_ms": "転送遅延 p99 [ms]",
    "offset_p50_ms": "時間差(撮影→関節角) 中央値 [ms]",
    "offset_within_ms": "時間差 エピソード内の幅 [ms]",
    "offset_between_ms": "時間差 エピソード間の差 [ms]",
}


def label(key: str) -> str:
    if "." in key:
        cam, k = key.split(".", 1)
        return f"{cam}: {CAM_LABELS.get(k, k)}"
    return LABELS.get(key, key)


def fmt(v: float | int | None) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}"


def render_table(headers: list[str], rows: list[tuple[str, list[str]]]) -> str:
    """rows = [(行ラベル, [列ごとの値])]。日本語の幅は2として揃える。"""

    def width(s: str) -> int:
        return sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)

    def pad(s: str, w: int, right: bool) -> str:
        sp = " " * max(w - width(s), 0)
        return sp + s if right else s + sp

    w0 = max([width(r[0]) for r in rows] + [0])
    ws = [max([width(h)] + [width(r[1][i]) for r in rows]) for i, h in enumerate(headers)]
    lines = [pad("", w0, False) + "  " + "  ".join(pad(h, w, True) for h, w in zip(headers, ws))]
    for name, vals in rows:
        lines.append(pad(name, w0, False) + "  " + "  ".join(pad(v, w, True) for v, w in zip(vals, ws)))
    return "\n".join(lines)


def side_by_side(names: list[str], summaries: list[dict]) -> str:
    keys: list[str] = []
    for s in summaries:
        for k in s:
            if k not in keys:
                keys.append(k)
    # どの列でも値が無い指標は出さない
    keys = [k for k in keys if any(s.get(k) is not None for s in summaries)]
    rows = [(label(k), [fmt(s.get(k)) for s in summaries]) for k in keys]
    return render_table(names, rows)


def main() -> int:
    """timestamp_logs を集計して、条件ごとに横に並べて表示する。

        python so101/scripts/timing_report.py <データセットのディレクトリ>... [--per-episode]
        python so101/scripts/timing_report.py <rollout_*.parquet のあるディレクトリ> --fps 30
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="+", help="データセットのディレクトリ、または rollout_*.parquet（のあるディレクトリ）")
    ap.add_argument("--include-reset", action="store_true", help="リセット中の周回も含める")
    ap.add_argument("--per-episode", action="store_true", help="エピソード別の表も出す")
    ap.add_argument("--fps", type=float, default=30, help="rollout_*.parquet を直接読むときの fps")
    args = ap.parse_args()

    reports = []
    for t in args.target:
        p = Path(t).expanduser()
        if p.suffix == ".parquet" or (p.is_dir() and any(p.glob("rollout_*.parquet"))):
            rep = load_raw(p, args.fps)
        else:
            rep = load(p, include_reset=args.include_reset)
        if rep.error:
            print(f"{t}: {rep.error}", file=sys.stderr)
            return 1
        reports.append((t, rep))

    for i, (t, rep) in enumerate(reports, 1):
        print(f"[{i}] {t}  ({rep.fps:g} fps, カメラ {', '.join(rep.cameras) or 'なし'})")
    print()
    print(side_by_side([f"[{i}]" for i in range(1, len(reports) + 1)], [summarize(rep) for _, rep in reports]))
    if args.per_episode:
        for i, (t, rep) in enumerate(reports, 1):
            rows = per_episode(rep)
            print(f"\n[{i}] {t} のエピソード別")
            print(side_by_side([f"ep{r['episode']}" for r in rows], rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
