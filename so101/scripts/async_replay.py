"""推論エンジン（sync / async_chunk の各設定）を、データセットのエピソードを実時間で流して比べる（アームにもカメラにも触らない）。

1エピソードの観測を先にデコードしておき、fps の実時間で engine.get_action() に渡す（rollout の制御ループと同じ呼び方）。
観測は収録のものなので、ロボットはこちらの行動に反応しない（開ループ）。タイミングと行動のつながりを比べるための道具。

周期ごとに記録するもの:
- get_action の時間（制御ループのスレッドが待たされた時間。30Hz の予算 33.3ms に効く）
- 保持（その周期の行動が無く、前の行動を繰り返した）
- 行動の跳び |a_k - a_{k-1}|（chunk を切り替えた周期と、それ以外）。収録データの跳びの分布と比べる
- 収録した行動との差

    python so101/scripts/async_replay.py <pretrained_model> --dataset-root <dir> --episode 0 --episode 1 \\
        --policy-override noise_scheduler_type=DDIM --policy-override num_inference_steps=10 \\
        --engine sync --engine "--inference.type=async_chunk --inference.accel.half_weights=true" \\
        [--seconds 20] [--csv-dir out/]

--engine は何度でも指定でき、同じエピソードを順に流す。sync は lerobot の SyncInferenceEngine そのもの。
"""

from __future__ import annotations

import argparse
import shlex
import time
from pathlib import Path

import numpy as np
import torch


def load_policy(model_dir: Path, overrides: list[str]):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(model_dir)
    cfg.device = "cuda"
    for o in overrides:
        key, value = o.split("=", 1)
        old = getattr(cfg, key)
        setattr(cfg, key, type(old)(value) if old is not None and not isinstance(old, str) else value)
    policy = get_policy_class(cfg.type).from_pretrained(model_dir, config=cfg).to("cuda").eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(model_dir),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    return policy, pre, post


def episode_frames(root: Path, episode: int, seconds: float) -> tuple[list[dict], np.ndarray, list[str], float]:
    """エピソードの観測を build_dataset_frame と同じ形（画像は HWC uint8、state は float32）で返す。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root, episodes=[episode])
    n = min(len(ds), int(seconds * ds.fps))
    frames, actions = [], []
    image_keys = [k for k in ds.meta.features if k.startswith("observation.images.")]
    for i in range(n):
        it = ds[i]
        f = {"observation.state": it["observation.state"].numpy().astype(np.float32)}
        for k in image_keys:
            f[k] = (it[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        frames.append(f)
        actions.append(it["action"].numpy())
    names = list(ds.meta.features["action"]["names"])
    return frames, np.stack(actions), names, float(ds.fps)


def make_engine(spec: str, policy, pre, post, features: dict, keys: list[str], fps: float, task: str = "replay"):
    from lerobot.rollout.inference import SyncInferenceEngine

    if spec.strip() == "sync":
        return SyncInferenceEngine(policy=policy, preprocessor=pre, postprocessor=post, dataset_features=features,
                                   ordered_action_keys=keys, task=task, device="cuda", robot_type="so101_follower")
    import draccus

    from lerobot.rollout.inference.async_chunk import AsyncChunkInferenceConfig, build_async_chunk_engine

    args = [a.replace("--inference.", "--", 1) for a in shlex.split(spec) if not a.startswith("--inference.type")]
    cfg = draccus.parse(AsyncChunkInferenceConfig, args=args)
    return build_async_chunk_engine(cfg, policy=policy, preprocessor=pre, postprocessor=post, dataset_features=features,
                                    ordered_action_keys=keys, task=task, robot_type="so101_follower", fps=fps,
                                    device="cuda")


def run(engine, frames: list[dict], fps: float) -> dict:
    is_async = hasattr(engine, "_tick_rows")
    engine.reset()
    engine.start()
    # rollout と同じ空打ち（strategies/core.py の _warm_up_inference）: 1回推論して捨て、reset する。
    engine.get_action(frames[0])
    engine.reset()
    engine.resume()
    acts, call_ms = [], []
    t0 = time.perf_counter()
    for k, f in enumerate(frames):
        while (wait := t0 + k / fps - time.perf_counter()) > 0:
            time.sleep(min(wait, 0.002))
        ts = time.perf_counter()
        a = engine.get_action(f)
        call_ms.append((time.perf_counter() - ts) * 1e3)
        acts.append(a.numpy() if a is not None else np.full(len(acts[-1]) if acts else 6, np.nan))
    # 空打ちの記録を除く（reset ごとに epoch が増える）
    epoch = engine._epoch if is_async else None
    ticks = [r for r in engine._tick_rows if r["epoch"] == epoch] if is_async else []
    chunks = [r for r in engine._chunk_rows if r["epoch"] == epoch] if is_async else []
    engine.stop()
    return {"actions": np.stack(acts), "call_ms": np.array(call_ms), "ticks": ticks, "chunks": chunks}


def summarize(r: dict, rec: np.ndarray, budget: float) -> dict:
    acts, ms = r["actions"], r["call_ms"]
    jump = np.abs(np.diff(acts, axis=0)).max(axis=1)
    if r["ticks"]:
        cid = np.array([row["chunk_id"] for row in r["ticks"]])
        switch = np.zeros(len(acts) - 1, bool)
        switch[: len(cid) - 1] = cid[1:] != cid[:-1]
        held = sum(row["held"] for row in r["ticks"])
        delays = [c["delay_ticks"] for c in r["chunks"]]
        latency = [c["latency_ms"] for c in r["chunks"]]
    else:
        # sync: chunk を計算した周期 = get_action が予算の半分を超えた周期とみなす
        switch = ms[1:] > budget / 2
        held, delays, latency = 0, [0] * int((ms > budget / 2).sum()), list(ms[ms > budget / 2])
    valid = ~np.isnan(acts).any(axis=1)
    # 行き戻り: 隣どうしの差の符号が逆で、両方 0.3 度超（どれかの関節で）
    d = np.diff(acts, axis=0)
    rev = ((d[1:] * d[:-1] < 0) & (np.abs(d[1:]) > 0.3) & (np.abs(d[:-1]) > 0.3)).any(axis=1)
    return {"ms": ms, "jump": jump, "switch": switch, "held": held, "delays": delays, "latency": latency,
            "err": np.abs(acts[valid] - rec[valid]).ravel(), "over": int((ms > budget).sum()), "rev": rev}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--episode", type=int, action="append", help="何度でも指定できる（既定 0）")
    ap.add_argument("--seconds", type=float, default=20.0, help="各エピソードの先頭から流す長さ")
    ap.add_argument("--policy-override", action="append", default=[], help="config の項目=値（例 num_inference_steps=10）")
    ap.add_argument("--engine", action="append", required=True, help="sync か、--inference.* の引数列（async_chunk）")
    ap.add_argument("--csv-dir", type=Path, help="周期ごとの記録を設定・エピソードごとに CSV で書き出す")
    ap.add_argument("--task", default="replay", help="言語の指示（SmolVLA など言語を使うポリシーでは学習の文言に合わせる）")
    a = ap.parse_args()
    episodes = a.episode or [0]

    model_dir = a.model / "pretrained_model" if (a.model / "pretrained_model").is_dir() else a.model
    root = a.dataset_root.expanduser()
    data = []
    for ep in episodes:
        t = time.perf_counter()
        frames, rec, names, fps = episode_frames(root, ep, a.seconds)
        data.append((ep, frames, rec))
        print(f"{root.name} エピソード {ep}: {len(frames)} 周期（{len(frames) / fps:.1f} 秒）、デコード {time.perf_counter() - t:.0f} 秒")
    features = {"action": {"names": names}}
    rec_jump = np.concatenate([np.abs(np.diff(rec, axis=0)).max(axis=1) for _, _, rec in data])
    print(f"収録の行動の跳び（周期ごとの全関節の最大、度）: 中央値 {np.median(rec_jump):.2f} / p99 {np.percentile(rec_jump, 99):.2f} / 最大 {rec_jump.max():.2f}")
    budget = 1e3 / fps

    print(f"\n{'エンジン':48s} {'get_action ms 中央/p99/最大':>24s} {'予算超':>4s} {'保持':>4s} {'chunk':>5s} {'計算 ms':>7s} {'遅れ':>5s} "
          f"{'跳び 切替 中央/p99/最大':>20s} {'跳び 全周期 p99':>13s} {'収録との差':>6s} {'行き戻り':>5s}")
    asyncs = [e for e in a.engine if e.strip() != "sync"]
    common = set.intersection(*(set(shlex.split(e)) for e in asyncs)) if len(asyncs) > 1 else set()
    for spec in a.engine:
        # エンジンごとに読み込み直す（高速化の設定が重みの dtype などを書き換えるため）
        policy, pre, post = load_policy(model_dir, a.policy_override)
        engine = make_engine(spec, policy, pre, post, features, names, fps, a.task)
        runs = []
        for ep, frames, rec in data:
            r = run(engine, frames, fps)
            runs.append(summarize(r, rec, budget))
            if a.csv_dir:
                import csv

                a.csv_dir.mkdir(parents=True, exist_ok=True)
                tag = "sync" if spec.strip() == "sync" else "_".join(x.split("=")[-1] for x in shlex.split(spec) if "=" in x and x not in common) or "async"
                with (a.csv_dir / f"replay_{tag}_ep{ep}.csv").open("w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["tick", "get_action_ms", "chunk_switch", *[f"a_{n}" for n in names], *[f"rec_{n}" for n in names]])
                    sw = runs[-1]["switch"]
                    for k in range(len(r["actions"])):
                        w.writerow([k, f"{r['call_ms'][k]:.3f}", int(k > 0 and sw[k - 1]), *r["actions"][k], *rec[k]])
        cat = {k: np.concatenate([np.atleast_1d(np.asarray(x[k], dtype=float)) for x in runs]) for k in ("ms", "jump", "latency", "delays", "err", "rev")}
        sw = np.concatenate([x["jump"][x["switch"]] for x in runs])
        held = sum(x["held"] for x in runs)
        over = sum(x["over"] for x in runs)
        label = " ".join(x.replace("--inference.", "") for x in shlex.split(spec) if x not in common) or spec
        label = label if len(label) <= 48 else label[:45] + "..."
        ms = cat["ms"]
        sw_s = f"{np.median(sw):5.1f}/{np.percentile(sw, 99):5.1f}/{sw.max():5.1f}" if len(sw) else "-"
        delay_s = f"{np.median(cat['delays']):.0f}/{cat['delays'].max():.0f}" if len(cat["delays"]) else "-"
        lat_s = f"{np.median(cat['latency']):.0f}" if len(cat["latency"]) else "-"
        print(f"{label:48s} {np.median(ms):6.2f}/{np.percentile(ms, 99):6.2f}/{ms.max():7.1f} {over:4d} {held:4d} {len(sw):5d} {lat_s:>7s} {delay_s:>5s} "
              f"{sw_s:>20s} {np.percentile(cat['jump'], 99):13.2f} {cat['err'].mean():6.2f} {cat['rev'].mean() * 100:5.1f}%")
        del policy, engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
