"""Diffusion Policy / Flow Matching Policy の推論の設定（スケジューラ・ステップ数・高速化）ごとに、行動の質をデータセットで比べる（アームなし）。

データセットのフレームを選び、学習と同じ並び（観測 n_obs_steps コマ、行動 horizon 個）で取り出して chunk を予測する。
予測した行動（実行に使う n_action_steps 個）と、収録した行動との差を関節ごとの度で出す。
同じフレームには同じ初期ノイズを使うので、設定の違いだけが比べられる。

- 「収録との差」: モデルの誤差 + 設定による劣化
- 「基準との差」: 基準の設定（既定 Diffusion は DDPM100、Flow Matching は FM100）の出力との差。設定による変化だけを見る
- 基準を別のノイズで回した「基準どうしの差」が、サンプリングそのものの揺らぎ（これより小さい変化は無視できる）

    python so101/scripts/offline_eval.py <pretrained_model> --dataset-root <dir> [--dataset-root <dir> ...] \
        --configs DDPM100,DDIM25,DDIM10,DDIM5 [--frames 200] [--csv out.csv]

設定の書式: <DDPM|DDIM|FM><ステップ数>[+fast]。FM（Euler のステップ数）は Flow Matching のモデル専用、DDPM/DDIM は Diffusion 専用。
+fast は 重み fp16・U-Net だけ cuDNN なし・エンコーダ channels_last（推論の高速化）。
フレームと初期ノイズは --seed で決まるので、Diffusion と Flow Matching を別々に回しても同じフレーム・同じノイズで比べられる。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import re
import time
from pathlib import Path

import numpy as np
import torch


def parse_config(s: str) -> tuple[str, int, bool]:
    m = re.fullmatch(r"(DDPM|DDIM|FM)(\d+)(\+fast)?", s)
    if not m:
        raise SystemExit(f"設定の書式が違う: {s}")
    return m.group(1), int(m.group(2)), bool(m.group(3))


def make_scheduler(cfg, kind: str):
    from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler

    return _make_noise_scheduler(
        kind,
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        clip_sample=cfg.clip_sample,
        clip_sample_range=cfg.clip_sample_range,
        prediction_type=cfg.prediction_type,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--dataset-root", type=Path, action="append", required=True)
    ap.add_argument("--configs", help="既定: Diffusion は DDPM100,DDIM25,DDIM16,DDIM10,DDIM8,DDIM5、Flow Matching は FM10,FM4,FM2,FM1")
    ap.add_argument("--reference", help="既定: Diffusion は DDPM100、Flow Matching は FM100")
    ap.add_argument("--frames", type=int, default=200, help="データセットごとに選ぶフレームの数")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, help="フレームごと・設定ごとの差を書き出す")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class

    model_dir = a.model / "pretrained_model" if (a.model / "pretrained_model").is_dir() else a.model
    policy_type = PreTrainedConfig.from_pretrained(model_dir).type
    policy = get_policy_class(policy_type).from_pretrained(model_dir).to("cuda").eval()
    cfg = policy.config
    is_fm = policy_type == "flow_matching"
    if a.configs is None:
        a.configs = "FM10,FM4,FM2,FM1" if is_fm else "DDPM100,DDIM25,DDIM16,DDIM10,DDIM8,DDIM5"
    if a.reference is None:
        a.reference = "FM100" if is_fm else "DDPM100"
    m = policy.diffusion
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(model_dir),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    n_obs, horizon = cfg.n_obs_steps, cfg.horizon
    n_exec = min(cfg.n_action_steps, horizon - n_obs + 1)  # 実際に使われる行動の数（切り出し [n_obs-1 : n_obs-1+n_action_steps]）
    configs = a.configs.split(",")
    if a.reference not in configs:
        configs.insert(0, a.reference)
    for conf in configs:
        if (parse_config(conf)[0] == "FM") != is_fm:
            raise SystemExit(f"{conf} は {policy_type} のモデルには使えない")
    print(f"モデル: {model_dir}（{policy_type}）\n  実行に使う行動 {n_exec} 個（n_action_steps={cfg.n_action_steps}, horizon={horizon}, n_obs_steps={n_obs}）")

    rng = np.random.default_rng(a.seed)
    samples = []  # (データセット名, batch dict)
    for root in a.dataset_root:
        root = root.expanduser()
        probe = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
        fps = probe.fps
        delta = {k: [i / fps for i in cfg.observation_delta_indices] for k in [*cfg.image_features, "observation.state"]}
        delta["action"] = [i / fps for i in cfg.action_delta_indices]
        ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root, delta_timestamps=delta)
        idx = np.sort(rng.choice(len(ds), size=min(a.frames, len(ds)), replace=False))
        t0 = time.perf_counter()
        items = [ds[int(i)] for i in idx]
        print(f"  {root.name}: {ds.num_episodes} エピソード {len(ds)} フレームから {len(items)} 個（読み込み {time.perf_counter() - t0:.0f} 秒）")
        samples += [(root.name, it) for it in items]

    keys = [*cfg.image_features, "observation.state"]
    noise_gen = torch.Generator(device="cuda").manual_seed(a.seed)
    noises = torch.randn(len(samples), horizon, cfg.action_feature.shape[0], device="cuda", generator=noise_gen)
    noises_alt = torch.randn(len(samples), horizon, cfg.action_feature.shape[0], device="cuda", generator=noise_gen)

    def predict(conf: str, noise: torch.Tensor) -> tuple[np.ndarray, float]:
        kind, k, fast = parse_config(conf)
        if kind != "FM":
            m.noise_scheduler = make_scheduler(cfg, kind)
        m.num_inference_steps = k
        if fast:
            m.to(torch.float16)
            m.rgb_encoder.to(memory_format=torch.channels_last)
        else:
            m.to(torch.float32)
            m.rgb_encoder.to(memory_format=torch.contiguous_format)
        unet_forward = m.unet.forward
        if fast:
            def no_cudnn(*args, **kwargs):
                with torch.backends.cudnn.flags(enabled=False):
                    return unet_forward(*args, **kwargs)
            m.unet.forward = no_cudnn
        outs, elapsed = [], 0.0
        try:
            for s in range(0, len(samples), a.batch):
                chunk = samples[s : s + a.batch]
                batch = {k_: torch.stack([it[k_] for _, it in chunk]) for k_ in keys}
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    batch = pre(batch)
                    policy.reset()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    act = policy.predict_action_chunk(batch, noise=noise[s : s + len(chunk)])
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - t0
                    act = post(act.float())
                outs.append(act.float().cpu().numpy())
        finally:
            m.unet.forward = unet_forward
        return np.concatenate(outs), elapsed / len(samples) * 1e3

    gt = np.stack([it["action"].numpy()[n_obs - 1 : n_obs - 1 + n_exec] for _, it in samples])
    pad = np.stack([it["action_is_pad"].numpy()[n_obs - 1 : n_obs - 1 + n_exec] for _, it in samples])
    valid = ~pad[..., None]
    names = [n for n, _ in samples]

    def mae(x: np.ndarray, y: np.ndarray, sel: np.ndarray) -> float:
        w = (valid[sel] * np.ones_like(x[sel])).astype(bool)
        return float(np.abs(x[sel] - y[sel])[w].mean())

    preds, ms = {}, {}
    for conf in configs:
        preds[conf], ms[conf] = predict(conf, noises)
    ref_alt, _ = predict(a.reference, noises_alt)

    groups = {n: np.array([x == n for x in names]) for n in dict.fromkeys(names)}
    groups["全体"] = np.ones(len(names), bool)
    print(f"\n行動の差（全関節・全 {n_exec} 個の平均、度。gripper も同じ単位で含む）。時間は batch {a.batch} で割った 1 フレームあたり")
    print(f"基準 {a.reference} を別のノイズで回した差（サンプリングの揺らぎ）: "
          + " / ".join(f"{g} {mae(ref_alt, preds[a.reference], sel):.2f}" for g, sel in groups.items()))
    header = f"{'設定':14s} {'ms/フレーム':>10s} " + " ".join(f"{'収録との差 ' + g:>22s} {'基準との差 ' + g:>22s}" for g in groups)
    print(header)
    for conf in configs:
        cells = " ".join(f"{mae(preds[conf], gt, sel):22.2f} {mae(preds[conf], preds[a.reference], sel):22.2f}" for sel in groups.values())
        print(f"{conf:14s} {ms[conf]:10.1f} {cells}")

    # chunk の中の滑らかさ: 隣の行動との差（全関節の最大）と、行って戻るジグザグ（同じ関節で向きが逆の 2 度超の差が続く）
    def smooth(x: np.ndarray) -> tuple[float, float, float]:
        dx = np.diff(x, axis=1)
        ok = valid[:, 1:, 0] & valid[:, :-1, 0]
        jump = np.abs(dx).max(-1)[ok]
        zz = (np.sign(dx[:, 1:]) * np.sign(dx[:, :-1]) < 0) & (np.abs(dx[:, 1:]) > 2) & (np.abs(dx[:, :-1]) > 2)
        zz_ok = ok[:, 1:] & ok[:, :-1]
        return float(np.percentile(jump, 99)), float(jump.max()), float(zz.any(-1)[zz_ok].mean() * 100)

    print("\nchunk の中の滑らかさ（全体）: 隣の行動との差 p99 / 最大（度）、ジグザグの割合（%、隣どうし逆向きに 2 度超）")
    r = smooth(gt)
    print(f"  {'収録':14s} {r[0]:6.2f} {r[1]:6.2f} {r[2]:6.2f}")
    for conf in configs:
        r = smooth(preds[conf])
        print(f"  {conf:14s} {r[0]:6.2f} {r[1]:6.2f} {r[2]:6.2f}")

    # 関節別（全体）
    joints = [f"j{i}" for i in range(gt.shape[-1])]
    print("\n関節別の収録との差（全体、度）: " + " ".join(f"{j:>7s}" for j in joints))
    for conf in configs:
        err = np.abs(preds[conf] - gt)
        per = [float(err[..., j][valid[..., 0]].mean()) for j in range(gt.shape[-1])]
        print(f"  {conf:14s}" + " ".join(f"{v:7.2f}" for v in per))

    # chunk の中の位置別（全体）: 先の予測ほど外れる。async_chunk の切り替えでは古い chunk の末尾と新しい chunk の頭がつながる
    bins = [(0, 10), (10, 30), (30, 50), (50, n_exec)]
    print("\nchunk の中の位置別の収録との差（全体、度）: " + " ".join(f"{f'{s}-{e - 1}':>7s}" for s, e in bins))
    for conf in configs:
        err = np.abs(preds[conf] - gt).mean(-1)
        per = [float(err[:, s:e][valid[:, s:e, 0]].mean()) for s, e in bins]
        print(f"  {conf:14s}" + " ".join(f"{v:7.2f}" for v in per))

    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "sample", "config", "ms_per_frame", "mae_vs_recorded_deg", "mae_vs_reference_deg"])
            for conf in configs:
                for i in range(len(samples)):
                    sel = np.zeros(len(samples), bool)
                    sel[i] = True
                    w.writerow([names[i], i, conf, f"{ms[conf]:.2f}", f"{mae(preds[conf], gt, sel):.4f}",
                                f"{mae(preds[conf], preds[a.reference], sel):.4f}"])
            for i in range(len(samples)):
                sel = np.zeros(len(samples), bool)
                sel[i] = True
                w.writerow([names[i], i, f"{a.reference}(別ノイズ)", "", f"{mae(ref_alt, gt, sel):.4f}",
                            f"{mae(ref_alt, preds[a.reference], sel):.4f}"])
        print(f"\nCSV: {a.csv}")


if __name__ == "__main__":
    main()
