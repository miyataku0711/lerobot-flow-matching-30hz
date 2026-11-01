"""Diffusion Policy / Flow Matching Policy の chunk 1回の内訳を測る（アームにもカメラにも触らない）。

chunk の計算 = 画像エンコーダ 1回 + (U-Net 1回 + scheduler.step 1回) × num_inference_steps。
Flow Matching は scheduler の代わりに Euler の 1 ステップ（足し算 1 回）で、chunk は FM<ステップ数> と表示する。
それぞれを「CPU が指示を出し終えるまで」と「GPU が終わるまで」に分けて測り、
精度と勾配の文脈（rollout は inference_mode + autocast）・スケジューラ（DDPM / DDIM）・ステップ数を変えて比べる。

    python so101/scripts/chunk_profile.py <pretrained_model> [--steps 10,16,100] [--profile]

--profile を付けると、U-Net 1回を torch.profiler で測り、GPU 時間の多いカーネルを表示する。
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import torch


def timed(fn, n: int, warmup: int = 3) -> tuple[float, float]:
    """(CPU が指示を出し終えるまで, GPU が終わるまで) の中央値 [ms]。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    cpu, total = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        cpu.append(t1 - t0)
        total.append(t2 - t0)
    return float(np.median(cpu) * 1e3), float(np.median(total) * 1e3)


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
    ap.add_argument("--steps", default="10,100", help="測る num_inference_steps（カンマ区切り）。DDPM は 100 のみ")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--profile", action="store_true", help="U-Net 1回のカーネル別の GPU 時間を表示する")
    a = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import get_policy_class

    model_dir = a.model / "pretrained_model" if (a.model / "pretrained_model").is_dir() else a.model
    policy_type = PreTrainedConfig.from_pretrained(model_dir).type
    policy = get_policy_class(policy_type).from_pretrained(model_dir).to("cuda").eval()
    m, cfg = policy.diffusion, policy.config
    is_fm = policy_type == "flow_matching"
    n_cam = len(cfg.image_features)
    c, h, w = next(iter(cfg.image_features.values())).shape
    state_dim = cfg.robot_state_feature.shape[0]
    print(f"モデル: {model_dir}（{policy_type}）")
    print(f"  n_obs_steps={cfg.n_obs_steps} horizon={cfg.horizon} n_action_steps={cfg.n_action_steps} "
          f"カメラ {n_cam} 台 {c}x{h}x{w} down_dims={list(cfg.down_dims)}")
    n_unet = sum(p.numel() for p in m.unet.parameters())
    n_enc = sum(p.numel() for p in m.rgb_encoder.parameters())
    print(f"  パラメータ U-Net {n_unet / 1e6:.1f}M / 画像エンコーダ {n_enc / 1e6:.1f}M")

    def batch(dtype=torch.float32):
        return {
            "observation.state": torch.randn(1, cfg.n_obs_steps, state_dim, device="cuda", dtype=dtype),
            "observation.images": torch.rand(1, cfg.n_obs_steps, n_cam, c, h, w, device="cuda", dtype=dtype),
        }

    # (名前, 勾配の文脈, autocast, 重みの dtype, 高速化)。rollout の sync エンジンは inference_mode + autocast。
    # 高速化 = エンコーダ channels_last + U-Net だけ cuDNN なし（async_chunk の --inference.accel と同じ）
    modes = [
        ("fp32（autocast なし）", torch.inference_mode, False, torch.float32, False),
        ("inference_mode + autocast（rollout と同じ）", torch.inference_mode, True, torch.float32, False),
        ("no_grad + autocast", torch.no_grad, True, torch.float32, False),
        ("重み fp16 + inference_mode + autocast", torch.inference_mode, True, torch.float16, False),
        ("重み fp16 + no_grad + autocast + 高速化", torch.no_grad, True, torch.float16, True),
    ]
    unet_forward = m.unet.forward

    def unet_without_cudnn(*args, **kwargs):
        with torch.backends.cudnn.flags(enabled=False):
            return unet_forward(*args, **kwargs)

    steps = [int(s) for s in a.steps.split(",")]
    print(f"\n{'条件':44s} {'enc':>12s} {'U-Net 1回':>12s} {'step 1回':>12s}  chunk（スケジューラ・ステップ数）  [ms, CPU/GPU 完了]")
    for name, grad_ctx, amp, wdtype, fast in modes:
        m.to(wdtype)
        m.rgb_encoder.to(memory_format=torch.channels_last if fast else torch.contiguous_format)
        m.unet.forward = unet_without_cudnn if fast else unet_forward
        amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else contextlib.nullcontext
        with grad_ctx(), amp_ctx():
            b = batch()
            gc = m._prepare_global_conditioning(b)
            x = torch.randn(1, cfg.horizon, cfg.action_feature.shape[0], device="cuda", dtype=gc.dtype)
            tt = torch.full((1,), 50, dtype=torch.long, device="cuda")
            enc = timed(lambda: m._prepare_global_conditioning(b), a.runs)
            unet = timed(lambda: m.unet(x, tt, global_cond=gc), a.runs * 3)
            out = m.unet(x, tt, global_cond=gc)
            if is_fm:
                step = timed(lambda: x + -0.1 * out, a.runs * 3)
            else:
                sched = make_scheduler(cfg, "DDIM")
                sched.set_timesteps(10)
                step = timed(lambda: sched.step(out, sched.timesteps[3], x), a.runs * 3)
            chunks = []
            for k in steps if is_fm else []:
                m.num_inference_steps = k
                t = timed(lambda: m.generate_actions(b), max(3, a.runs // (2 if k > 20 else 1)), warmup=1)
                chunks.append(f"FM{k} {t[1]:.0f}")
            for kind in () if is_fm else ("DDPM", "DDIM"):
                for k in steps:
                    if kind == "DDPM" and k != cfg.num_train_timesteps:
                        continue
                    m.noise_scheduler = make_scheduler(cfg, kind)
                    m.num_inference_steps = k
                    t = timed(lambda: m.generate_actions(b), max(3, a.runs // (2 if k > 20 else 1)), warmup=1)
                    chunks.append(f"{kind}{k} {t[1]:.0f}")
        print(f"{name:44s} {enc[0]:5.1f}/{enc[1]:5.1f} {unet[0]:5.2f}/{unet[1]:5.2f} {step[0]:5.2f}/{step[1]:5.2f}  "
              + "  ".join(chunks))
    m.to(torch.float32)
    m.rgb_encoder.to(memory_format=torch.contiguous_format)
    m.unet.forward = unet_forward

    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            b = batch()
            gc = m._prepare_global_conditioning(b)
            x = torch.randn(1, cfg.horizon, cfg.action_feature.shape[0], device="cuda")
            tt = torch.full((1,), 50, dtype=torch.long, device="cuda")
            for _ in range(3):
                m.unet(x, tt, global_cond=gc)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    m.unet(x, tt, global_cond=gc)
                torch.cuda.synchronize()
        print("\nU-Net 5回（inference_mode + autocast）の GPU 時間の多いカーネル")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=70))


if __name__ == "__main__":
    main()
