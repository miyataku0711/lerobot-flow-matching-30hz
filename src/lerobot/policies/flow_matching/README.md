# Flow Matching Policy

Diffusion Policy's network trained with a flow-matching (rectified flow) objective, so that a few Euler
steps replace the DDPM/DDIM denoising loop at inference time.

- **Network**: unchanged from `policies/diffusion` (ResNet + SpatialSoftmax image encoders, one per camera
  by default, and the 1D conditional U-Net with FiLM). The modules are imported, not copied, and keep
  Diffusion Policy's attribute names (`diffusion.rgb_encoder`, `diffusion.unet`).
- **Objective**: `x_t = t * noise + (1 - t) * actions`, the U-Net regresses the velocity `noise - actions`
  (MSE). Training times follow openpi's `Beta(1.5, 1) * 0.999 + 0.001` by default
  (`--policy.time_sampling=uniform` for uniform). The U-Net's step embedding sees `t * 100`, the range
  Diffusion Policy trains it on.
- **Sampler**: forward Euler from t=1 (noise) to t=0 with `--policy.num_inference_steps` steps
  (`lerobot.policies.common.flow_matching.euler_integrate`, shared with pi0 / SmolVLA). The final sample
  is clipped to `[-1, 1]` like Diffusion Policy's `clip_sample`.

The flow convention is the same as pi0 and SmolVLA, so the async-chunk rollout guides (`inpaint`, `rtc`)
apply directly (`rollout/inference/async_chunk/adapters.py: FlowMatchingAdapter`).

## Usage

```bash
lerobot-train --policy.type=flow_matching --policy.horizon=64 --policy.n_action_steps=64 \
  --dataset.repo_id=<repo_id> --dataset.root=<root> --policy.use_amp=true ...

lerobot-rollout --policy.path=<pretrained_model> --policy.num_inference_steps=4 \
  --inference.type=async_chunk --inference.guide.type=inpaint \
  --inference.accel.half_weights=true --inference.accel.channels_last=true --inference.accel.unet_cudnn=false ...
```

The step count is an inference-time setting: a model trained once can be run with any
`num_inference_steps`.

## References

- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (https://huggingface.co/papers/2303.04137)
- Lipman et al., "Flow Matching for Generative Modeling" (https://huggingface.co/papers/2210.02747)
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (https://huggingface.co/papers/2209.03003)
- Black et al., "pi0: A Vision-Language-Action Flow Model for General Robot Control" (https://huggingface.co/papers/2410.24164)
