# Flow Matching policy for SO-101 at 30 Hz (LeRobot fork)

Physical AI 講座 応用編1 最終課題のコードです。
[LeRobot](https://github.com/huggingface/lerobot) v0.6.1 を土台に、SO-101 アームで Flow Matching ポリシーを
30Hz（1 周期 33.3ms）で動かすための変更を加えています。upstream の README は [README.lerobot.md](README.lerobot.md) にあります。

*English summary:* a Flow Matching policy (same network as Diffusion Policy, velocity-regression target,
4-step Euler sampling), an asynchronous chunk-inference engine for `lerobot-rollout`, and control-loop /
camera timing fixes, on top of LeRobot v0.6.1. Measured on an NVIDIA DGX Spark (GB10, aarch64).

## タスク

机の上の赤いブロックを黄色の箱へ、青いブロックを透明の箱へ入れて、home の姿勢に戻る。
SO-101 リーダー/フォロワーで遠隔操作して 240 エピソードを収録し、ACT・Diffusion Policy・Flow Matching を同じデータで学習した。
観測は関節角 6 個とカメラ 3 台（手首・左・右、640×480）。

## upstream からの変更

| 変更 | 場所 | 内容 |
|---|---|---|
| Flow Matching ポリシー | `src/lerobot/policies/flow_matching/` | 網は Diffusion Policy と同じ（カメラごとの ResNet-18 + 1D U-Net）。学習は速度 `noise − action` の回帰、推論は t=1→0 の Euler 法（π0・SmolVLA と同じ向き）。`--policy.type=flow_matching` |
| 非同期 chunk 推論 | `src/lerobot/rollout/inference/async_chunk/` | `--inference.type=async_chunk`。次の chunk を裏のスレッドで計算し、制御ループは止まらない。いつ計算を始めるか（trigger）、前の chunk への寄せ方（guide: inpaint / rtc）、つなぎ方（blend）、高速化（重み fp16・channels_last・U-Net の cuDNN 無効・CUDA Graph）を CLI で選ぶ |
| 制御ループを絶対時刻で刻む | `src/lerobot/utils/robot_utils.py`（`LoopClock`） | 「残り時間だけ眠る」と寝過ごし（約 0.5ms）が毎周期積もり 29.7Hz になっていた。`t0 + k·周期` まで眠る。`LEROBOT_LOOP_PACING=relative` で旧方式 |
| 画像を「前回の次の 1 枚」で選ぶ | `src/lerobot/cameras/frame_sequence.py`、OpenCV カメラ、`so_follower` | 「最新の 1 枚」だと到着の揺れで重複と欠けが交互に起きる。`LEROBOT_FRAME_SELECT=latest` で旧方式 |
| 周期ごとの時刻の記録 | `src/lerobot/utils/timestamp_logger.py`、rollout の各 strategy、`lerobot-record` | 周期・カメラ画像の時刻を parquet に記録（同期の計測用）。rollout の base は `--strategy.timestamp_log_dir` |
| 画像を uint8 のまま GPU へ | `src/lerobot/policies/utils.py` | CPU で float32（4 倍）にしてから送るのをやめ、GPU で変換。毎周期 5.8 → 0.6ms |
| 学習の混合精度と compile | `src/lerobot/scripts/lerobot_train.py`、`configs/train.py` | `--policy.use_amp=true` を学習でも bf16 で効かせる。`--torch_compile=true` を追加 |

試験: `tests/policies/flow_matching/`、`tests/test_rollout_async_chunk.py`。

## 環境

- NVIDIA DGX Spark（GB10、aarch64）、Ubuntu 24.04、CUDA 13.0、Python 3.12
- torch 2.11、torchvision 0.26。そのほかは upstream と同じ

```bash
pip install -e ".[feetech,diffusion,dataset,training]"
```

## 使い方

`so101/configs/rollout.yaml` の `<...>` を自分の機器のパスに置き換えてから使う。

### 学習

```bash
lerobot-train --dataset.repo_id=local/<dataset> --dataset.root=<dataset_dir> \
  --policy.type=flow_matching --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=true \
  --output_dir=<output_dir> --job_name=<run_name> \
  --batch_size=32 --num_workers=4 --steps=40000 --torch_compile=true --save_freq=5000
```

DGX Spark で 1 step 1.25 秒、40K step で約 14 時間。

### 推論（実機、30Hz）

```bash
taskset -c 5-9,15-19 lerobot-rollout --strategy.type=base --config_path=so101/configs/rollout.yaml \
  --policy.path=<pretrained_model> --policy.num_inference_steps=4 \
  --inference.type=async_chunk --inference.guide.type=inpaint --inference.guide.schedule=linear \
  --inference.trigger.type=remaining --inference.trigger.k=20 \
  --inference.blend.type=smooth --inference.blend.window=5 \
  --inference.accel.half_weights=true --inference.accel.channels_last=true --inference.accel.unet_cudnn=false \
  --inference.accel.cuda_graph=true \
  --inference.log_dir=<log_dir> --strategy.timestamp_log_dir=<log_dir> \
  --task="<task>" --fps=30 --duration=60 --display_data=false
```

- `taskset` は DGX Spark の高性能コア（Cortex-X925）に固定するため。省電力コアに置かれると chunk の周期が約 2 倍になる。
- `--inference.guide.schedule=linear` が Flow Matching では重要。既定の `exp` は決定的な Euler 法では前の chunk に寄りにくく、切り替えで段差が出る。

### 評価スクリプト（`so101/scripts/`）

| スクリプト | 用途 |
|---|---|
| `offline_eval.py` | データセットのフレームで chunk を予測し、収録した行動との差（度）を設定ごとに比べる（アームなし） |
| `chunk_profile.py` | chunk 1 個の計算時間をステップ数・高速化ごとに測る |
| `async_replay.py` | 収録エピソードを実時間で流して、推論エンジンの設定ごとに chunk の切り替えの跳びを比べる（アームなし） |
| `rollout_summary.py` / `backtrack.py` | 実機の記録（`--inference.log_dir` と `--strategy.timestamp_log_dir`）から周期・計算時間・切り替えの跳び・後戻りを集計 |
| `timing_report.py` | 収録・推論の時刻の記録から、制御周期とカメラ画像の重複・欠けを集計 |
| `success_tally.py` | 成功率の記録表（CSV）を方策ごとに集計 |

## 主な結果（DGX Spark、実機）

| | Diffusion Policy（DDPM 10 ステップ） | Flow Matching（4 ステップ） |
|---|---|---|
| 制御周期 | 30.000Hz | 30.000Hz |
| chunk 1 個の計算 | 41ms | 21ms |
| 観測から行動に効くまでの遅れ | 2 周期 | 1 周期 |
| 収録した操作との差（オフライン、600 フレーム） | 3.84° | 3.78° |

詳しくはレポートを参照。

## データとモデル

収録データと学習済みモデルは公開していません。

## ライセンス

Apache License 2.0（upstream の LeRobot と同じ。[LICENSE](LICENSE)）。
