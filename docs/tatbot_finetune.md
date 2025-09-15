# Finetuning pi‑0.5 on Tatbot (WANDB + Sweep + Remote Inference)

This guide shows how to finetune pi‑0.5 on the HF dataset `tatbot/wow-2025y-09m-10d-15h-34m-08s`, run a WANDB sweep on a Datacrunch H100 SXM5 80GB, upload checkpoints, and deploy local inference with an NVIDIA Jetson AGX Orin (policy server) and an Intel NUC (2× RealSense, 2× Trossen WXAI arms) on the same LAN.

## 1) Prerequisites
- Python via `uv`: `uv python install && uv sync --all-extras --dev`
- FFmpeg (for tests/media): `sudo apt-get update && sudo apt-get install -y ffmpeg`
- WANDB: `wandb login` (set `WANDB_PROJECT=openpi` or a project of your choice)
- Dataset: dataset should be in LeRobot format. If your keys differ from ALOHA naming (`cam_high`, `cam_left_wrist`, `cam_right_wrist`, `observation.state`, `action`), adapt the repack mapping accordingly.

## 2) Convert Dataset to ALOHA Format
Run the converter with Tatbot-specific camera mapping and episode stroke images:

```bash
# realsense1 -> cam_left_wrist, realsense2 -> cam_right_wrist
# cam_high is derived per-episode from episode_{idx}/stroke_{l|r}.png and duplicated across frames.
uv run python scripts/convert_tatbot_to_lerobot_aloha.py \
  --src-repo-id tatbot/wow-2025y-09m-10d-15h-34m-08s \
  --dst-repo-id <your-hf-username>/tatbot_wow_pi05_aloha \
  --push-to-hub
```

Then use the new repo id in training below.

## 3) Create a TrainConfig for Tatbot
Add a new entry to `src/openpi/training/config.py` near other fine‑tuning configs:

```python
TrainConfig(
    name="pi05_tatbot",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,          # set to your action dimension
        action_horizon=16,      # chunk length used during inference
    ),
    data=LeRobotAlohaDataConfig(
        repo_id="<your-hf-username>/tatbot_wow_pi05_aloha",
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
            asset_id="trossen",
        ),
        # Mapping aligns to ALOHA: cam_high + wrist cameras; prompt from task_index
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=50_000,
    batch_size=256,            # fits H100‑80GB; adjust if OOM
    log_interval=100,
    save_interval=2000,
)
```

Notes
- Adjust `action_dim`/`action_horizon` to your robot (e.g., joint dims + grippers).
- Keep `asset_id="trossen"` for WXAI arms to reuse normalization stats.

## 4) Train on Datacrunch H100 SXM5 80GB
On the cloud VM:

```bash
# System deps
sudo apt-get update && sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev

# Env + deps
uv python install
uv sync --all-extras --dev
wandb login  # ensure WANDB_PROJECT is set

# Single run (override basics via Tyro CLI)
uv run python scripts/train.py pi05_tatbot \
  --exp_name=tatbot_pi05_h100 \
  --checkpoint_base_dir=/workspace/checkpoints \
  --assets_base_dir=/workspace/assets \
  --batch_size=256 \
  --lr_schedule.peak_lr=5e-5 \
  --lr_schedule.warmup_steps=2000 \
  --ema_decay=0.999
```

## 5) WANDB Hyperparameter Sweep
Create `sweeps/tatbot_pi05.yaml`:

```yaml
method: bayes
metric: { name: loss, goal: minimize }
parameters:
  batch_size: { values: [128, 192, 256] }
  lr_schedule.peak_lr: { values: [2.5e-5, 5e-5, 7.5e-5] }
  lr_schedule.warmup_steps: { values: [1000, 2000, 4000] }
  ema_decay: { values: [0.99, 0.995, 0.999] }
command:
  - uv
  - run
  - python
  - ${program}
  - pi05_tatbot
  - --exp_name
  - sweep-${wandb.run.id}
  - ${args}
program: scripts/train.py
```

Run the sweep:

```bash
wandb sweep sweeps/tatbot_pi05.yaml   # returns a SWEEP_ID
wandb agent $WANDB_ENTITY/$WANDB_PROJECT/SWEEP_ID
```

## 6) Upload Checkpoints
- To Hugging Face Hub (recommended for sharing):

```bash
pip install huggingface_hub git-lfs
huggingface-cli login
huggingface-cli repo create tatbot/pi05_tatbot --type=model
cd checkpoints/pi05_tatbot/exp
git init && git lfs install
git remote add origin https://huggingface.co/tatbot/pi05_tatbot
git add . && git commit -m "Add pi05 tatbot checkpoints"
git push -u origin HEAD
```

- Or as WANDB Artifacts:
```bash
wandb artifact put checkpoints/pi05_tatbot/exp --name tatbot-pi05-checkpoints
```

## 7) Remote Inference on LAN
- Policy server on AGX Orin:
```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_tatbot \
  --policy.dir=/workspace/checkpoints/pi05_tatbot/exp/50000 \
  --port=8000
```

- Robot client on Intel NUC (ROS + RealSense + WXAI arms):
  - Follow `examples/aloha_real/README.md` (update RealSense serials, bring up ROS nodes).
  - Install client and run:

```bash
uv venv --python 3.10 examples/aloha_real/.venv && source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client
python -m examples.aloha_real.main --host <AGX_ORIN_IP> --port 8000
```

Tips
- Ensure both machines can ping each other; open port 8000.
- Match `action_horizon` between training and client (`examples/aloha_real/main.py`).
- For non‑ALOHA key naming, adjust `repack_transforms` in your TrainConfig to map dataset keys to the expected fields.
