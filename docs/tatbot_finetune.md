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

Add a new entry to `src/openpi/training/config.py` near other fine‑tuning configs

Notes
- Adjust `action_dim`/`action_horizon` to your robot (e.g., joint dims + grippers).
- Keep `asset_id="trossen"` for WXAI arms to reuse normalization stats.

## 4) Train on Datacrunch H100 SXM5 80GB

On the cloud VM:

```bash
# ssh into node
ssh root@31.22.104.62

# clone openpi
git clone https://github.com/hu-po/openpi.git && cd openpi

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install
uv sync --all-extras --dev
source .venv/bin/activate

# make sure to replace the lerobot version in the openpi/pyproject.toml file with the latest version
# FOR INFERENCE:
> lerobot = { git = "https://github.com/hu-po/lerobot", rev = "main" }
# FOR TRAINING:
> lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" }

# install wandb
export WANDB_PROJECT="openpi-full-H100"
export WANDB_ENTITY="hug"
wandb login

# setup huggingface
huggingface-cli login

# memory allocation optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# create the sweep and start the agent
wandb sweep sweeps/tatbot_pi05_full.yaml
wandb agent $WANDB_ENTITY/$WANDB_PROJECT/SWEEP_ID

# to reset the cloud instance:
git pull
rm -rf wandb/
```

Upload trained checkpoints to Hugging Face Hub (recommended for sharing):

```bash
pip install huggingface_hub git-lfs
huggingface-cli login
huggingface-cli repo create tatbot/pi05_full_tatbot_finetune --type=model
cd checkpoints/pi05_tatbot/exp
git init && git lfs install
git remote add origin https://huggingface.co/tatbot/pi05_tatbot
git add . && git commit -m "Add pi05 tatbot checkpoints"
git push -u origin HEAD
```

## Remote Inference on Tatbot (No ROS)

- Policy server on oop (3090):
- Robot client on hog (Intel NUC)

Notes
- Inputs for Tatbot pi‑0.5: `cam_high` (required), `cam_left_wrist`, `cam_right_wrist`, 14‑D state, `prompt`.
- Client pre‑resizes images to 224; server performs the final normalization and any additional resizing.

Tips
- Ensure both machines can ping each other; open port 8000.
- Match `action_horizon` between training and client (`examples/aloha_real/main.py`).
- For non‑ALOHA key naming, adjust `repack_transforms` in your TrainConfig to map dataset keys to the expected fields.

## Full vs LoRA Finetuning (pi‑0.5)
- Full finetune: Recommended when you have ample compute (e.g., H100). Default `pi05_*` configs in this repo use full finetuning with EMA; best performance and flexibility.
- LoRA finetune: Recommended for low‑memory setups or fast iteration. Enable via `Pi0Config(..., paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")` and set `freeze_filter=Pi0Config(...).get_freeze_filter()`; disable EMA.
- Practical guidance: Keep `prompt_from_task=True`, map images to `cam_high` + wrist views, and compute dataset‑specific norm stats.

## Local LoRA Sweep (Dev Box)
- Use the provided low‑mem config `pi05_tatbot_low_mem` for quick local runs (batch=1, smaller `action_horizon`/`max_token_len`).
- Recommended env to reduce VRAM preallocation:
  - `export XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `export XLA_PYTHON_CLIENT_ALLOCATOR=platform`
- Run a small WANDB sweep locally (edits allowed):

```bash
wandb sweep sweeps/tatbot_pi05_lora.yaml   # shows a SWEEP_ID
wandb agent $WANDB_ENTITY/$WANDB_PROJECT/SWEEP_ID
```

This sweep varies LR, warmup, clip norm, and small sequence sizes for memory safety. For cloud H100 runs, switch back to `pi05_tatbot` with larger sequences and batch sizes, and sweep LR/warmup/EMA.

Notes on sweep checkpoints
- The sweep command uses `--exp_name=sweep-${wandb.run.id}`.
- Train script expands that placeholder (or appends a short random suffix when needed) so each run writes to a unique checkpoint directory, avoiding collisions without deleting prior runs.

# Experiment Logs

## LoRA finetune on local 3090

experiment logging
- https://wandb.ai/hug/openpi-lora-3090

analysis
- https://grok.com/share/bGVnYWN5LWNvcHk%3D_42092685-77f0-4153-a4ea-01e02687741f
- https://chatgpt.com/share/68c97743-87c0-8009-a768-f6eed62150f2
- https://g.co/gemini/share/8b739714e005

models agree that best run to test is `sweep-kc83rnit`

run inference server on oop (3090)

```bash
cd ~/openpi
source .venv/bin/activate
uv run scripts/serve_policy.py policy:checkpoint \
--policy.config=pi05_tatbot_low_mem \
--policy.dir="$(pwd)/checkpoints/pi05_tatbot_low_mem/sweep-kc83rnit/199"
```

run robot client on hog (intel nuc)

```bash
cd ~/openpi
source .venv/bin/activate
uv pip install -e packages/openpi-client
uv pip install "lerobot[tatbot,intelrealsense] @ git+https://github.com/hu-po/lerobot.git@main"
uv run python examples/tatbot/infer.py
```

policy inference works, but arms mostly just kinda move slowly around and eventually collide with each other and the environment, leading to an estop.