# MyCobot Drawing

this task uses a MyCobot280pi 6-DOF robot arm to draw on a Wacom Intuos Pro tablet.
the goal is to draw a target black and white image.
target images are generated.
the reward is the L2 distance between the robot's drawing and the target image.
each episode consists of some maximum number of robot steps.

## Setup

### Ubuntu PC (policy server)

###  (robot computer)

the robot computer is the raspberry pi attached to the MyCobot280pi

`Linux er 5.4.0-1073-raspi aarch64 GNU/Linux`

this is a cheap robot, so when the servos are active they will produce a high pitched noise.
try to not leave the robot in this state, reset the robot (release servos) if you aren't using it.

```bash
uv run examples/mycobot/hardware.py -c release
```

plug in a usb camera to the robot computer, verify it is available:

```bash
ls /dev/video*
```

plug in the tablet via usb-c to the robot computer, verify it is available:

```bash
ls /dev/input/
sudo apt-get install evtest
sudo evtest
uv pip install evdev
```

you will need to modify permissions to access the tablet:

```bash
sudo usermod -a -G input $USER
newgrp input
```

## Record Baseline Dataset

the baseline dataset will use the robot's "free drag" mode to scribble on the tablet.
this data will be used to finetune the pi0 base policy so that it becomes familiar with the embodiment.
datasets will be stored on huggingface.
login to huggingface, create a read and write token:

```bash
uv pip install -U "huggingface_hub[cli]"
uv run huggingface-cli login
```

to record a dataset of mycobot trajectories:

```bash
PYTHONPATH=$PYTHONPATH:. uv run examples/mycobot/record_dataset.py
```

calculate norm stats:

```bash
uv run examples/mycobot/calculate_norm_stats.py --repo-id oop/mycobot-dataset
```

## Finetune

to finetune the pi0 base policy

## Inference

on policy server

```bash
uv run scripts/serve_policy.py --env MCB
```

on robot computer

```bash
uv run examples/mcb/main.py
```

## Development

development is being done on a branch of openpi:

https://github.com/hu-po/openpi

to see the latest changes:

https://github.com/Physical-Intelligence/openpi/compare/main...hu-po:openpi:main