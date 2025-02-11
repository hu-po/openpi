
on policy server (ubuntu computer with 3090)

```bash
uv run scripts/serve_policy.py --env MCB
```

on robot (raspberry pi with mycobot280 and usb camera on /dev/video0)

```bash
uv run examples/mcb/main.py
```

login to huggingface, create a read and write token:

```bash
uv pip install -U "huggingface_hub[cli]"
uv run huggingface-cli login
```


to record a dataset of mycobot trajectories:

```bash
PYTHONPATH=$PYTHONPATH:. uv run examples/mycobot/record_dataset.py --repo-id hu-po/mycobot-test
```

if the robot is on after a script, you can reset it with:

```bash
PYTHONPATH=$PYTHONPATH:. uv run examples/mycobot/reset_robot.py
```

calculate norm stats:

```bash
uv run examples/mycobot/calculate_norm_stats.py --repo-id oop/mycobot-dataset
```
