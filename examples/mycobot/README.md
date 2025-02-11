
on policy server (ubuntu computer with 3090)

```bash
uv run scripts/serve_policy.py --env MCB
```

on robot (raspberry pi with mycobot280 and usb camera on /dev/video0)

```bash
uv run examples/mcb/main.py
```

to record a dataset of mycobot trajectories:

```bash
uv run examples/mycobot/record_dataset.py --repo-id oop/mycobot-dataset
```

calculate norm stats:

```bash
uv run examples/mycobot/calculate_norm_stats.py --repo-id oop/mycobot-dataset
```
