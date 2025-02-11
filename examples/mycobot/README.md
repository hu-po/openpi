
on policy server (ubuntu computer with 3090)

```bash
uv run scripts/serve_policy.py --env MCB
```

on robot (raspberry pi with mycobot280 and usb camera on /dev/video0)

```bash
uv run examples/mcb/main.py
```
