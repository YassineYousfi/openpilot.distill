# Reinforcement learning

## Interactive

Terminal 1:

```bash
uv run python -m rl.server
```

Terminal 2:

```bash
uv run python -m rl.env --actor "supercombo|wasd" --cache-dir outputs/rl/cache
```

`supercombo` uses `models/big_driving_supercombo.onnx` through ONNX Runtime by default. Pass a supervised Torch
state dict to use the Torch runtime instead:

```bash
uv run python -m rl.env --actor supercombo --model /path/to/supercombo.pt
```

## Training

With the server running:

```bash
uv run python -m rl.train --model /path/to/supercombo.pt --steps 10
```

Try the RL trained policy

```bash
uv run python -m rl.env --actor supercombo --model /path/to/supercombo.pt --on-policy /path/to/on_policy.pt
```