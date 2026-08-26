# Reinforcement learning

## Interactive

Terminal 1:

```bash
uv run python -m rl.server
```

Terminal 2:

```bash
uv run python -m rl.env --actor "supercombo|wasd"
```

## Training

With the server running:

```bash
uv run python -m rl.train -n 10
```
