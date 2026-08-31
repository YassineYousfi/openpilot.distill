# Supervised learning

`wandb login` or set `WANDB_MODE=offline`.
## TODO Download script


## Training
```bash
uv run torchrun --standalone --nproc-per-node=1 -m supervised.train --steps 5000 --batch-size 8 --workers 1
```

## Watch it train