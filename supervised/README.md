# Supervised learning

`wandb login` or set `WANDB_MODE=offline`.

## Download the teacher model

```bash
OPENPILOT_SHA=084747c75d2cbd23af65ab7a9e770bbd7b98bac9
mkdir -p models
curl --fail --location --continue-at - \
  "https://media.githubusercontent.com/media/commaai/openpilot/${OPENPILOT_SHA}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx" \
  --output models/big_driving_supercombo.onnx
```

## Training

```bash
uv run torchrun --standalone --nproc-per-node=1 -m supervised.train --steps 5000 --batch-size 8 --workers 1
```
