# Supervised learning

`wandb login` or set `WANDB_MODE=offline`.

read the config defaults in `torchtitan/torchtitan/experiments/path/config_registry.py`

Fine-tune the Supercombo model on your segment using MSE loss and localized targets.

```
cd torchtitan

UUID="$(cat /proc/sys/kernel/random/uuid)"
export UUID

WANDB_PROJECT=openpilot.distill.supervised \
COMMA1M_DATASET_PATH=/home/batman/openpilot.distill \
CUDA_VISIBLE_DEVICES=0 \
NGPU=1 \
MODULE=path \
CONFIG=convnext_xxlarge_comma1m \
./run_train.sh \
  --checkpoint.initial-load-path=/home/batman/openpilot.distill/models/supercombo \
  --dump-folder=/home/batman/openpilot.distill/outputs/supervised/$UUID
```