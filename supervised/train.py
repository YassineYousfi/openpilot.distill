"""Train Supercombo on logged comma1M segments."""

import argparse
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb

from .dataset import Comma1MDataset, DatasetConfig, prefill_supercombo_batch
from .model import Supercombo
from .model_for_inference import InputQueues, ORTSupercomboForInference


ROOT = Path(__file__).resolve().parents[1]
TARGET_OUTPUTS = ("wide_from_device_euler", "pose", "plan", "action")
LOSS_WEIGHTS = {"gt/imgs": 100.0, "distill/lead": 0.05, "distill/hidden_state": 10.0}


def loss_fn(predictions, targets, teacher_targets):
    predicted_images = predictions["imgs"][:, -1].float() / 127.5 - 1
    target_images = targets["imgs"].permute(0, 3, 1, 2).float() / 127.5 - 1
    target_losses = {"imgs": F.mse_loss(predicted_images, target_images)}
    target_losses |= {
        name: F.mse_loss(
            predictions[name][..., : targets[name].shape[-1]].float(),
            targets[name].float(),
        )
        for name in TARGET_OUTPUTS
    }
    distill_losses = {
        name: F.mse_loss(
            (predictions["hidden_state"].mean(-2) if name == "hidden_state" else predictions[name]).float(),
            target.float(),
        )
        for name, target in teacher_targets.items()
    }
    target_loss = sum(LOSS_WEIGHTS.get(f"gt/{name}", 1.0) * value for name, value in target_losses.items())
    distill_loss = sum(LOSS_WEIGHTS.get(f"distill/{name}", 1.0) * value for name, value in distill_losses.items())
    return target_loss + distill_loss, {
        "gt": target_loss,
        "distill": distill_loss,
        **{f"gt/{name}": value for name, value in target_losses.items()},
        **{f"distill/{name}": value for name, value in distill_losses.items()},
    }


def _identity(value):
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT), help="comma1M dataset root")
    parser.add_argument(
        "--teacher",
        default=str(ROOT / "models/big_driving_supercombo.onnx"),
        help="teacher Supercombo ONNX model",
    )
    parser.add_argument("--steps", type=int, default=1, help="training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="samples per training batch")
    parser.add_argument("--workers", type=int, default=0, help="data-loader workers")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "outputs/supervised/supercombo.pt",
        help="checkpoint output path",
    )
    args = parser.parse_args()

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.set_float32_matmul_precision("high")

    train_dataset = Comma1MDataset(
        DatasetConfig(args.dataset, batch_size=args.batch_size),
        local_rank=local_rank,
        global_rank=rank,
        global_world_size=world_size,
    )
    batch_iterator = iter(
        DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=args.workers,
            collate_fn=_identity,
            multiprocessing_context="spawn" if args.workers else None,
            persistent_workers=bool(args.workers),
        )
    )

    raw_model = Supercombo().to(device)
    model = DDP(raw_model, device_ids=[local_rank]) if distributed else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-3,
        fused=True,
    )

    if is_main:
        print(f"student: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M parameters")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    teacher = ORTSupercomboForInference.from_supercombo(
        args.teacher,
        providers=[("CUDAExecutionProvider", {"device_id": local_rank}), "CPUExecutionProvider"],
    )
    run = (
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "openpilot.distill.supervised"),
            dir=args.output.parent,
            config={**vars(args), "world_size": world_size, "loss_weights": LOSS_WEIGHTS},
        )
        if is_main
        else None
    )

    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()
        inputs, targets = next(batch_iterator)
        while not len(inputs["img"]):
            inputs, targets = next(batch_iterator)

        queues = InputQueues.from_model(teacher, batch_size=len(inputs["img"]))
        teacher_predictions = prefill_supercombo_batch(teacher, queues, inputs)
        teacher_targets = {name: torch.from_numpy(value).to(device) for name, value in teacher_predictions.items()}
        inputs = {name: torch.from_numpy(value).to(device) for name, value in inputs.items()}
        targets = {name: torch.from_numpy(value).to(device) for name, value in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(inputs)
            loss, loss_metrics = loss_fn(predictions, targets, teacher_targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        if is_main:
            torch.cuda.synchronize()
            metrics = {
                "loss": loss.item(),
                **{name: value.item() for name, value in loss_metrics.items()},
                "grad_norm": grad_norm.item(),
                "step_time": time.perf_counter() - step_start,
            }
            if step == 1 or step % 100 == 0:
                target_images = targets["imgs"][0].cpu().numpy()
                predicted_images = (
                    predictions["imgs"][0, -1]
                    .detach()
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                metrics |= {
                    "imgs/gt_narrow": wandb.Image(np.ascontiguousarray(target_images[..., :3])),
                    "imgs/pred_narrow": wandb.Image(np.ascontiguousarray(predicted_images[..., :3])),
                    "imgs/gt_wide": wandb.Image(np.ascontiguousarray(target_images[..., 3:])),
                    "imgs/pred_wide": wandb.Image(np.ascontiguousarray(predicted_images[..., 3:])),
                }
            print(
                f"step {step}/{args.steps}: loss={metrics['loss']:.6g} "
                f"gt={metrics['gt']:.6g} distill={metrics['distill']:.6g} "
                f"time={metrics['step_time']:.2f}s"
            )
            run.log(metrics, step=step)

    if is_main:
        torch.save(
            {name: value.detach().cpu() for name, value in raw_model.state_dict().items()},
            args.output,
        )
        run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
