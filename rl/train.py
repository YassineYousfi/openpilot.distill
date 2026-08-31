import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED, get_accel_from_plan, get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

from rl.actor import SupercomboActor
from rl.env import ENCODE_BATCH_SIZE, FRAME_SKIP, ROLLOUT_FRAMES, SEGMENT, Episode, Env, rollout
from rl.server import SERVER_URL, RuntimeClient
from supervised.model import ModelInputs


ROOT = Path(__file__).resolve().parents[1]
T_IDXS = np.asarray(ModelConstants.T_IDXS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--server", default=SERVER_URL, help="resident Runtime server URL")
    parser.add_argument("--segment", default=SEGMENT, help="segment ID under data/")
    parser.add_argument("--sampling-steps", type=int, default=30, help="world-model diffusion steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="world-model guidance scale")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU used for decoding and training")
    parser.add_argument("--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE, help="VAE encoding batch size")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/rl/cache", help="optional latent cache directory")
    parser.add_argument("--model", type=Path, required=True, help="supervised Torch Supercombo state dict")
    parser.add_argument("--on-policy", type=Path, help="on-policy state dict to resume from")
    parser.add_argument("--rollout-frames", type=int, help="generated 5 Hz frames per rollout")
    parser.add_argument("--steps", type=int, default=10, help="training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="samples per training batch")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="AdamW learning rate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "outputs/rl/on_policy.pt",
        help="checkpoint output path",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", args.gpu_id) if torch.cuda.is_available() else torch.device("cpu")
    runtime = RuntimeClient(args.server, args.sampling_steps, args.cfg)
    print(f"world-model server ready ({runtime.context_frames} context frames)")
    max_rollout_frames = ROLLOUT_FRAMES - runtime.history_frames - runtime.future_frames
    rollout_frames = max_rollout_frames if args.rollout_frames is None else args.rollout_frames
    if not 2 <= rollout_frames <= max_rollout_frames:
        parser.error(f"rollout frames must be in [2, {max_rollout_frames}]")
    episode = Episode(
        runtime,
        args.segment,
        gpu_id=args.gpu_id,
        encode_batch_size=args.encode_batch_size,
        cache_dir=args.cache_dir,
    )
    last_start = len(episode.latents) - runtime.future_frames - rollout_frames
    actor = SupercomboActor.from_torch_checkpoint(args.model, on_policy=args.on_policy, device=device)
    supercombo = actor.model.model
    supercombo.requires_grad_(False).eval()
    model = supercombo.on_policy_temporal
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()
        actor.reset()
        start_index = int(np.random.randint(runtime.history_frames, last_start + 1))
        end_index = start_index + rollout_frames
        rollout_result = rollout(
            Env(runtime, episode, start_index * FRAME_SKIP, end_index * FRAME_SKIP),
            actor,
        )
        rollout_inputs: dict[str, list[torch.Tensor]] = {
            ModelInputs.FEATURES: [],
            ModelInputs.DESIRE: [],
            ModelInputs.TRAFFIC: [],
            ModelInputs.ACTION_T: [],
        }
        target_actions: list[torch.Tensor] = []
        for model_input, model_output, plan in zip(
            rollout_result.model_inputs,
            rollout_result.model_outputs,
            rollout_result.worldmodel_outputs["plan"],
            strict=True,
        ):
            if model_input is None or model_output is None:
                continue
            rollout_inputs[ModelInputs.FEATURES].append(
                torch.cat((model_input[ModelInputs.FEATURES], model_output["hidden_state"][:, None]), dim=1)
            )
            rollout_inputs[ModelInputs.DESIRE].append(model_input[ModelInputs.DESIRE])
            rollout_inputs[ModelInputs.TRAFFIC].append(model_input[ModelInputs.TRAFFIC])
            rollout_inputs[ModelInputs.ACTION_T].append(model_input[ModelInputs.ACTION_T])

            plan = np.asarray(plan)
            action_t = model_input[ModelInputs.ACTION_T][0, -1]
            speed = max(float(plan[0, Plan.VELOCITY.start]), MIN_SPEED)
            curvature = get_curvature_from_plan(
                plan[:, Plan.T_FROM_CURRENT_EULER.start + 2],
                plan[:, Plan.ORIENTATION_RATE.start + 2],
                T_IDXS,
                speed,
                float(action_t[0]),
            )
            accel = get_accel_from_plan(
                plan[:, Plan.VELOCITY.start],
                plan[:, Plan.ACCELERATION.start],
                T_IDXS,
                action_t=float(action_t[1]),
            )
            target_action = torch.tensor((curvature * speed**2, accel), dtype=torch.float32)
            if not torch.isfinite(target_action).all():
                raise ValueError(f"world-model plan produced a non-finite target: {target_action.tolist()}")
            target_actions.append(target_action)

        inputs = {name: torch.cat(values) for name, values in rollout_inputs.items()}
        targets = torch.stack(target_actions)

        model.requires_grad_(True).train()
        total_loss = 0.0
        for batch_start in range(0, len(targets), args.batch_size):
            batch_inputs = {
                name: value[batch_start : batch_start + args.batch_size].to(device) for name, value in inputs.items()
            }
            batch_targets = targets[batch_start : batch_start + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                batch_inputs[ModelInputs.FEATURES],
                batch_inputs[ModelInputs.DESIRE],
                batch_inputs[ModelInputs.TRAFFIC],
                batch_inputs[ModelInputs.ACTION_T],
            )["action"][:, -1, :2]
            loss = F.mse_loss(predictions.float(), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(batch_targets)
        model.requires_grad_(False).eval()

        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            args.output,
        )
        mean_loss = total_loss / len(targets)
        print(
            f"step {step}/{args.steps}: loss={mean_loss:.6g} "
            f"frames={start_index * FRAME_SKIP}:{end_index * FRAME_SKIP} "
            f"time={time.perf_counter() - step_start:.2f}s"
        )


if __name__ == "__main__":
    main()
