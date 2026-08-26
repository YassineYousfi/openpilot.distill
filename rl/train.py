#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED, get_accel_from_plan, get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import Plan
from torchtitan.experiments.path.model_constants import ModelInputs, T_IDXS

from rl.actor import SupercomboActor
from rl.env import FRAME_SKIP, ROLLOUT_FRAMES, SEGMENT, Episode, Env, rollout
from rl.server import SERVER_URL, RuntimeClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--server", default=SERVER_URL, help="resident Runtime server URL")
    parser.add_argument("--segment", default=SEGMENT, help="segment ID under data/")
    parser.add_argument("--sampling-steps", type=int, default=30, help="world-model diffusion steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="world-model guidance scale")
    parser.add_argument("--on-policy", type=Path, help="on-policy state dict to resume from")
    parser.add_argument("--rollout-frames", type=int, help="generated 5 Hz frames per rollout")
    parser.add_argument("-n", "--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/rl/on_policy.pt"))
    args = parser.parse_args()

    runtime = RuntimeClient(args.server, args.sampling_steps, args.cfg)
    print(f"world-model server ready ({runtime.context_frames} context frames)")
    max_rollout_frames = ROLLOUT_FRAMES - runtime.history_frames - runtime.future_frames
    rollout_frames = max_rollout_frames if args.rollout_frames is None else args.rollout_frames
    if not 2 <= rollout_frames <= max_rollout_frames:
        parser.error(f"rollout frames must be in [2, {max_rollout_frames}]")
    episode = Episode(runtime, args.segment)
    last_start = len(episode.latents) - runtime.future_frames - rollout_frames
    if last_start < runtime.history_frames:
        raise ValueError("episode is too short for this rollout")
    actor = SupercomboActor(args.on_policy)
    actor.model.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(actor.model.on_policy.parameters(), lr=args.learning_rate, weight_decay=0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, args.iterations + 1):
        actor.reset()
        start = int(np.random.randint(runtime.history_frames, last_start + 1))
        end = start + rollout_frames
        result = rollout(Env(runtime, episode, start * FRAME_SKIP, end * FRAME_SKIP), actor)
        replay: dict[str, list[torch.Tensor]] = {
            ModelInputs.FEATURES: [],
            ModelInputs.DESIRE: [],
            ModelInputs.TRAFFIC: [],
            ModelInputs.ACTION_T: [],
        }
        targets = []
        for model_input, model_output, plan in zip(
            result.model_inputs,
            result.model_outputs,
            result.worldmodel_outputs["plan"],
            strict=True,
        ):
            if model_input is None or model_output is None:
                continue
            replay[ModelInputs.FEATURES].append(
                torch.cat((model_input["features_buffer"], model_output["vision_features"][:, None]), dim=1)
            )
            replay[ModelInputs.DESIRE].append(model_input[ModelInputs.DESIRE])
            replay[ModelInputs.TRAFFIC].append(model_input[ModelInputs.TRAFFIC][:, None])
            replay[ModelInputs.ACTION_T].append(model_input[ModelInputs.ACTION_T][:, None])

            plan = np.asarray(plan)
            if plan.shape != (len(T_IDXS), 15):
                raise ValueError(f"expected a {len(T_IDXS)}x15 plan, got {plan.shape}")
            action_t = model_input[ModelInputs.ACTION_T][0]
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
            target = torch.tensor((curvature * speed**2, accel), dtype=torch.float32)
            if not torch.isfinite(target).all():
                raise ValueError(f"world-model plan produced a non-finite target: {target.tolist()}")
            targets.append(target)

        if not targets:
            raise RuntimeError("rollout has no trainable samples; Supercombo needs at least two environment steps")
        inputs = {name: torch.cat(values) for name, values in replay.items()}
        targets_tensor = torch.stack(targets)

        actor.model.on_policy.requires_grad_(True).train()
        total_loss = 0.0
        for batch_start in range(0, len(targets_tensor), args.batch_size):
            batch = {
                name: value[batch_start : batch_start + args.batch_size].to(actor.device)
                for name, value in inputs.items()
            }
            target = targets_tensor[batch_start : batch_start + args.batch_size].to(actor.device)
            optimizer.zero_grad(set_to_none=True)
            action = actor.model.on_policy(
                batch[ModelInputs.FEATURES],
                batch[ModelInputs.DESIRE],
                batch[ModelInputs.TRAFFIC],
                batch[ModelInputs.ACTION_T],
            )["action"][:, :2]
            loss = F.mse_loss(action.float(), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(target)
        actor.model.on_policy.requires_grad_(False).eval()

        torch.save(
            {name: value.detach().cpu() for name, value in actor.model.on_policy.state_dict().items()},
            args.output,
        )
        print(
            f"iteration {iteration}/{args.iterations}: "
            f"frames={start * FRAME_SKIP}:{end * FRAME_SKIP}, loss={total_loss / len(targets_tensor):.6f}"
        )


if __name__ == "__main__":
    main()
