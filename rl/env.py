#!/usr/bin/env python3
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from openpilot.cereal import log
from openpilot.common.transformations.camera import DEVICE_CAMERAS, img_from_device
from openpilot.common.transformations.model import medmodel_intrinsics, sbigmodel_intrinsics
from safetensors.numpy import load_file

from helpers.video_helpers import calibration_view_eulers, calibration_warp_matrix, decode_frames
from rl.actor import SUPERCOMBO, ConstantActor, SupercomboActor, WASDActor
from rl.server import SERVER_URL, RuntimeClient


ROOT = Path(__file__).resolve().parents[1]
SEGMENT = "2333e255f1de1fe83ea5975b24a3cc67"
CACHE_DIR = ROOT / "outputs/rl/cache"
FPS = 5
ROLLOUT_FRAMES = 50
FRAME_SKIP = 20 // FPS
ENCODE_BATCH_SIZE = 32
CAMERAS = DEVICE_CAMERAS[("mici", "os04c10")]
FCAM_INTRINSICS = medmodel_intrinsics * np.array([[0.5], [0.5], [1.0]])


@dataclass(slots=True)
class Rollout:
    worldmodel_outputs: dict[str, list[torch.Tensor | np.ndarray]]
    model_inputs: list[dict[str, torch.Tensor] | None]
    model_outputs: list[dict[str, torch.Tensor] | None]


def draw_worldmodel_outputs(frame: np.ndarray, plan: np.ndarray) -> np.ndarray:
    rendered = frame.copy()
    fcam = rendered[..., :3].copy()
    path = plan[:, :3].copy()
    path[:, 2] += 1.22
    path = path[path[:, 0] > 0]
    if len(path) > 1:
        points = img_from_device(path)
        points = np.column_stack((points, np.ones(len(points)))) @ FCAM_INTRINSICS.T
        points = points[:, :2]
        valid = np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1) & (points < frame.shape[1::-1]).all(axis=1)
        if valid.sum() > 1:
            cv2.polylines(fcam, [points[valid].astype(np.int32)], False, (255, 80, 20), 2, cv2.LINE_AA)
    rendered[..., :3] = fcam
    return rendered


def _load_cached_latents(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    try:
        return torch.from_numpy(np.load(path, allow_pickle=False))
    except (EOFError, OSError, ValueError):
        return None


def _save_cached_latents(path: Path, latents: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, latents.detach().cpu().float().numpy())
    temporary.replace(path)


class Episode:
    """Encoded segment data shared across rollouts."""

    def __init__(
        self,
        runtime: RuntimeClient,
        segment: str = SEGMENT,
        gpu_id: int = 0,
        encode_batch_size: int = ENCODE_BATCH_SIZE,
        cache_dir: Path | None = None,
    ):
        if encode_batch_size <= 0:
            raise ValueError("encode_batch_size must be positive")
        directory = ROOT / "data" / segment
        files = {
            name: directory / name
            for name in ("fcamera.hevc", "ecamera.hevc", "frame_info.safetensors", "localizer.safetensors")
        }
        frame_info = load_file(files["frame_info.safetensors"])
        if int(frame_info["device_type"].item()) != log.InitData.DeviceType.mici:
            raise ValueError("the v0 environment expects a comma four segment")
        localizer = load_file(files["localizer.safetensors"])
        calibration = localizer["rpy"]
        self.speeds = np.linalg.norm(localizer["frame_states"][::FRAME_SKIP, 7:10], axis=1)

        cache_path = None
        if cache_dir is not None:
            vae_dir = runtime.vae.replace("/", "--")
            cache_path = cache_dir / vae_dir / f"{segment}.npy"
            cached = _load_cached_latents(cache_path)
            if cached is not None and len(cached) == len(self.speeds):
                self.latents = cached
                print(f"loaded latent cache: {cache_path}")
                return

        view_euler = calibration_view_eulers(calibration)
        views = []
        for camera, source_k, target_k in (
            ("fcamera", CAMERAS.narrow_road.intrinsics, medmodel_intrinsics),
            ("ecamera", CAMERAS.wide_road.intrinsics, sbigmodel_intrinsics),
        ):
            matrix = calibration_warp_matrix(source_k, target_k, view_euler)
            matrix[:2] *= 0.5
            index = frame_info[f"{camera}/index"]
            wanted = set(range(0, len(index) - 1, FRAME_SKIP))
            decoded = decode_frames(files[f"{camera}.hevc"], index, wanted, gpu_id=gpu_id)
            frames = [
                cv2.warpPerspective(decoded[frame_index], matrix, (256, 128), borderMode=cv2.BORDER_REPLICATE)
                for frame_index in sorted(wanted)
            ]
            views.append(np.stack(frames))

        frames = np.concatenate(views, axis=-1)
        self.latents = torch.cat(
            [runtime.encode(frames[start : start + encode_batch_size]) for start in range(0, len(frames), encode_batch_size)]
        )
        if cache_path is not None:
            _save_cached_latents(cache_path, self.latents)
            print(f"wrote latent cache: {cache_path}")


class Physics:
    def __init__(self, speed: float):
        self.speed = speed

    def step(self, curvature: float, accel: float) -> tuple[float, float, float]:
        self.speed = max(self.speed + accel / FPS, 0.0)
        yaw = self.speed * curvature / FPS
        distance = self.speed / FPS
        return distance * np.cos(yaw / 2), distance * np.sin(yaw / 2), yaw


class Env:
    """Mutable rollout state backed by the resident Runtime server."""

    def __init__(
        self,
        runtime: RuntimeClient,
        episode: Episode,
        start: int | None = None,
        end: int | None = None,
    ):
        start = runtime.history_frames * FRAME_SKIP if start is None else start
        end = (
            (start // FRAME_SKIP + ROLLOUT_FRAMES - runtime.history_frames - runtime.future_frames) * FRAME_SKIP
            if end is None
            else end
        )
        if start % FRAME_SKIP or end % FRAME_SKIP:
            raise ValueError("start and end must align to the 5 Hz model frames")
        if not runtime.history_frames * FRAME_SKIP <= start < end:
            raise ValueError("start must include model history and precede end")

        start_index = start // FRAME_SKIP
        end_index = end // FRAME_SKIP
        if end_index + runtime.future_frames > len(episode.latents):
            raise ValueError("future frames exceed the episode")

        self.runtime = runtime
        self.start = self.target = start
        self.end = end
        self.initial_frame = runtime.decode(episode.latents[start_index - 1 : start_index])[0]
        self.context = episode.latents[start_index - runtime.history_frames : start_index].clone()
        self.future = episode.latents[end_index : end_index + runtime.future_frames].clone()
        future_start = runtime.history_frames + end_index - start_index
        self.future_fidxs = torch.arange(future_start, future_start + runtime.future_frames)
        self.physics = Physics(float(episode.speeds[start_index - 1]))

    @property
    def done(self) -> bool:
        return self.target >= self.end

    @torch.inference_mode()
    def step(self, curvature: float, accel: float) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Advance one 0.2 s frame; curvature is 1/m and accel is m/s^2."""
        if self.done:
            raise StopIteration

        dx, dy, yaw = self.physics.step(curvature, accel)
        latents = torch.cat((self.future, self.context, torch.randn_like(self.context[:1])))[None]
        position = latents.new_zeros((*latents.shape[:2], 3))
        euler = torch.zeros_like(position)
        pose_mask = torch.ones(latents.shape[:2], dtype=torch.int64)
        position[0, -1, 0] = dx
        position[0, -1, 1] = dy
        euler[0, -1, 2] = yaw
        pose_mask[0, -1] = 0
        fidxs = torch.cat((self.future_fidxs, torch.arange(self.runtime.context_frames)))[None]

        latent, plan = self.runtime.predict(latents, position, euler, pose_mask, fidxs)
        self.context = torch.roll(self.context, -1, dims=0)
        self.context[-1] = latent
        self.future_fidxs -= 1
        self.target += FRAME_SKIP
        return latent, self.runtime.decode(latent[None])[0], plan


def rollout(
    env: Env,
    actor: ConstantActor | WASDActor | SupercomboActor,
    viewer=None,
) -> Rollout:
    """Run synchronously for training or keep an interactive viewer responsive."""
    worldmodel_outputs: dict[str, list[torch.Tensor | np.ndarray]] = {
        "latents": [],
        "frames": [],
        "plan": [],
    }
    model_inputs: list[dict[str, torch.Tensor] | None] = []
    model_outputs: list[dict[str, torch.Tensor] | None] = []

    def record(latent, frame, plan, inputs, outputs) -> None:
        worldmodel_outputs["latents"].append(latent)
        worldmodel_outputs["frames"].append(frame)
        worldmodel_outputs["plan"].append(plan)
        model_inputs.append(inputs)
        model_outputs.append(outputs)

    curvature, accel = actor.act(env.initial_frame, env.physics.speed)
    if viewer is None:
        while not env.done:
            inputs, outputs = actor.model_inputs, actor.model_outputs
            latent, frame, plan = env.step(curvature, accel)
            record(latent, frame, plan, inputs, outputs)
            curvature, accel = actor.act(frame, env.physics.speed)
        return Rollout(worldmodel_outputs, model_inputs, model_outputs)

    pending: Future[tuple[torch.Tensor, np.ndarray, np.ndarray]] | None = None
    pending_inputs: dict[str, torch.Tensor] | None = None
    pending_outputs: dict[str, torch.Tensor] | None = None
    target, speed = env.target, env.physics.speed
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        while viewer.open and (pending is not None or not env.done):
            frame = None
            if pending is not None and pending.done():
                completed, pending = pending, None
                latent, frame, plan = completed.result()
                curvature, accel = actor.act(frame, env.physics.speed)
                record(latent, frame, plan, pending_inputs, pending_outputs)
                target, speed = env.target, env.physics.speed
                frame = draw_worldmodel_outputs(frame, plan)
            else:
                curvature, accel = actor.act(None, speed)

            viewer.render(frame, speed, target, env.end, curvature, accel, pending is None and env.done)
            if pending is None and not env.done:
                pending_inputs, pending_outputs = actor.model_inputs, actor.model_outputs
                pending = worker.submit(env.step, curvature, accel)
    finally:
        viewer.close()
        worker.shutdown(cancel_futures=True)
        if pending is not None and not pending.cancelled():
            latent, frame, plan = pending.result()
            record(latent, frame, plan, pending_inputs, pending_outputs)
    return Rollout(worldmodel_outputs, model_inputs, model_outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--segment", default=SEGMENT, help="segment ID under data/")
    parser.add_argument("--server", default=SERVER_URL, help="resident Runtime server URL")
    parser.add_argument("--sampling-steps", type=int, default=15, help="diffusion steps per generated frame")
    parser.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance scale")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU used for video decoding")
    parser.add_argument("--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE, help="VAE encoding batch size")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="optional latent cache directory")
    parser.add_argument("--actor", choices=("wasd", "supercombo"), default="wasd")
    parser.add_argument("--model", type=Path, default=SUPERCOMBO, help="ONNX model or Torch state dict")
    parser.add_argument("--on-policy", type=Path, help="finetuned on-policy state dict")
    parser.add_argument("--start", type=int, help="first 20 Hz camera frame to dream")
    parser.add_argument("--end", type=int, help="exclusive 20 Hz frame")
    args = parser.parse_args()

    runtime = RuntimeClient(args.server, args.sampling_steps, args.cfg)
    episode = Episode(
        runtime,
        args.segment,
        gpu_id=args.gpu_id,
        encode_batch_size=args.encode_batch_size,
        cache_dir=args.cache_dir,
    )
    env = Env(runtime, episode, args.start, args.end)
    if args.actor == "supercombo":
        actor = (
            SupercomboActor.from_onnx(args.model)
            if args.model.suffix == ".onnx"
            else SupercomboActor.from_torch_checkpoint(args.model, on_policy=args.on_policy)
        )
    else:
        actor = WASDActor()
    from rl.viewer import Viewer

    rollout(env, actor, Viewer(env.initial_frame))


if __name__ == "__main__":
    main()
