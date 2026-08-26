from pathlib import Path

import cv2
import numpy as np
import pyray as rl
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.experiments.path.model_constants import (
    DESIRE_LEN,
    ModelInputs,
    SUPERCOMBO_FPS as FPS,
    frame_constants_from_fps,
)
from torchtitan.experiments.rldriving.supercombo import Supercombo


SUPERCOMBO = Path(__file__).resolve().parents[1] / "models/supercombo"


class ConstantActor:
    model_inputs = None
    model_outputs = None

    def __init__(self, curvature: float = 0.0, accel: float = 0.0):
        self.curvature = curvature
        self.accel = accel

    def act(self, frame: np.ndarray | None = None, speed: float = 0.0) -> tuple[float, float]:
        return self.curvature, self.accel


class WASDActor:
    model_inputs = None
    model_outputs = None

    def act(self, frame: np.ndarray | None = None, speed: float = 0.0) -> tuple[float, float]:
        curvature = 0.005 * (rl.is_key_down(rl.KEY_A) - rl.is_key_down(rl.KEY_D))
        accel = 0.5 * (rl.is_key_down(rl.KEY_W) - rl.is_key_down(rl.KEY_S))
        return curvature, accel


class SupercomboActor:
    """Run the local TorchTitan Supercombo checkpoint on rollout frames."""

    def __init__(self, on_policy: Path | None = None):
        self.device = torch.device("cuda", torch.cuda.current_device())
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            self.model = Supercombo()
        finally:
            torch.set_default_dtype(default_dtype)

        dcp.load(
            {
                "vision": self.model.vision,
                "point_policy": self.model.point_policy,
                "temporal_policy": self.model.off_policy,
            },
            checkpoint_id=SUPERCOMBO,
            no_dist=True,
        )
        if on_policy is None:
            off_policy = self.model.off_policy.state_dict()
            state = {name: off_policy[name] for name in self.model.on_policy.state_dict()}
        else:
            state = torch.load(on_policy, map_location="cpu", weights_only=True)
        self.model.on_policy.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        constants = frame_constants_from_fps(FPS)
        spatial_size = self.model.off_policy.temporal_summarizer.spatial_size
        vision_features = self.model.vision.config.vision_features
        dtype = next(self.model.parameters()).dtype
        self.features = torch.zeros(
            1, constants["temporal_len"] - 1, spatial_size, vision_features, device=self.device, dtype=dtype
        )
        self.desire = torch.zeros(1, constants["temporal_len"], DESIRE_LEN, device=self.device, dtype=dtype)
        self.traffic = torch.tensor([[1.0, 0.0]], device=self.device, dtype=dtype)
        self.action_t = torch.full((1, 2), 1 / FPS, device=self.device, dtype=dtype)
        self.previous: dict[str, torch.Tensor] = {}
        self.hidden_size = spatial_size * vision_features
        self.model_inputs: dict[str, torch.Tensor] | None = None
        self.model_outputs: dict[str, torch.Tensor] | None = None
        self.reset()

    @torch.inference_mode()
    def reset(self) -> None:
        self.features.zero_()
        self.previous.clear()
        self.action = (0.0, 0.0)
        self.model_inputs = None
        self.model_outputs = None

    @torch.inference_mode()
    def act(self, frame: np.ndarray | None = None, speed: float = 0.0) -> tuple[float, float]:
        if frame is None:
            return self.action

        current_images = {}
        for name, image in ((ModelInputs.IMG, frame[..., :3]), (ModelInputs.BIG_IMG, frame[..., 3:])):
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420)
            tensor = np.empty((6, 128, 256), dtype=np.uint8)
            tensor[0] = yuv[:256:2, 0::2]
            tensor[1] = yuv[1:256:2, 0::2]
            tensor[2] = yuv[:256:2, 1::2]
            tensor[3] = yuv[1:256:2, 1::2]
            tensor[4] = yuv[256:320].reshape(128, 256)
            tensor[5] = yuv[320:384].reshape(128, 256)
            current_images[name] = torch.from_numpy(tensor).to(self.device)
        if not self.previous:
            self.previous = current_images
            return self.action

        inputs: dict[str, torch.Tensor] = {
            name: torch.cat((self.previous[name], current))[None] for name, current in current_images.items()
        }
        self.previous = current_images
        inputs.update(
            {
                "features_buffer": self.features,
                ModelInputs.DESIRE: self.desire,
                ModelInputs.TRAFFIC: self.traffic,
                ModelInputs.ACTION_T: self.action_t,
            }
        )

        output = self.model(inputs) # TODO: this should be a dict
        tail = output[:, -(4 + self.hidden_size + self.model.pad.shape[1]) :]
        lat_accel, accel = tail[0, :2].float().cpu().tolist()
        self.model_inputs = {name: value.detach().cpu() for name, value in inputs.items()}
        self.model_outputs = {
            "outputs": output.detach().cpu(),
            "action": tail[:, :4].detach().cpu(),
            "vision_features": tail[:, 4 : 4 + self.hidden_size].reshape_as(self.features[:, -1]).detach().cpu(),
        }
        self.features = torch.roll(self.features, -1, dims=1)
        self.features[:, -1] = tail[:, 4 : 4 + self.hidden_size].reshape_as(self.features[:, -1])
        self.action = (lat_accel / max(speed, 1.0) ** 2, accel)
        return self.action
