from pathlib import Path

import cv2
import numpy as np
import pyray as rl
import torch

from supervised.model import ModelInputs, Supercombo, SupercomboConfig, VisionConfig, rl_policies
from supervised.model_for_inference import (
    InputQueues,
    ORTSupercomboForInference,
    SupercomboForInference,
    TorchSupercomboForInference,
)


FPS = 5
SUPERCOMBO = Path(__file__).resolve().parents[1] / "models/big_driving_supercombo.onnx"


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
        curvature = 0.005 * (rl.is_key_down(rl.KEY_D) - rl.is_key_down(rl.KEY_A))
        accel = 0.5 * (rl.is_key_down(rl.KEY_W) - rl.is_key_down(rl.KEY_S))
        return curvature, accel


class SupercomboActor:
    """Run either the Torch or ONNX Supercombo one frame at a time."""

    @classmethod
    def from_torch(cls, model: Supercombo) -> "SupercomboActor":
        return cls(TorchSupercomboForInference(model))

    @classmethod
    def from_torch_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        on_policy: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> "SupercomboActor":
        config = SupercomboConfig(vision=VisionConfig(pretrained=False))
        model = Supercombo(config, rl_policies(config))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=False)
        temporal = model.temporal_policy.state_dict()
        model.on_policy_temporal.load_state_dict(
            {name: temporal[name] for name in model.on_policy_temporal.state_dict()}
        )
        if on_policy is not None:
            model.on_policy_temporal.load_state_dict(
                torch.load(on_policy, map_location="cpu", weights_only=True)
            )
        device = device or (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        return cls.from_torch(model.to(device).eval())

    @classmethod
    def from_onnx(
        cls,
        model: str | Path = SUPERCOMBO,
        *,
        providers: list[object] | None = None,
    ) -> "SupercomboActor":
        if providers is None:
            providers = (
                [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device()}), "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
        return cls(ORTSupercomboForInference.from_supercombo(model, providers=providers))

    def __init__(self, model: SupercomboForInference):
        self.model = model
        self.queues = InputQueues.from_model(model)
        self.previous: dict[str, np.ndarray] = {}
        self.model_inputs: dict[str, torch.Tensor] | None = None
        self.model_outputs: dict[str, torch.Tensor] | None = None
        self.reset()

    @staticmethod
    def _cpu(value: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return torch.from_numpy(np.array(value, copy=True))

    @torch.inference_mode()
    def reset(self) -> None:
        self.queues.reset()
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
            current_images[name] = tensor
        if not self.previous:
            self.previous = current_images
            return self.action

        inputs = {
            name: np.concatenate((self.previous[name], current), axis=0)[None]
            for name, current in current_images.items()
        }
        self.previous = current_images
        if self.model.backend == "torch":
            inputs = {name: torch.from_numpy(value).to(self.queues.device) for name, value in inputs.items()}
        inputs |= {
            ModelInputs.DESIRE: np.zeros((1, 8), dtype=np.float32),
            ModelInputs.TRAFFIC: np.array([[1.0, 0.0]], dtype=np.float32),
            ModelInputs.ACTION_T: np.full((1, 2), 1 / FPS, dtype=np.float32),
        }

        features = self._cpu(self.queues.q[ModelInputs.FEATURES])
        outputs = self.model.decode(inputs, self.queues)
        self.model_inputs = {
            **{name: self._cpu(inputs[name]) for name in self.model.image_names},
            ModelInputs.FEATURES: features,
            **{
                name: self._cpu(self.queues.q[name])
                for name in (ModelInputs.DESIRE, ModelInputs.TRAFFIC, ModelInputs.ACTION_T)
            },
        }
        self.model_outputs = {name: self._cpu(value) for name, value in outputs.items()}

        lat_accel, accel = self.model_outputs["action"][0, :2].float().tolist()
        self.action = (lat_accel / max(speed, 1.0) ** 2, accel)
        return self.action
