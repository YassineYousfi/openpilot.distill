"""Backend-native, state-explicit inference for Supercombo."""

from __future__ import annotations

import base64
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .model import ModelInputs, Supercombo, SupercomboConfig


_POLICY_FEATURES = "features"


class InputQueues:
    """Backend-native rolling state passed explicitly to an inference model."""

    @classmethod
    def from_model(
        cls,
        model: SupercomboForInference,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> InputQueues:
        inputs = model.example_inputs(batch_size=batch_size, device=device)
        state = {name: value for name, value in inputs.items() if name not in model.image_names}
        if model.backend == "torch":
            device = next(iter(state.values())).device
        return cls(
            {name: tuple(value.shape) for name, value in state.items()},
            {name: value.dtype for name, value in state.items()},
            backend=model.backend,
            device=device,
        )

    @classmethod
    def from_torch_model(
        cls,
        model: TorchSupercomboForInference,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> InputQueues:
        return cls.from_model(model, batch_size=batch_size, device=device)

    @classmethod
    def from_onnx_model(
        cls,
        model: ORTSupercomboForInference,
        *,
        batch_size: int = 1,
    ) -> InputQueues:
        return cls.from_model(model, batch_size=batch_size)

    def __init__(
        self,
        shapes: dict[str, tuple[int, ...]],
        dtypes: dict[str, object],
        *,
        backend: Literal["numpy", "torch"],
        device: torch.device | str | None = None,
    ) -> None:
        self.shapes = dict(shapes)
        self.dtypes = dict(dtypes)
        self.backend = backend
        self.device = torch.device(device or "cpu") if backend == "torch" else None
        if backend == "numpy":
            self.q = {name: np.zeros(shape, dtype=dtypes[name]) for name, shape in self.shapes.items()}
        else:
            self.q = {
                name: torch.zeros(shape, dtype=dtypes[name], device=self.device) for name, shape in self.shapes.items()
            }

    def reset(self) -> None:
        for queue in self.q.values():
            queue[...] = 0

    def enqueue(self, inputs: dict[str, object]) -> None:
        for name, value in inputs.items():
            queue = self.q[name]
            if self.backend == "numpy":
                value = np.asarray(value, dtype=queue.dtype)
            else:
                value = torch.as_tensor(value, device=queue.device, dtype=queue.dtype).detach()
            if value.ndim > queue.ndim:
                value = value[:, -1]
            if value.ndim == queue.ndim - 1:
                value = value[:, None]
            value = value[:, -queue.shape[1] :]
            steps = value.shape[1]
            if steps < queue.shape[1]:
                previous = queue[:, steps:].copy() if self.backend == "numpy" else queue[:, steps:].clone()
                queue[:, :-steps] = previous
                queue[:, -steps:] = value
            else:
                queue[...] = value

    def get(self, *names: str) -> dict[str, np.ndarray | torch.Tensor]:
        names = names or tuple(self.q)
        return {name: self.q[name] for name in names}


def _enqueue_inputs(inputs: dict[str, object], queues: InputQueues) -> None:
    queues.enqueue({name: value for name, value in inputs.items() if name in queues.q and name != ModelInputs.FEATURES})


class SupercomboForInference:
    """Shared queue, prefill, and one-frame decode mechanics."""

    backend: Literal["numpy", "torch"]
    image_names: tuple[str, ...]
    history_len: int

    def example_inputs(
        self,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> dict[str, np.ndarray | torch.Tensor]:
        raise NotImplementedError

    def encode(self, inputs: dict[str, object]):
        raise NotImplementedError

    def _run_policies(self, features, state: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError

    def _concatenate(self, values: tuple[object, ...], axis: int):
        raise NotImplementedError

    def _forward(
        self,
        inputs: dict[str, object],
        queues: InputQueues,
        *,
        dense: bool,
    ) -> dict[str, object]:
        _enqueue_inputs(inputs, queues)
        current = self.encode({name: inputs[name] for name in self.image_names})
        features = self._concatenate((queues.q[ModelInputs.FEATURES], current), axis=1)
        features = features[:, -self.history_len :]
        outputs = self._run_policies(features, queues.q)
        outputs.setdefault("hidden_state", features)
        queues.enqueue({ModelInputs.FEATURES: current})
        if dense:
            return outputs
        return {name: value[:, -1] for name, value in outputs.items()}

    def prefill(
        self,
        inputs: dict[str, object],
        queues: InputQueues,
        *,
        dense: bool = False,
    ) -> dict[str, object]:
        """Encode an image block and update the queues once."""

        return self._forward(inputs, queues, dense=dense)

    def decode(
        self,
        inputs: dict[str, object],
        queues: InputQueues,
        *,
        dense: bool = False,
    ) -> dict[str, object]:
        """Decode one new image frame and update the queues once."""

        inputs = dict(inputs)
        for name in self.image_names:
            inputs[name] = inputs[name][:, None]
        return self._forward(inputs, queues, dense=dense)

    def forward_streaming(
        self,
        inputs: dict[str, object],
        queues: InputQueues,
        *,
        dense: bool = False,
    ) -> dict[str, object]:
        return self.decode(inputs, queues, dense=dense)


class TorchSupercomboForInference(SupercomboForInference):
    """Torch implementation of split vision and dense policy inference."""

    backend = "torch"

    def __init__(self, model: Supercombo) -> None:
        self.model = model.eval()
        self.image_names = model.config.vision.input_frame_names
        self.history_len = model.config.history_len

    @property
    def config(self) -> SupercomboConfig:
        return self.model.config

    def example_inputs(
        self,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        parameter = next(self.model.parameters())
        device = device or parameter.device
        config = self.config
        channels = config.vision.channels_per_input
        height, width = config.vision.input_size
        return {
            **{
                name: torch.zeros(batch_size, channels, height, width, dtype=torch.uint8, device=device)
                for name in self.image_names
            },
            ModelInputs.FEATURES: torch.zeros(
                batch_size,
                config.history_len - 1,
                self.model.spatial_size,
                self.model.vision.output_dim,
                dtype=parameter.dtype,
                device=device,
            ),
            ModelInputs.DESIRE: torch.zeros(
                batch_size,
                config.temporal_len,
                config.desire_dim,
                dtype=parameter.dtype,
                device=device,
            ),
            ModelInputs.TRAFFIC: torch.zeros(
                batch_size,
                config.temporal_len,
                config.traffic_dim,
                dtype=parameter.dtype,
                device=device,
            ),
            ModelInputs.ACTION_T: torch.zeros(
                batch_size,
                config.temporal_len,
                config.action_t_dim,
                dtype=parameter.dtype,
                device=device,
            ),
        }

    @torch.inference_mode()
    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        first = inputs[self.image_names[0]]
        leading_shape = first.shape[:-3]
        vision_inputs = {name: value.reshape(-1, *value.shape[-3:]) for name, value in inputs.items()}
        features = self.model.vision(vision_inputs)
        return features.reshape(*leading_shape, *features.shape[1:])

    @torch.inference_mode()
    def _run_policies(
        self,
        features: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        outputs = {}
        for name in self.model.policy_order:
            policy_outputs = getattr(self.model, name)(
                features,
                state[ModelInputs.DESIRE],
                state[ModelInputs.TRAFFIC],
                state[ModelInputs.ACTION_T],
            )
            for output_name, value in policy_outputs.items():
                outputs.pop(output_name, None)
                outputs[output_name] = value
        return outputs

    def _concatenate(self, values: tuple[torch.Tensor, ...], axis: int) -> torch.Tensor:
        return torch.cat(values, dim=axis)


class ORTSupercomboForInference(SupercomboForInference):
    """ONNX Runtime implementation using split vision and dense policy graphs."""

    backend = "numpy"

    def __init__(
        self,
        vision_model: str | Path | object,
        policies_model: str | Path | object,
        *,
        providers: list[object] | None = None,
    ) -> None:
        self.vision_session = self._session(vision_model, providers)
        self.policies_session = self._session(policies_model, providers)
        self.vision_inputs = {value.name: value for value in self.vision_session.get_inputs()}
        self.policy_inputs = {value.name: value for value in self.policies_session.get_inputs()}
        self.image_names = tuple(self.vision_inputs)
        self.vision_output = self.vision_session.get_outputs()[0].name
        self.policy_output_names = tuple(value.name for value in self.policies_session.get_outputs())
        metadata = self.policies_session.get_modelmeta().custom_metadata_map
        self.output_slices = pickle.loads(base64.b64decode(metadata["output_slices"]))

        features = self.policy_inputs[_POLICY_FEATURES]
        desire = self.policy_inputs[ModelInputs.DESIRE]
        self.history_len = int(features.shape[1])
        self.feature_shape = tuple(int(value) for value in features.shape[2:])
        self.temporal_len = int(desire.shape[1])

    @staticmethod
    def _session(model: str | Path | object, providers: list[object] | None):
        if hasattr(model, "run"):
            return model
        import onnxruntime as ort

        source = model.SerializeToString() if hasattr(model, "SerializeToString") else str(model)
        return ort.InferenceSession(source, providers=providers)

    @classmethod
    def from_supercombo(
        cls,
        model: str | Path | object,
        *,
        providers: list[object] | None = None,
    ) -> ORTSupercomboForInference:
        import onnx

        from .onnx_helpers import make_dense, split_supercombo

        model = model if hasattr(model, "graph") else onnx.load(str(model))
        vision, policies = split_supercombo(model)
        return cls(vision, make_dense(policies), providers=providers)

    @staticmethod
    def _dtype(value) -> np.dtype:
        name = value.type.removeprefix("tensor(").removesuffix(")")
        return np.dtype({"float": "float32", "double": "float64"}.get(name, name))

    def example_inputs(
        self,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> dict[str, np.ndarray]:
        del device
        inputs = {
            name: np.zeros(
                (batch_size, *(int(value) for value in info.shape[1:])),
                dtype=self._dtype(info),
            )
            for name, info in self.vision_inputs.items()
        }
        feature_info = self.policy_inputs[_POLICY_FEATURES]
        inputs[ModelInputs.FEATURES] = np.zeros(
            (batch_size, self.history_len - 1, *self.feature_shape),
            dtype=self._dtype(feature_info),
        )
        desire_info = self.policy_inputs[ModelInputs.DESIRE]
        inputs[ModelInputs.DESIRE] = np.zeros(
            (batch_size, *(int(value) for value in desire_info.shape[1:])),
            dtype=self._dtype(desire_info),
        )
        for name in (ModelInputs.TRAFFIC, ModelInputs.ACTION_T):
            info = self.policy_inputs[name]
            trailing_shape = info.shape[2:] if len(info.shape) > 2 else info.shape[1:]
            inputs[name] = np.zeros(
                (batch_size, self.temporal_len, *(int(value) for value in trailing_shape)),
                dtype=self._dtype(info),
            )
        return inputs

    def encode(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        first = inputs[self.image_names[0]]
        leading_shape = first.shape[:-3]
        feed = {
            name: np.ascontiguousarray(
                value.reshape(-1, *value.shape[-3:]),
                dtype=self._dtype(self.vision_inputs[name]),
            )
            for name, value in inputs.items()
        }
        features = self.vision_session.run([self.vision_output], feed)[0]
        return features.reshape(*leading_shape, *features.shape[1:])

    def _unpack_outputs(self, values: list[np.ndarray]) -> dict[str, np.ndarray]:
        outputs = dict(zip(self.policy_output_names, values, strict=True))
        if len(outputs) != 1:
            return outputs

        flat = next(iter(outputs.values()))
        unpacked = {}
        for name, output_slice in self.output_slices.items():
            if name == "pad":
                continue
            value = flat[..., output_slice]
            if name == "hidden_state":
                value = value.reshape(*flat.shape[:-1], *self.feature_shape)
            unpacked[name] = value
        return unpacked

    def _run_policies(
        self,
        features: np.ndarray,
        state: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        feed = {
            _POLICY_FEATURES: np.ascontiguousarray(
                features,
                dtype=self._dtype(self.policy_inputs[_POLICY_FEATURES]),
            )
        }
        for name, info in self.policy_inputs.items():
            if name == _POLICY_FEATURES:
                continue
            value = state[name]
            if value.ndim > len(info.shape):
                value = value[:, -1]
            feed[name] = np.ascontiguousarray(value, dtype=self._dtype(info))
        values = self.policies_session.run(list(self.policy_output_names), feed)
        return self._unpack_outputs(values)

    def _concatenate(self, values: tuple[np.ndarray, ...], axis: int) -> np.ndarray:
        return np.concatenate(values, axis=axis)


__all__ = [
    "InputQueues",
    "ORTSupercomboForInference",
    "SupercomboForInference",
    "TorchSupercomboForInference",
]
