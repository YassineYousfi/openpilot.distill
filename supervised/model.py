from __future__ import annotations

from dataclasses import dataclass, field

import timm
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class ModelInputs:
    IMG = "img"
    BIG_IMG = "big_img"
    FEATURES = "features_buffer"
    DESIRE = "desire_pulse"
    TRAFFIC = "traffic_convention"
    ACTION_T = "action_t"


@dataclass(frozen=True, slots=True)
class VisionConfig:
    model_name: str = "convnext_pico"
    input_frame_names: tuple[str, ...] = (ModelInputs.IMG, ModelInputs.BIG_IMG)
    n_frames: int = 2
    channels_per_frame: int = 6
    input_size: tuple[int, int] = (128, 256)
    output_stride: int = 32
    pretrained: bool = True
    drop_path_rate: float = 0.0
    mean: float = 255.0 / 2.0
    std: float = 255.0 / 4.0

    @property
    def channels_per_input(self) -> int:
        return self.n_frames * self.channels_per_frame

    @property
    def in_channels(self) -> int:
        return len(self.input_frame_names) * self.channels_per_input

    @property
    def grid_size(self) -> tuple[int, int]:
        height, width = self.input_size
        return height // self.output_stride, width // self.output_stride


@dataclass(frozen=True, slots=True)
class SupercomboConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    history_len: int = 9
    desire_window_len: int = 25
    desire_dim: int = 8
    traffic_dim: int = 2
    action_t_dim: int = 2
    action_size: int = 2
    plan_size: int = 33 * 15
    n_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    policy_layers: int = 1
    unvision_dim: int = 256
    unvision_heads: int = 8
    unvision_layers: int = 1
    output_size: tuple[int, int] = (128, 256)
    output_channels: int = 6

    @property
    def temporal_len(self) -> int:
        return self.history_len + self.desire_window_len - 1


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(dim * ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(self.norm(x))
        x = F.gelu(x, approximate="tanh")
        return self.dropout(self.proj(x))


class SelfAttention(nn.Module):
    """Pre-normalized multi-head attention backed by PyTorch SDPA."""

    def __init__(self, dim: int, n_heads: int, dropout: float, *, causal: bool) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.causal = causal
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = rearrange(
            self.qkv(self.norm(x)),
            "b n (qkv h d) -> qkv b h n d",
            qkv=3,
            h=self.n_heads,
            d=self.head_dim,
        )
        query, key, value = qkv.unbind(0)
        query = self.q_norm(query)
        key = self.k_norm(key)
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.proj_dropout(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float, *, causal: bool) -> None:
        super().__init__()
        self.attention = SelfAttention(dim, n_heads, dropout, causal=causal)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.mlp(x)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        n_layers: int,
        *,
        causal: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, mlp_ratio, dropout, causal=causal) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Hydra(nn.Module):
    """Supercombo-style named heads using GPT-style final normalization."""

    def __init__(self, dim: int, output_sizes: dict[str, int]) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, output_size))
                for name, output_size in output_sizes.items()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: projection(x) for name, projection in self.projections.items()}


class LinearEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Vision(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            num_classes=0,
            global_pool="",
            drop_path_rate=config.drop_path_rate,
        )
        self.output_dim = int(self.encoder.num_features)
        self.grid_size = config.grid_size
        self.register_buffer("mean", torch.tensor(config.mean).reshape(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(config.std).reshape(1, 1, 1, 1))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        frames = [inputs[name] for name in self.config.input_frame_names]
        x = torch.cat(frames, dim=1)
        dtype = next(self.encoder.parameters()).dtype
        x = x.to(dtype=dtype)
        x = self.encoder((x - self.mean.to(dtype=dtype)) / self.std.to(dtype=dtype))
        return rearrange(x, "b c h w -> b (h w) c")


class Policy(nn.Module):
    def forward(
        self,
        features: torch.Tensor,
        desire: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class PointPolicy(Policy):
    """Predict geometry, metadata, and pose from spatially pooled features."""

    def __init__(
        self,
        dim: int,
        spatial_size: int,
        config: SupercomboConfig,
        output_sizes: dict[str, int],
    ) -> None:
        super().__init__()
        del spatial_size
        self.position = nn.Embedding(config.history_len, dim)
        self.transformer = Transformer(
            dim,
            config.n_heads,
            config.mlp_ratio,
            config.dropout,
            config.policy_layers,
            causal=True,
        )
        self.hydra = Hydra(dim, output_sizes)

    def forward(
        self,
        features: torch.Tensor,
        desire: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del desire, traffic_convention, action_t
        features = features.mean(dim=2)
        timesteps = features.shape[1]
        position = self.position(torch.arange(timesteps, device=features.device))
        return self.hydra(self.transformer(features + position))


class TemporalPolicy(Policy):
    """Predict dense plans from causal spatiotemporal vision features."""

    def __init__(
        self,
        dim: int,
        spatial_size: int,
        config: SupercomboConfig,
        output_sizes: dict[str, int],
    ) -> None:
        super().__init__()
        self.desire_window_len = config.desire_window_len
        self.desire_encoder = LinearEncoder(config.desire_window_len * config.desire_dim, dim)
        self.traffic_encoder = LinearEncoder(config.traffic_dim, dim)
        self.action_t_encoder = LinearEncoder(config.action_t_dim, dim)
        self.temporal_position = nn.Embedding(config.history_len, dim)
        self.spatial_position = nn.Embedding(spatial_size, dim)
        self.transformer = Transformer(
            dim,
            config.n_heads,
            config.mlp_ratio,
            config.dropout,
            config.policy_layers,
            causal=True,
        )
        self.hydra = Hydra(dim, output_sizes)

    def _window_desire(self, desire: torch.Tensor, timesteps: int) -> torch.Tensor:
        required = timesteps + self.desire_window_len - 1
        windows = desire[:, :required].unfold(1, self.desire_window_len, 1)
        return rearrange(windows, "b t d w -> b t (w d)")

    def forward(
        self,
        features: torch.Tensor,
        desire: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        timesteps, spatial = features.shape[1:3]
        dtype = features.dtype
        desire = self.desire_encoder(self._window_desire(desire.to(dtype), timesteps))
        traffic = self.traffic_encoder(traffic_convention[:, -1].to(dtype))
        action_t = self.action_t_encoder(action_t[:, -1].to(dtype))
        temporal_position = self.temporal_position(torch.arange(timesteps, device=features.device))
        spatial_position = self.spatial_position(torch.arange(spatial, device=features.device))

        x = features
        x = x + rearrange(desire, "b t c -> b t 1 c")
        x = x + rearrange(traffic, "b c -> b 1 1 c")
        x = x + rearrange(action_t, "b c -> b 1 1 c")
        x = x + rearrange(temporal_position, "t c -> 1 t 1 c")
        x = x + rearrange(spatial_position, "s c -> 1 1 s c")
        x = self.transformer(rearrange(x, "b t s c -> b (t s) c"))
        x = rearrange(x, "b (t s) c -> b t s c", t=timesteps, s=spatial)
        return self.hydra(x[:, :, -1])  # EXPERIMENT: maybe mean is better


class Unvision(Policy):
    """Decode every spatial feature grid into narrow and wide RGB."""

    def __init__(
        self,
        dim: int,
        spatial_size: int,
        config: SupercomboConfig,
        output_sizes: dict[str, int],
    ) -> None:
        super().__init__()
        del spatial_size
        grid_size = config.vision.grid_size
        grid_height, grid_width = grid_size
        output_height, output_width = config.output_size
        self.grid_size = grid_size
        self.output_size = config.output_size
        self.output_name, self.output_channels = next(iter(output_sizes.items()))
        self.patch_size = output_height // grid_height, output_width // grid_width
        self.input_projection = nn.Linear(dim, config.unvision_dim)
        self.input_norm = nn.LayerNorm(config.unvision_dim)
        self.position = nn.Embedding(grid_height * grid_width, config.unvision_dim)
        self.transformer = Transformer(
            config.unvision_dim,
            config.unvision_heads,
            config.mlp_ratio,
            config.dropout,
            config.unvision_layers,
            causal=False,
        )
        self.output_norm = nn.LayerNorm(config.unvision_dim)
        patch_area = self.patch_size[0] * self.patch_size[1]
        self.output_projection = nn.Linear(config.unvision_dim, config.output_channels * patch_area)

    def forward(
        self,
        features: torch.Tensor,
        desire: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del desire, traffic_convention, action_t
        batch, timesteps, spatial, dim = features.shape
        features = features.reshape(batch * timesteps, spatial, dim)
        spatial = features.shape[1]
        position = self.position(torch.arange(spatial, device=features.device))
        x = self.input_norm(self.input_projection(features)) + position
        x = self.output_projection(self.output_norm(self.transformer(x)))
        grid_height, grid_width = self.grid_size
        patch_height, patch_width = self.patch_size
        images = rearrange(
            x,
            "b (gh gw) (c ph pw) -> b c (gh ph) (gw pw)",
            gh=grid_height,
            gw=grid_width,
            c=self.output_channels,
            ph=patch_height,
            pw=patch_width,
        )
        images = images.reshape(batch, timesteps, *images.shape[1:])
        return {self.output_name: (images + 1.0) * 127.5}


type PolicyEntry = tuple[str, type[Policy], dict[str, int]]


def pretraining_policies(config: SupercomboConfig) -> list[PolicyEntry]:
    trajectory_points = config.plan_size // 15
    return [
        (
            "point_policy",
            PointPolicy,
            {
                "lane_lines": 4 * 2 * (2 * trajectory_points),
                "lane_lines_prob": 8,
                "road_edges": 2 * 2 * (2 * trajectory_points),
                "meta": 55,
                "desire_pred": config.desire_dim * 4,
                "road_transform": 6 * 2,
                "wide_from_device_euler": 3 * 2,
                "pose": 6 * 2,
            },
        ),
        ("image_policy", Unvision, {"imgs": config.output_channels}),
        (
            "temporal_policy",
            TemporalPolicy,
            {
                "plan": 2 * config.plan_size,
                "lead": 3 * 2 * (6 * 4),
                "lead_prob": 3,
                "action": 2 * config.action_size,
                "desire_state": config.desire_dim,
            },
        ),
    ]


def rl_policies(config: SupercomboConfig) -> list[PolicyEntry]:
    return [
        *pretraining_policies(config),
        ("on_policy_temporal", TemporalPolicy, {"action": 2 * config.action_size}),
    ]


class Supercombo(nn.Module):
    """Dense training model that processes the complete image history."""

    def __init__(
        self,
        config: SupercomboConfig | None = None,
        policies: list[PolicyEntry] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SupercomboConfig()
        self.vision = Vision(self.config.vision)
        self.spatial_size = self.config.vision.grid_size[0] * self.config.vision.grid_size[1]
        policy_names = []
        policies = pretraining_policies(self.config) if policies is None else policies
        for name, policy_type, output_sizes in policies:
            policy = policy_type(self.vision.output_dim, self.spatial_size, self.config, output_sizes)
            self.add_module(name, policy)
            policy_names.append(name)
        self.policy_order = tuple(policy_names)

    @staticmethod
    def example_inputs(
        config: SupercomboConfig | None = None,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        config = config or SupercomboConfig()
        temporal_len = config.temporal_len
        channels = config.vision.channels_per_input
        height, width = config.vision.input_size
        return {
            ModelInputs.IMG: torch.zeros(
                batch_size, config.history_len, channels, height, width, dtype=torch.uint8, device=device
            ),
            ModelInputs.BIG_IMG: torch.zeros(
                batch_size, config.history_len, channels, height, width, dtype=torch.uint8, device=device
            ),
            ModelInputs.DESIRE: torch.zeros(
                batch_size, temporal_len, config.desire_dim, dtype=torch.float32, device=device
            ),
            ModelInputs.TRAFFIC: torch.zeros(
                batch_size, temporal_len, config.traffic_dim, dtype=torch.float32, device=device
            ),
            ModelInputs.ACTION_T: torch.zeros(
                batch_size, temporal_len, config.action_t_dim, dtype=torch.float32, device=device
            ),
        }

    def forward(
        self,
        inputs: dict[str, torch.Tensor] | torch.Tensor,
        big_img: torch.Tensor | None = None,
        desire_pulse: torch.Tensor | None = None,
        traffic_convention: torch.Tensor | None = None,
        action_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            inputs = {
                ModelInputs.IMG: inputs,
                ModelInputs.BIG_IMG: big_img,
                ModelInputs.DESIRE: desire_pulse,
                ModelInputs.TRAFFIC: traffic_convention,
                ModelInputs.ACTION_T: action_t,
            }

        img = inputs[ModelInputs.IMG]
        batch, timesteps = img.shape[:2]
        vision_inputs = {
            name: rearrange(inputs[name], "b t c h w -> (b t) c h w", b=batch, t=timesteps)
            for name in self.config.vision.input_frame_names
        }
        features = self.vision(vision_inputs)
        features = rearrange(features, "(b t) s c -> b t s c", b=batch, t=timesteps)

        outputs = {}
        for name in self.policy_order:
            policy = getattr(self, name)
            policy_outputs = policy(
                features,
                inputs[ModelInputs.DESIRE],
                inputs[ModelInputs.TRAFFIC],
                inputs[ModelInputs.ACTION_T],
            )
            for output_name, value in policy_outputs.items():
                outputs.pop(output_name, None)
                outputs[output_name] = value
        outputs["hidden_state"] = features
        return outputs


PathModel = Supercombo
Model = Supercombo

__all__ = [
    "Model",
    "ModelInputs",
    "PathModel",
    "Policy",
    "PolicyEntry",
    "Supercombo",
    "SupercomboConfig",
    "VisionConfig",
    "pretraining_policies",
    "rl_policies",
]
