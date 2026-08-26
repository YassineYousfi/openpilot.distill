#!/usr/bin/env python3
import argparse
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request as URLRequest, urlopen

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from huggingface_hub import hf_hub_download
from safetensors.torch import load, save
from torch.package import PackageImporter


ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8006
SERVER_URL = f"http://{HOST}:{PORT}"
MEDIA_TYPE = "application/octet-stream"
MODEL = ROOT / "models/no_future_model.fp8_nvfp4.torchpackage"
VAE = "commaai/vit-ae-2x-f8c32"
FUTURE_FRAMES = 0


class Runtime:
    def __init__(self, model_path: Path, device: str):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the v0 environment requires CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        importer = PackageImporter(str(model_path))
        model_io = importer.load_pickle("meta", "meta.pkl")["model_io"]
        model_frames = model_io["in_shape"]["latents"][1]
        if model_frames <= FUTURE_FRAMES:
            raise ValueError(f"model must have more than {FUTURE_FRAMES} frame slots")
        self.future_frames = FUTURE_FRAMES
        self.context_frames = model_frames - self.future_frames
        self.history_frames = self.context_frames - 1
        self.model_dtype = model_io["in_dtype"]["latents"]
        self.vae_dtype = torch.bfloat16

        self.encoder = torch.export.load(hf_hub_download(VAE, "encoder.pt2")).module().to(self.device, self.vae_dtype)
        self.decoder = torch.export.load(hf_hub_download(VAE, "decoder.pt2")).module().to(self.device, self.vae_dtype)
        self.model = importer.load_pickle("model", "model.pkl")
        state = torch.load(
            io.BytesIO(importer.load_binary("assets", "state_dict.pt")),
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.eval()
        del state
        self.model.compile_for_inference()

    @torch.inference_mode()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.device, self.vae_dtype)
        return self.encoder(pixels.div(127.5).sub(1)).to(self.model_dtype)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> np.ndarray:
        frames = self.decoder(latents.to(self.vae_dtype)).float().add(1).mul(127.5).clamp(0, 255).byte()
        return frames.cpu().numpy()

    @torch.inference_mode()
    def predict(
        self,
        latents: torch.Tensor,
        positions: torch.Tensor,
        eulers: torch.Tensor,
        pose_mask: torch.Tensor,
        fidxs: torch.Tensor,
        sampling_steps: int,
        cfg: float,
    ) -> tuple[torch.Tensor, np.ndarray]:
        output = self.model.generate(
            latents=latents.clone(),
            augments_pos_ref_augment=positions,
            ref_augment_from_augments_euler=eulers,
            pose_mask=pose_mask,
            fidxs=fidxs,
            steps=sampling_steps,
            num_prefill_frames=latents.shape[1] - 1,
            dtype=self.model_dtype,
            inference_schedule="linear",
            cfg=cfg,
        )
        plan = output["plan"][0]
        plan = plan[: plan.numel() // 2].reshape(-1, 15).float().cpu().numpy()
        return output["latents"][0, 0], plan


class RuntimeClient:
    def __init__(self, url: str, sampling_steps: int = 30, cfg: float = 2.0):
        self.url = url.rstrip("/")
        self.sampling_steps = sampling_steps
        self.cfg = cfg
        with urlopen(f"{self.url}/health") as response:
            health = json.load(response)
        self.future_frames = int(health["future_frames"])
        self.context_frames = int(health["context_frames"])
        self.history_frames = int(health["history_frames"])

    def _call(
        self,
        method: str,
        tensors: dict[str, torch.Tensor],
        query: dict[str, int | float] | None = None,
    ) -> dict[str, torch.Tensor]:
        body = save({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()})
        url = f"{self.url}/{method}"
        if query:
            url = f"{url}?{urlencode(query)}"
        request = URLRequest(
            url,
            data=body,
            headers={"Content-Type": MEDIA_TYPE},
        )
        with urlopen(request) as response:
            return load(response.read())

    def encode(self, frames: np.ndarray) -> torch.Tensor:
        return self._call("encode", {"frames": torch.from_numpy(np.ascontiguousarray(frames))})["latents"]

    def decode(self, latents: torch.Tensor) -> np.ndarray:
        return self._call("decode", {"latents": latents})["frames"].numpy()

    def predict(
        self,
        latents: torch.Tensor,
        positions: torch.Tensor,
        eulers: torch.Tensor,
        pose_mask: torch.Tensor,
        fidxs: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray]:
        output = self._call(
            "predict",
            {
                "latents": latents,
                "positions": positions,
                "eulers": eulers,
                "pose_mask": pose_mask,
                "fidxs": fidxs,
            },
            {"sampling_steps": self.sampling_steps, "cfg": self.cfg},
        )
        return output["latents"], output["plan"].numpy()


async def _read_tensors(request: Request) -> dict[str, torch.Tensor]:
    try:
        return load(await request.body())
    except Exception as exc:
        raise HTTPException(400, f"invalid Safetensors body: {exc}") from exc


def _require(tensors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in tensors:
        raise HTTPException(422, f"missing tensor: {name}")
    return tensors[name]


def _tensor_response(**tensors: torch.Tensor) -> Response:
    body = save({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()})
    return Response(body, media_type=MEDIA_TYPE)


def create_app(model: Path = MODEL, device: str = "cuda") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = Runtime(model, device)
        yield

    api = FastAPI(title="World-model Runtime", lifespan=lifespan)

    @api.get("/health")
    async def health() -> dict[str, object]:
        runtime: Runtime = api.state.runtime
        return {
            "ready": True,
            "device": str(runtime.device),
            "future_frames": runtime.future_frames,
            "context_frames": runtime.context_frames,
            "history_frames": runtime.history_frames,
        }

    @api.post("/encode")
    async def encode(request: Request) -> Response:
        frames = _require(await _read_tensors(request), "frames")
        if frames.ndim == 3:
            frames = frames[None]
        latents = api.state.runtime.encode(frames.numpy())
        return _tensor_response(latents=latents)

    @api.post("/predict")
    async def predict(request: Request, sampling_steps: int = 30, cfg: float = 2.0) -> Response:
        if sampling_steps <= 0:
            raise HTTPException(422, "sampling_steps must be positive")
        tensors = await _read_tensors(request)
        runtime: Runtime = api.state.runtime
        latents = _require(tensors, "latents").to(runtime.device, runtime.model_dtype)
        positions = _require(tensors, "positions").to(runtime.device, runtime.model_dtype)
        eulers = _require(tensors, "eulers").to(runtime.device, runtime.model_dtype)
        pose_mask = _require(tensors, "pose_mask").to(runtime.device, torch.int64)
        fidxs = _require(tensors, "fidxs").to(runtime.device, torch.int64)
        latent, plan = runtime.predict(latents, positions, eulers, pose_mask, fidxs, sampling_steps, cfg)
        return _tensor_response(latents=latent, plan=torch.from_numpy(plan))

    @api.post("/decode")
    async def decode(request: Request) -> Response:
        runtime: Runtime = api.state.runtime
        latents = _require(await _read_tensors(request), "latents")
        if latents.ndim == 3:
            latents = latents[None]
        frames = runtime.decode(latents.to(runtime.device))
        return _tensor_response(frames=torch.from_numpy(frames))

    return api


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", nargs="?", type=Path, default=MODEL, help="local world-model .torchpackage")
    parser.add_argument("--device", default="cuda", help="PyTorch CUDA device")
    parser.add_argument("--host", default=HOST, help="listen address")
    parser.add_argument("--port", type=int, default=PORT, help="listen port")
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
