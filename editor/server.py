#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ServiceConfig:
    service_name: str = "diffusers_flux2"
    model_path: str = ""
    output_dir: str = "outputs/editor"
    profile_name: str = "eager_4b_bf16"
    device: str = "cuda"
    dtype: str = "bf16"
    steps: int = 4
    cfg: float = 1.0
    sigmas: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    max_sequence_length: int = 512
    text_encoder_out_layers: tuple[int, ...] = (9, 18, 27)
    base_seed: int = 764952063587760
    compile: bool = False
    attention_backend: str = ""
    cache_mode: str = "none"
    cache_threshold: float = 0.05
    mag_ratios: tuple[float, ...] = ()


class EditRequest(BaseModel):
    image_path: str
    positive_prompt: str = ""
    negative_prompt: str = ""
    region_prompt: str = ""
    edit_plan: dict[str, Any] = Field(default_factory=dict)
    request_index: int = 0
    seed: int | None = None


class RunnerProtocol(Protocol):
    ready: bool
    last_error: str | None
    metadata: dict[str, Any]

    def edit(self, request: EditRequest) -> dict[str, Any]:
        ...


def _config_metadata(config: ServiceConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["sigmas"] = list(config.sigmas)
    payload["text_encoder_out_layers"] = list(config.text_encoder_out_layers)
    payload["mag_ratios"] = list(config.mag_ratios)
    return payload


def create_app(runner: RunnerProtocol, config: ServiceConfig) -> FastAPI:
    app = FastAPI(title=config.service_name)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": bool(runner.ready),
            "ready": bool(runner.ready),
            "service": config.service_name,
            "last_error": runner.last_error,
            **_config_metadata(config),
            **runner.metadata,
        }

    @app.post("/edit")
    async def edit(request: EditRequest):
        if not Path(request.image_path).is_file():
            raise HTTPException(status_code=404, detail="image_path does not exist")
        if not request.positive_prompt.strip() and not request.region_prompt.strip():
            raise HTTPException(status_code=422, detail="positive_prompt and region_prompt are empty")
        if not runner.ready:
            raise HTTPException(status_code=503, detail=runner.last_error or "model_not_ready")
        try:
            result = await run_in_threadpool(runner.edit, request)
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "backend": "diffusers_flux2",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        return {"backend": "diffusers_flux2", **result}

    return app


class Flux2Runner:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.ready = False
        self.last_error: str | None = None
        self.metadata: dict[str, Any] = {"backend": "diffusers_flux2"}
        self.pipe = None
        self._lock = threading.Lock()

    def load(self) -> None:
        import diffusers
        import torch
        import transformers
        from diffusers import Flux2KleinPipeline

        if self.config.dtype != "bf16":
            raise ValueError(f"unsupported dtype: {self.config.dtype}")
        if self.config.device != "cuda":
            raise ValueError(f"unsupported device: {self.config.device}")

        try:
            pipe = Flux2KleinPipeline.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ).to(self.config.device)
            if self.config.attention_backend:
                setter = getattr(pipe.transformer, "set_attention_backend", None)
                if setter is None:
                    raise RuntimeError("unsupported_optimization: attention backend dispatcher unavailable")
                setter(self.config.attention_backend)
            if self.config.cache_mode == "first_block":
                from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache

                apply_first_block_cache(
                    pipe.transformer,
                    FirstBlockCacheConfig(threshold=self.config.cache_threshold),
                )
            elif self.config.cache_mode == "magcache":
                from diffusers import MagCacheConfig

                if not self.config.mag_ratios:
                    raise RuntimeError("unsupported_optimization: magcache requires calibrated mag_ratios")
                pipe.transformer.enable_cache(
                    MagCacheConfig(
                        calibrate=False,
                        mag_ratios=list(self.config.mag_ratios),
                        num_inference_steps=self.config.steps,
                    )
                )
            elif self.config.cache_mode != "none":
                raise RuntimeError(f"unsupported_optimization: cache mode {self.config.cache_mode!r}")
            if self.config.compile:
                pipe.transformer.to(memory_format=torch.channels_last)
                pipe.transformer.compile(mode="default", fullgraph=False)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.ready = False
            raise

        self.pipe = pipe
        self.metadata = {
            "backend": "diffusers_flux2",
            "model_path": self.config.model_path,
            "profile_name": self.config.profile_name,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "diffusers_version": diffusers.__version__,
            "transformers_version": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
        }
        self.last_error = None
        self.ready = True

    def _merged_prompt(self, request: EditRequest) -> str:
        positive = request.positive_prompt.strip()
        region = request.region_prompt.strip()
        if positive and region:
            return f"{positive}\n\nLocal edit directives:\n{region}"
        return positive or region

    def edit(self, request: EditRequest) -> dict[str, Any]:
        import torch
        from PIL import Image

        if not self.ready or self.pipe is None:
            raise RuntimeError(self.last_error or "model_not_ready")
        seed = int(request.seed) if request.seed is not None else self.config.base_seed + int(request.request_index)
        source_path = Path(request.image_path)
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")

        with self._lock:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                result = self.pipe(
                    prompt=self._merged_prompt(request),
                    image=source,
                    height=source.height,
                    width=source.width,
                    num_inference_steps=self.config.steps,
                    sigmas=list(self.config.sigmas),
                    guidance_scale=self.config.cfg,
                    generator=torch.Generator("cuda").manual_seed(seed),
                    max_sequence_length=self.config.max_sequence_length,
                    text_encoder_out_layers=self.config.text_encoder_out_layers,
                )
                edited = result.images[0].convert("RGB")
                runtime_sec = time.perf_counter() - started
                peak_memory_bytes = int(torch.cuda.max_memory_allocated())
            except torch.OutOfMemoryError as exc:
                self.ready = False
                self.last_error = f"OutOfMemoryError: {exc}"
                torch.cuda.empty_cache()
                raise

        raw_size = list(edited.size)
        size_fixed = edited.size != source.size
        if size_fixed:
            edited = edited.resize(source.size, Image.Resampling.LANCZOS)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.request_index:06d}_{source_path.stem}_seed{seed}.png"
        edited.save(output_path)
        return {
            "status": "success",
            "edited_path": str(output_path),
            "seed": seed,
            "runtime_sec": runtime_sec,
            "peak_memory_bytes": peak_memory_bytes,
            "input_size": list(source.size),
            "raw_output_size": raw_size,
            "output_size": list(edited.size),
            "size_fixed": size_fixed,
            "negative_prompt_supported": False,
            "region_prompt_merged": bool(request.region_prompt.strip()),
            "profile_name": self.config.profile_name,
        }


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8212)
    parser.add_argument("--service-name", default="diffusers_flux2")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="outputs/editor")
    parser.add_argument("--profile-name", default="eager_4b_bf16")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sigmas", default="1.0,0.75,0.5,0.25")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--text-encoder-out-layers", default="9,18,27")
    parser.add_argument("--base-seed", type=int, default=764952063587760)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--cache-mode", choices=["none", "first_block", "magcache"], default="none")
    parser.add_argument("--cache-threshold", type=float, default=0.05)
    parser.add_argument("--mag-ratios", default="")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    config = ServiceConfig(
        service_name=args.service_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        profile_name=args.profile_name,
        steps=args.steps,
        cfg=args.cfg,
        sigmas=_csv_floats(args.sigmas),
        max_sequence_length=args.max_sequence_length,
        text_encoder_out_layers=_csv_ints(args.text_encoder_out_layers),
        base_seed=args.base_seed,
        compile=args.compile,
        attention_backend=args.attention_backend,
        cache_mode=args.cache_mode,
        cache_threshold=args.cache_threshold,
        mag_ratios=_csv_floats(args.mag_ratios),
    )
    runner = Flux2Runner(config)
    runner.load()
    uvicorn.run(create_app(runner, config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
