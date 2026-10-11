"""Small FastAPI inference service for the trained MiniGPT checkpoint.

The service deliberately loads a weights-only checkpoint.  It never trains or
modifies model weights, and exposes conservative request limits so a public
deployment cannot accidentally create very long CPU-bound generations.
"""

from __future__ import annotations

import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlopen

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.generate import generate_text, load_model_checkpoint


GENERATED_TEXT_NOTICE = (
    "Generated text is experimental language-model output and may be incoherent "
    "or inaccurate."
)
DEFAULT_CHECKPOINT = "checkpoints/minigpt_final.pt"

# Lazy loading keeps /health useful while a hosted checkpoint has not yet been
# supplied.  The lock prevents two simultaneous first requests loading it twice.
_runtime: tuple | None = None
_runtime_lock = threading.Lock()
_startup_error: str | None = None


class GenerateRequest(BaseModel):
    """Validated generation controls exposed by POST /generate."""

    prompt: str = Field(..., min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=80, ge=1, le=400)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=65)


class GenerateResponse(BaseModel):
    prompt: str
    generated: str
    model_parameters: int
    context_length: int
    temperature: float
    notice: str


def checkpoint_path() -> Path:
    """Return the preferred checkpoint path, retaining legacy env compatibility."""

    configured = os.getenv("CHECKPOINT_PATH") or os.getenv("MODEL_CHECKPOINT_PATH")
    return Path(configured or DEFAULT_CHECKPOINT)


def ensure_checkpoint() -> Path:
    """Use a local checkpoint or download one from the explicitly configured URL."""

    path = checkpoint_path()
    if path.exists():
        return path

    checkpoint_url = os.getenv("CHECKPOINT_URL") or os.getenv("MODEL_CHECKPOINT_URL")
    if not checkpoint_url:
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Set CHECKPOINT_URL to download it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".download")
    try:
        # Write atomically so an interrupted deployment never leaves a partial .pt
        # file that later requests mistake for a valid checkpoint.
        with urlopen(checkpoint_url, timeout=60) as response, temporary_path.open("wb") as dest:
            shutil.copyfileobj(response, dest)
        temporary_path.replace(path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download checkpoint: {exc}") from exc

    return path


def get_runtime() -> tuple:
    """Load the model once on CPU and return (model, tokenizer, config, device)."""

    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                # The service is deliberately CPU-only. `generate_text` derives
                # its tensor device from this model, so it needs no device argument.
                model, tokenizer, config, _device = load_model_checkpoint(
                    ensure_checkpoint(), torch.device("cpu")
                )
                model = model.to(torch.device("cpu"))
                _runtime = model, tokenizer, config, "cpu"
    return _runtime


def model_metadata() -> dict:
    """Return metadata only after successfully loading the configured checkpoint."""

    model, _tokenizer, config, _device = get_runtime()
    return {
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "context_length": int(config["context_length"]),
        "checkpoint": str(checkpoint_path()),
        "device": "cpu",
        "notice": GENERATED_TEXT_NOTICE,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the checkpoint exactly once before accepting API requests.

    A missing hosted checkpoint is retained as an error state so ``/health`` can
    explain it, rather than repeatedly attempting a network download per request.
    """

    global _startup_error
    try:
        get_runtime()
        _startup_error = None
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        _startup_error = str(exc)
    yield


app = FastAPI(
    title="MiniGPT from Scratch",
    version="1.0.0",
    description=GENERATED_TEXT_NOTICE,
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve a tiny self-contained usage page (Swagger remains at /docs)."""

    return f"""<!doctype html>
<html><head><title>MiniGPT from Scratch</title></head>
<body><h1>MiniGPT from Scratch</h1>
<p><strong>Notice:</strong> {GENERATED_TEXT_NOTICE}</p>
<p>Use <a href=\"/docs\">/docs</a> for interactive Swagger documentation.</p>
<pre>curl -X POST http://localhost:8000/generate \\
  -H \"Content-Type: application/json\" \\
  -d '{{"prompt":"ROMEO:","max_new_tokens":80,"temperature":0.8,"top_k":40}}'</pre>
</body></html>"""


@app.get("/health")
def health() -> dict:
    """Report whether the one startup model-load attempt succeeded."""

    path = checkpoint_path()
    if _startup_error:
        return {"status": "checkpoint_unavailable", "checkpoint": str(path), "detail": _startup_error}
    return {"status": "ok", "checkpoint": str(path)}


@app.get("/model-info")
def model_info() -> dict:
    if _startup_error:
        raise HTTPException(status_code=503, detail=_startup_error)
    try:
        return model_metadata()
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if _startup_error:
        raise HTTPException(status_code=503, detail=_startup_error)
    try:
        model, tokenizer, config, _device = get_runtime()
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=None,
            greedy=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return GenerateResponse(
        prompt=request.prompt,
        generated=generated,
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        context_length=int(config["context_length"]),
        temperature=request.temperature,
        notice=GENERATED_TEXT_NOTICE,
    )

