"""FastAPI service.

Design constraints come from the deployment target rather than from taste. This
runs on Cloud Run's always-free tier, which means scale to zero, one vCPU, 512
MiB, and a cold start on the first request after idle. So:

- The model is loaded once at import and held in module state. It is 0.57 MB,
  so this costs milliseconds, and baking it into the image avoids a bucket
  fetch on every cold start.
- ``/healthz`` never touches the model, so a liveness probe cannot be blocked by
  inference work.
- ``/readyz`` does touch it, because a container that is up but cannot score is
  not ready.
- Batch is a first-class endpoint. Scoring 500 domains in one request costs a
  fraction of 500 requests, and DNS log triage is inherently a batch problem.
- Every response carries the model version and the inference time, because a
  latency claim you cannot reproduce from the API itself is a latency claim
  nobody should believe.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from pydantic import BaseModel, Field

from .predictor import Predictor, Verdict

MODEL_PATH = Path(os.getenv("NXS_MODEL_PATH", "artifacts/serving_model.pt"))
MAX_BATCH = int(os.getenv("NXS_MAX_BATCH", "500"))
MAX_BODY_BYTES = 1024 * 1024
DomainInput = Annotated[str, Field(max_length=253)]

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    if _predictor is None:
        raise HTTPException(503, "model not loaded")
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = Predictor.load(MODEL_PATH, os.getenv("NXS_MODEL_SHA256"))
    # One throwaway inference at startup. The first call through torch pays for
    # lazy kernel setup, and paying it here rather than on a user's request is
    # the difference between a 400 ms cold start and a 2 s one.
    _predictor.predict_batch(["warmup-domain"])
    yield
    _predictor = None


app = FastAPI(
    title="nxdomain-sentinel",
    description="Domain threat classifier scored on malware families it has never seen.",
    version="0.1.0",
    lifespan=lifespan,
)


class ScoreBodyLimit:
    """Bound JSON request bytes before parsing, including chunked uploads."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in {"/v1/score", "/v1/score/batch",
                                     "/v1/score/", "/v1/score/batch/"}
        ):
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                await JSONResponse({"detail": "request body too large"}, 413)(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        async def replay():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


app.add_middleware(ScoreBodyLimit)


class ScoreRequest(BaseModel):
    domain: DomainInput = Field(examples=["mortiscontrastatim.com"])


class BatchRequest(BaseModel):
    domains: list[DomainInput] = Field(..., max_length=MAX_BATCH)


class ScoreResponse(BaseModel):
    domain: str
    core: str
    verdict: str
    dga_probability: float
    nearest_family: str | None
    novelty_score: float | None
    is_novel: bool
    above_novelty_threshold: bool | None = None
    note: str | None = None


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    count: int
    inference_ms: float
    model_version: str
    model_sha256: str | None = None


def _to_response(v: Verdict) -> ScoreResponse:
    return ScoreResponse(**v.to_dict())


@app.get("/healthz")
def healthz() -> dict:
    """Liveness. Deliberately does not touch the model."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    """Readiness. Fails while the model is unavailable."""
    p = get_predictor()
    return {"status": "ready", "model_version": p.version, "families": len(p.families)}


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    p = get_predictor()
    return _to_response(p.predict_batch([req.domain])[0])


@app.post("/v1/score/batch", response_model=BatchResponse)
def score_batch(req: BatchRequest) -> BatchResponse:
    if not req.domains:
        raise HTTPException(400, "domains must not be empty")
    if len(req.domains) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} domains per request")

    p = get_predictor()
    t0 = time.perf_counter()
    verdicts = p.predict_batch(req.domains)
    elapsed = (time.perf_counter() - t0) * 1000

    return BatchResponse(
        results=[_to_response(v) for v in verdicts],
        count=len(verdicts),
        inference_ms=round(elapsed, 3),
        model_version=p.version,
        model_sha256=p.artifact_sha256,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text()
