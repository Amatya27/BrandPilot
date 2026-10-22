#!/usr/bin/env python3
"""FastAPI app for the LoRA brand assistant."""

from __future__ import annotations

import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.routing import Match

from brand_assistant_keras_backend import KerasBackendConfig
from brand_assistant_openai_backend import OpenAICompatibleConfig
from brand_assistant_service import BrandAssistantService


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "webapp"
STATIC_DIR = WEB_DIR / "static"


class GenerateRequest(BaseModel):
    company: str = Field(..., min_length=1, max_length=120)
    person: str = Field(default="", max_length=120)


backend = os.environ.get("BRAND_ASSISTANT_BACKEND", "keras")
service = BrandAssistantService(
    backend=backend,
    cache_dir=Path(os.environ.get("BRAND_ASSISTANT_CACHE_DIR", ".cache/brand_assistant")),
    output_dir=Path(os.environ.get("BRAND_ASSISTANT_APP_OUTPUT_DIR", "outputs/app")),
    keras_config=KerasBackendConfig(
        sequence_length=int(os.environ.get("BRAND_ASSISTANT_SEQUENCE_LENGTH", "1536")),
        response_token_budget=int(os.environ.get("BRAND_ASSISTANT_RESPONSE_BUDGET", "650")),
        prompt_passages=int(os.environ.get("BRAND_ASSISTANT_PROMPT_PASSAGES", "4")),
        passage_char_budget=int(os.environ.get("BRAND_ASSISTANT_PASSAGE_CHAR_BUDGET", "280")),
        prompt_style=os.environ.get("BRAND_ASSISTANT_PROMPT_STYLE", "legacy_sections"),
    ),
    openai_config=OpenAICompatibleConfig(
        api_base=os.environ.get("BRAND_ASSISTANT_API_BASE", "http://localhost:8080/v1"),
        model=os.environ.get("BRAND_ASSISTANT_MODEL", "mlx-community/gemma-3-1b-it-qat-4bit"),
        adapter_path=os.environ.get("BRAND_ASSISTANT_ADAPTER_PATH") or None,
        max_tokens=int(os.environ.get("BRAND_ASSISTANT_RESPONSE_BUDGET", "650")),
    ),
)
executor = ThreadPoolExecutor(max_workers=2)
job_lock = Lock()
jobs: dict[str, dict] = {}

app = FastAPI(title="LoRA Brand Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Prometheus metrics. Labels use the route template (/api/jobs/{job_id}), not the raw URL,
# so every job id does not become a new time series.
REQUESTS = Counter("brandpilot_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("brandpilot_http_request_seconds", "HTTP request latency in seconds", ["route"])
JOBS = Counter("brandpilot_generation_jobs_total", "Finished generation jobs", ["result"])
JOB_SECONDS = Histogram(
    "brandpilot_generation_job_seconds", "Time to finish a generation job",
    buckets=(1, 5, 10, 20, 30, 60, 120, 300),
)


def route_template(request: Request) -> str:
    for route in app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", "unmatched")
    return "unmatched"


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    route_path = route_template(request)
    start = time.perf_counter()
    response = await call_next(request)
    LATENCY.labels(route_path).observe(time.perf_counter() - start)
    REQUESTS.labels(request.method, route_path, str(response.status_code)).inc()
    return response


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "backend": service.backend}


def update_job(job_id: str, *, status: str | None = None, message: str | None = None, result: dict | None = None, error: str | None = None) -> None:
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return
        if status:
            job["status"] = status
        if message:
            job["message"] = message
            job.setdefault("events", []).append(message)
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error


def run_generation_job(job_id: str, company: str, person: str) -> None:
    started = time.perf_counter()
    try:
        def progress(step: str, message: str) -> None:
            update_job(job_id, status="running", message=message)

        result = service.generate(company, person, progress_callback=progress)
        update_job(job_id, status="completed", message="Generation completed", result=result)
        JOBS.labels("completed").inc()
    except Exception as exc:
        update_job(job_id, status="failed", message="Generation failed", error=str(exc))
        JOBS.labels("failed").inc()
    finally:
        JOB_SECONDS.observe(time.perf_counter() - started)


@app.post("/api/generate")
async def generate(payload: GenerateRequest) -> dict:
    company = payload.company.strip()
    person = payload.person.strip()
    if not company:
        raise HTTPException(status_code=400, detail="Company name is required.")

    job_id = uuid4().hex
    with job_lock:
        jobs[job_id] = {
            "status": "queued",
            "message": "Queued",
            "events": ["Queued"],
            "result": None,
            "error": None,
        }
    executor.submit(run_generation_job, job_id, company, person)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job
