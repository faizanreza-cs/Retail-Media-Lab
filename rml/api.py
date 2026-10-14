"""REST API: saved queries, analyses, budget optimizer and the Ads Copilot.

Run: ``uvicorn rml.api:app``. When ``RML_API_TOKEN`` is set, every request must send it in the
``X-API-Key`` header (set it before exposing the API, e.g. to a Copilot Studio connector).
"""

from __future__ import annotations

import os
import secrets
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from rml import __version__
from rml.agent import tools
from rml.agent.core import Agent
from rml.agent.provider import get_provider


def check_token(x_api_key: str = Header(default="")) -> None:
    token = os.environ.get("RML_API_TOKEN")
    if token and not secrets.compare_digest(x_api_key, token):
        raise HTTPException(401, "missing or wrong X-API-Key")


app = FastAPI(
    title="retail-media-lab",
    version=__version__,
    description="Amazon Ads analytics: AMC-style queries, attribution, geo incrementality, "
    "marketing mix model, budget optimizer and an analytics agent.",
    dependencies=[Depends(check_token)],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    provider: str
    latency_ms: int


class OptimizeRequest(BaseModel):
    total_daily_budget: float | None = Field(default=None, gt=0)
    max_change: float = Field(default=0.3, ge=0, le=1)
    min_roas: float | None = Field(default=None, gt=0)


def _call(tool: str, **args: Any) -> dict[str, Any]:
    import json

    out, is_error = tools.execute(tool, args)
    payload = json.loads(out)
    if is_error:
        status = 404 if "unknown query" in payload["error"] else 400
        raise HTTPException(status, payload["error"])
    return dict(payload)


@app.get("/health")
def health() -> dict[str, Any]:
    from rml.pipeline import load_results

    res = load_results()
    return {"status": "ok", "backend": res["backend"], "results_generated_at": res["generated_at"]}


@app.get("/queries")
def list_queries() -> dict[str, Any]:
    return _call("list_saved_queries")


@app.get("/queries/{name}")
def run_query(name: str) -> dict[str, Any]:
    return _call("run_saved_query", name=name)


@app.get("/attribution")
def attribution() -> dict[str, Any]:
    return _call("compare_attribution")


@app.get("/incrementality")
def incrementality() -> dict[str, Any]:
    return _call("geo_test_results")


@app.get("/roas")
def roas() -> dict[str, Any]:
    return _call("channel_roas")


@app.post("/optimize")
def optimize_budget(req: OptimizeRequest) -> dict[str, Any]:
    return _call("optimize_budget", **req.model_dump())


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # the provider comes from server settings (RML_PROVIDER), never from the caller
    r = Agent(get_provider()).ask(req.question)
    return AskResponse(
        answer=r.answer, tools_used=r.tools_used, provider=r.provider, latency_ms=r.latency_ms
    )
