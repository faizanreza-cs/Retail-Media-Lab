"""Provider-agnostic LLM interface.

The agent speaks one message format (Anthropic-style content blocks) and every provider
translates to and from it. That keeps the agent loop identical whether it runs on the
Anthropic API, OpenAI, Azure OpenAI, or the deterministic ``MockProvider`` used in tests and CI.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from rml.config import Settings, get_settings


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Turn:
    """One assistant response, normalised across providers."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "refusal" | "max_tokens" | ...
    # Exactly what to append to history as the assistant message content. For Anthropic this
    # is the raw response content (thinking blocks must be echoed back unchanged).
    content: list[Any]
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn: ...


# --------------------------------------------------------------------------- Anthropic


def _require_model(s: Settings, provider: str) -> str:
    if not s.model:
        raise ValueError(
            f"RML_MODEL is not set; set it to the {provider} model id to use, "
            "or use RML_PROVIDER=mock"
        )
    return s.model


class AnthropicProvider:
    """Anthropic Messages API via the official ``anthropic`` SDK."""

    name = "anthropic"

    def __init__(self, settings: Settings | None = None) -> None:
        import anthropic

        s = settings or get_settings()
        self.model = _require_model(s, "Anthropic")
        self.effort = s.effort
        self.fallbacks = s.fallbacks
        self.client = anthropic.Anthropic()

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": messages,
            "tools": tools,
            "output_config": {"effort": self.effort},
        }
        if self.fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
            resp = self.client.beta.messages.create(**kwargs)
        else:
            resp = self.client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, input=dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        if resp.stop_reason == "refusal":
            text = text or "The model declined to answer this request."
        return Turn(
            text=text,
            tool_calls=calls,
            stop_reason=str(resp.stop_reason),
            content=list(resp.content),
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )


# --------------------------------------------------------------------------- OpenAI


def _block(b: Any, key: str, default: Any = None) -> Any:
    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)


class OpenAIProvider:
    """Optional OpenAI backend (``pip install "retail-media-lab[llm]"``)."""

    name = "openai"

    def __init__(self, settings: Settings | None = None) -> None:
        from openai import OpenAI

        s = settings or get_settings()
        self.model = os.environ.get("RML_OPENAI_MODEL") or _require_model(s, "OpenAI")
        self.client = OpenAI()

    @staticmethod
    def _convert(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            if m["role"] == "assistant":
                text = "".join(
                    _block(b, "text", "") for b in content if _block(b, "type") == "text"
                )
                calls = [
                    {
                        "id": _block(b, "id"),
                        "type": "function",
                        "function": {
                            "name": _block(b, "name"),
                            "arguments": json.dumps(_block(b, "input")),
                        },
                    }
                    for b in content
                    if _block(b, "type") == "tool_use"
                ]
                msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:
                for b in content:
                    if _block(b, "type") == "tool_result":
                        out.append(
                            {
                                "role": "tool",
                                "tool_call_id": b["tool_use_id"],
                                "content": b["content"],
                            }
                        )
                    elif _block(b, "type") == "text":
                        out.append({"role": "user", "content": b["text"]})
        return out

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        oa_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        # plain dicts in the documented shape; the SDK's TypedDicts add nothing at runtime
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, self._convert(system, messages)),
            tools=cast(Any, oa_tools),
        )
        msg = resp.choices[0].message
        calls = []
        for c in msg.tool_calls or []:
            fn = getattr(c, "function", None)  # only function tools are defined
            if fn is not None:
                try:
                    args = json.loads(fn.arguments or "{}")
                except json.JSONDecodeError:  # e.g. cut off at max_tokens; the tool says so
                    args = {"_invalid_arguments": True}
                calls.append(ToolCall(id=c.id, name=fn.name, input=args))
        content: list[Any] = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        content += [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls
        ]
        return Turn(
            text=msg.content or "",
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            content=content,
        )


# --------------------------------------------------------------------------- Azure OpenAI


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and RML_MODEL to your
    deployment name (AZURE_OPENAI_API_VERSION is optional)."""

    name = "azure"

    def __init__(self, settings: Settings | None = None) -> None:
        from openai import AzureOpenAI

        s = settings or get_settings()
        self.model = _require_model(s, "Azure OpenAI deployment")
        self.client = AzureOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
        )


# --------------------------------------------------------------------------- Mock


@dataclass
class _Route:
    pattern: str
    tool: str
    args: dict[str, Any]


# Keyword router: maps a question to the tool an analyst would reach for. Used for offline demos,
# tests and CI evals of the tooling (not of model quality). Order matters: first match wins.
_ROUTES: list[_Route] = [
    _Route(r"saved quer|which quer|what quer|list (the )?quer", "list_saved_queries", {}),
    _Route(r"path|journey|sequence|order of ad", "run_saved_query",
           {"name": "path_to_conversion"}),
    _Route(r"new.to.brand|\bntb\b|new customer", "run_saved_query", {"name": "new_to_brand"}),
    _Route(r"frequency|how many (dsp )?impressions", "run_saved_query",
           {"name": "frequency_vs_conversion"}),
    _Route(r"overlap|combination", "run_saved_query", {"name": "ad_type_overlap"}),
    _Route(r"how long|time to (purchase|conversion|convert)|days (to|between)", "run_saved_query",
           {"name": "time_to_conversion"}),
    _Route(r"campaign", "run_saved_query", {"name": "campaign_reach"}),
    _Route(r"geo|holdout|experiment|incrementality|\biroas\b|lift test", "geo_test_results", {}),
    _Route(r"attribut|last.touch|first.touch|credit|markov|shapley", "compare_attribution", {}),
    _Route(r"budget|allocat|optimi|shift|reallocat|where should|spend more|move money",
           "optimize_budget", {}),
    _Route(r"roas|return on ad spend|marketing mix|\bmmm\b|saturat|marginal|diminishing",
           "channel_roas", {}),
]  # fmt: skip

# Canned SQL so the mock exercises run_sql end to end when a question asks for SQL.
_SQL_TEMPLATES: list[tuple[str, str]] = [
    (r"purchasers|buyers", "SELECT COUNT(DISTINCT user_id) AS purchasers FROM conversions"),
    (r"spend by channel|spend per channel",
     "SELECT channel, ROUND(SUM(spend), 0) AS spend FROM daily_geo_media GROUP BY channel "
     "ORDER BY spend DESC"),
    (r"sales by (asin|product)",
     "SELECT a.product_name, ROUND(SUM(c.product_sales), 2) AS sales FROM conversions c "
     "JOIN asins a ON a.tracked_asin = c.tracked_asin GROUP BY a.product_name ORDER BY sales DESC"),
]  # fmt: skip

_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?", re.I)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def mock_route(question: str) -> tuple[str, dict[str, Any]]:
    """Deterministically choose a tool + arguments for a question."""
    q = question.lower()
    if "sql" in q:
        for pattern, sql in _SQL_TEMPLATES:
            if re.search(pattern, q):
                return "run_sql", {"sql": sql}
    for r in _ROUTES:
        if re.search(r.pattern, q):
            args: dict[str, Any] = dict(r.args)
            if r.tool == "optimize_budget":
                money = _MONEY.search(question)
                if money and "budget" in q:
                    amount = float(money.group(1).replace(",", ""))
                    args["total_daily_budget"] = amount * (1000 if money.group(2) else 1)
                pct = _PCT.search(question)
                if pct and re.search(r"move|change|shift|cap|limit", q):
                    args["max_change"] = float(pct.group(1)) / 100
            return r.tool, args
    return "run_sql", {
        "sql": "SELECT channel, ROUND(SUM(spend), 0) AS spend FROM daily_geo_media "
        "GROUP BY channel ORDER BY spend DESC"
    }


class MockProvider:
    """Deterministic stand-in for an LLM: routes by keywords, then summarises tool output."""

    name = "mock"
    model = "mock-router-v1"

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Turn:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            tool, args = mock_route(last["content"])
            call = ToolCall(id=f"mock_{len(messages)}", name=tool, input=args)
            content = [{"type": "tool_use", "id": call.id, "name": tool, "input": args}]
            return Turn(text="", tool_calls=[call], stop_reason="tool_use", content=content)
        # after tool results: produce a grounded summary of the first result
        result = next(
            b["content"]
            for b in last["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        )
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = {"result": result}
        text = "Here is what the data shows (mock mode, no LLM call):\n\n" + _summarise(payload)
        return Turn(
            text=text,
            tool_calls=[],
            stop_reason="end_turn",
            content=[{"type": "text", "text": text}],
        )


def _summarise(payload: dict[str, Any]) -> str:
    if "error" in payload:
        return f"The tool returned an error: {payload['error']}"
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        keys = list(rows[0].keys())
        head = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
        body = "\n".join("| " + " | ".join(str(r.get(k)) for k in keys) + " |" for r in rows[:10])
        note = f"\n\n{payload['note']}" if payload.get("note") else ""
        return head + body + note
    return "```json\n" + json.dumps(payload, indent=2, default=str)[:2000] + "\n```"


# --------------------------------------------------------------------------- factory


def _has_anthropic_credentials() -> bool:
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_PROFILE")
    )


def get_provider(name: str | None = None, settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    name = (name or s.provider).lower()
    if name == "auto":
        name = "anthropic" if _has_anthropic_credentials() and s.model else "mock"
    if name == "anthropic":
        return AnthropicProvider(s)
    if name == "openai":
        return OpenAIProvider(s)
    if name == "azure":
        return AzureOpenAIProvider(s)
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider {name!r} (anthropic | openai | azure | mock | auto)")
