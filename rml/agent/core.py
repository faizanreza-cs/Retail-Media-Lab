"""Ads Copilot: question -> tool calls -> grounded answer.

A plain tool-use loop (no framework) so every step is visible and testable:

1. send the question with the tool catalogue,
2. execute each requested tool and return *all* results in one user message,
3. repeat until the model stops calling tools or ``max_steps`` is hit.

The trace (tool names, arguments, results, latency, tokens) is returned alongside the
answer so the API/UI can show its work and the eval harness can grade tool selection.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from rml.agent.provider import LLMProvider, get_provider
from rml.agent.tools import execute, tool_definitions
from rml.config import get_settings

SYSTEM_PROMPT = """\
You are Ads Copilot, an analytics assistant for the Amazon Ads team of Fernway Nutrition, a
supplements brand. The data is synthetic, shaped like Amazon Marketing Cloud: shopper-level ad
events and purchases for 60 days, plus two years of daily spend and sales for 20 geos. Channels:
Sponsored Products (SP), Sponsored Brands (SB), Sponsored Display (SD) and Amazon DSP.

How to answer:
- Use the tools for every number. Never estimate or recall figures; if no tool can answer, say
  so. Prefer a purpose-built tool or a saved query over run_sql.
- Lead with the answer in one or two sentences, then the supporting figures (a short table is
  fine), then one line on method or caveats.
- Be clear about the strength of evidence. Attribution shows who saw ads before buying, not what
  the ads caused. The geo test is an experiment. The marketing mix model's intervals show how
  sure it is. When they disagree, say so and give the experiment more weight.
- Money in dollars with thousands separators, shares as percentages with one decimal, ROAS with
  two decimals.
- Be concise. The audience is a marketing director deciding budgets.
"""


@dataclass
class Step:
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]
    is_error: bool
    latency_ms: int


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    stop_reason: str = ""
    latency_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tools_used"] = self.tools_used
        return d


class Agent:
    def __init__(self, provider: LLMProvider | None = None, max_steps: int | None = None) -> None:
        self.provider = provider or get_provider()
        self.max_steps = max_steps or get_settings().max_agent_steps
        self.tools = tool_definitions()

    def ask(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        result = AgentResult(
            question=question, answer="", provider=self.provider.name, model=self.provider.model
        )
        usage = {"input_tokens": 0, "output_tokens": 0}

        for _ in range(self.max_steps):
            turn = self.provider.complete(SYSTEM_PROMPT, messages, self.tools)
            for k, v in turn.usage.items():
                usage[k] = usage.get(k, 0) + v
            result.stop_reason = turn.stop_reason
            if turn.stop_reason == "refusal":
                result.answer = turn.text
                break
            messages.append({"role": "assistant", "content": turn.content})
            if not turn.tool_calls:
                result.answer = turn.text
                break
            tool_results = []
            for call in turn.tool_calls:
                s0 = time.perf_counter()
                out, is_error = execute(call.name, call.input)
                result.steps.append(
                    Step(
                        tool=call.name,
                        input=call.input,
                        output=json.loads(out),
                        is_error=is_error,
                        latency_ms=int((time.perf_counter() - s0) * 1000),
                    )
                )
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": out,
                }
                if is_error:
                    block["is_error"] = True
                tool_results.append(block)
            # all results for this turn go back in a single user message
            messages.append({"role": "user", "content": tool_results})
        else:
            result.answer = result.answer or (
                "I hit the step limit before finishing. Partial results are in the trace."
            )
            result.stop_reason = "max_steps"

        result.usage = usage
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        return result


def ask(question: str, provider: str | None = None) -> AgentResult:
    return Agent(get_provider(provider)).ask(question)
