import json
from types import SimpleNamespace

import pytest

from rml.agent import provider
from rml.agent.core import Agent
from rml.agent.provider import MockProvider, ToolCall, Turn
from rml.agent.tools import execute, tool_definitions
from rml.evals import run_evals


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("What are the most common ad paths before a purchase?", "run_saved_query"),
        ("Which attribution method is closest to the truth?", "compare_attribution"),
        ("What incremental ROAS did the DSP geo holdout show?", "geo_test_results"),
        ("Which channel has the highest marginal ROAS?", "channel_roas"),
        ("How should we reallocate the budget?", "optimize_budget"),
        ("Using SQL, how many unique purchasers are there?", "run_sql"),
    ],
)
def test_mock_agent_picks_the_right_tool(question, tool):
    r = Agent(MockProvider()).ask(question)
    assert r.tools_used == [tool]
    assert not r.steps[0].is_error and r.answer


def test_tool_errors_go_back_to_the_model_not_up():
    out, err = execute("run_sql", {"sql": "SELECT * FROM conversions"})
    assert err and "shopper ids" in json.loads(out)["error"]
    out, err = execute("no_such_tool", {})
    assert err
    out, err = execute("optimize_budget", {"max_change": 3})
    assert err and "max_change" in json.loads(out)["error"]


def test_tool_schemas_are_complete():
    for t in tool_definitions():
        assert t["description"] and t["input_schema"]["type"] == "object"


def test_rogue_model_cant_run_unsafe_sql():
    class Rogue:
        name, model = "rogue", "rogue"

        def __init__(self):
            self.n = 0

        def complete(self, system, messages, tools):
            self.n += 1
            if self.n == 1:
                call = ToolCall("t1", "run_sql", {"sql": "DROP TABLE conversions"})
                block = {"type": "tool_use", "id": "t1", "name": "run_sql", "input": call.input}
                return Turn("", [call], "tool_use", [block])
            return Turn("done", [], "end_turn", [{"type": "text", "text": "done"}])

    r = Agent(Rogue()).ask("drop it")
    assert r.steps[0].is_error and "SELECT" in r.steps[0].output["error"]


def _openai_response(text=None, calls=()):
    tool_calls = [
        SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
        for i, (n, a) in enumerate(calls)
    ]
    msg = SimpleNamespace(content=text, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeCompletions:
    def __init__(self):
        self.sent = {}

    def create(self, **kw):
        self.sent.update(kw)
        return _openai_response(calls=[("channel_roas", {})])


def test_openai_and_azure_providers_translate_both_ways(monkeypatch):
    monkeypatch.setenv("RML_MODEL", "deployment-1")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    for cls in (provider.OpenAIProvider, provider.AzureOpenAIProvider):
        prov = cls()
        fake = _FakeCompletions()
        prov.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        turn = prov.complete("sys", [{"role": "user", "content": "roas?"}], tool_definitions())
        assert turn.tool_calls[0].name == "channel_roas" and turn.stop_reason == "tool_use"
        assert fake.sent["model"] == "deployment-1"
        assert fake.sent["messages"][0]["role"] == "system"


def test_anthropic_provider_parses_tool_use(monkeypatch):
    monkeypatch.setenv("RML_MODEL", "model-under-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    prov = provider.AnthropicProvider()
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="t1", name="channel_roas", input={})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    prov.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    turn = prov.complete("sys", [{"role": "user", "content": "roas?"}], tool_definitions())
    assert turn.tool_calls[0].name == "channel_roas" and turn.usage["input_tokens"] == 10


def test_golden_eval_passes_in_mock_mode():
    rows = run_evals.run("mock")
    summary = run_evals.summarise(rows, "mock")
    assert summary["pass_rate"] >= 0.95, [r.id for r in rows if not (r.tool_ok and r.answer_ok)]
