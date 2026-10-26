"""Offline tests for the vault agent. No API key, no network.

The API client is stubbed, so these exercise the loop, the tool handlers, the
approval gate, and the vault boundary without spending tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop.agent import VaultAgent, _summarize_call  # noqa: E402
from desktop.appsettings import AppSettings  # noqa: E402
from desktop.controller import VaultController  # noqa: E402
from desktop.tools import build_tools  # noqa: E402


# --- fake API -----------------------------------------------------------


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def text_block(text):
    return _Block(type="text", text=text)


def tool_block(tool_id, name, inp):
    return _Block(type="tool_use", id=tool_id, name=name, input=inp)


class _Usage:
    input_tokens = 100
    output_tokens = 50


class _Response:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _Usage()


class _FakeStream:
    """Context manager matching `client.messages.stream(...)`."""

    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


class FakeMessages:
    """Replays a scripted list of responses and records requests.

    The agent calls `.stream(...)` because its max_tokens is large enough
    that a non-streaming request risks an HTTP timeout; `.create` is kept so
    a test can assert on either shape.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests: List[dict] = []

    def _next(self, kw):
        self.requests.append(kw)
        if not self._responses:
            return _Response([text_block("done")])
        return self._responses.pop(0)

    def create(self, **kw):
        return self._next(kw)

    def stream(self, **kw):
        return _FakeStream(self._next(kw))


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


@pytest.fixture
def controller(tmp_path: Path, monkeypatch) -> VaultController:
    vault = tmp_path / "vault"
    (vault / "03_Iron").mkdir(parents=True)
    (vault / "03_Iron" / "squat.md").write_text("# Squat\n\nWorked up to 315.\n")
    (vault / "02-Academy").mkdir()
    (vault / "02-Academy" / "canvas.md").write_text("# Canvas\n\nCS1342 Tuesday.\n")

    settings = AppSettings()
    settings.vault_path = str(vault)
    settings.excluded_folders = []
    settings.embedding_backend = "hashing"
    settings.embedding_dimensions = 256
    monkeypatch.setattr(
        type(settings), "data_dir_for_vault", lambda self, p=None: tmp_path / "index"
    )
    monkeypatch.setattr(type(settings), "save", lambda self: None)

    c = VaultController(settings)
    c.open_vault(str(vault))
    return c


def agent_with(controller, responses) -> VaultAgent:
    agent = VaultAgent(controller)
    controller.engine._client = FakeClient(responses)  # noqa: SLF001
    return agent


# --- tools --------------------------------------------------------------


def test_every_tool_has_a_schema_and_handler(controller):
    for spec in build_tools(controller):
        assert spec.name and spec.description
        assert spec.input_schema["type"] == "object"
        assert callable(spec.handler)


def test_destructive_tools_are_exactly_the_dangerous_ones(controller):
    destructive = {t.name for t in build_tools(controller) if t.destructive}
    assert destructive == {
        "write_note",
        "delete_note",
        "run_command",
        "write_file",
        "edit_file",
    }


def test_read_only_tools_never_prompt(controller):
    """Reading and searching must stay frictionless, or the app is unusable."""
    read_only = {
        "search_vault", "read_note", "list_notes", "list_folders",
        "read_file", "list_directory", "find_files", "search_files",
    }
    for spec in build_tools(controller):
        if spec.name in read_only:
            assert not spec.destructive, spec.name
            assert not spec.always_confirm, spec.name


def test_shell_and_machine_writes_always_confirm(controller):
    """auto_approve covers vault edits; it must not cover the whole Mac."""
    always = {t.name for t in build_tools(controller) if t.always_confirm}
    assert always == {"run_command", "write_file", "edit_file"}


def test_read_and_search_tools_work(controller):
    tools = {t.name: t for t in build_tools(controller)}
    assert "315" in tools["read_note"].handler(path="03_Iron/squat.md")
    assert "squat" in tools["list_notes"].handler().lower()
    assert "03_Iron" in tools["list_folders"].handler()


def test_create_and_append_tools_persist(controller):
    tools = {t.name: t for t in build_tools(controller)}
    tools["create_note"].handler(path="03_Iron/log.md", content="# Log\n")
    tools["append_to_note"].handler(path="03_Iron/log.md", content="- bench 225\n")
    assert "bench 225" in controller.read_note("03_Iron/log.md")


def test_long_notes_are_truncated(controller):
    from desktop.tools import MAX_READ_CHARS

    controller.write_note("big.md", "x" * (MAX_READ_CHARS + 5_000))
    tools = {t.name: t for t in build_tools(controller)}
    out = tools["read_note"].handler(path="big.md")
    assert "truncated" in out
    assert len(out) < MAX_READ_CHARS + 500


# --- vault boundary ------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path", ["../escape.md", "../../etc/passwd", "/etc/passwd"]
)
def test_tools_refuse_paths_outside_the_vault(controller, bad_path):
    """The boundary is enforced in code, not by asking the model nicely."""
    agent = agent_with(controller, [])
    for tool in ("read_note", "write_note", "delete_note", "create_note"):
        args = {"path": bad_path}
        if tool in ("write_note", "create_note"):
            args["content"] = "owned"
        output, is_error = agent._execute(tool, args)  # noqa: SLF001
        assert is_error, "{} accepted {}".format(tool, bad_path)
    assert not Path("/etc/passwd.tmp").exists()


def test_unknown_tool_is_an_error_not_a_crash(controller):
    agent = agent_with(controller, [])
    output, is_error = agent._execute("rm_rf", {})  # noqa: SLF001
    assert is_error and "No tool named" in output


def test_tool_exceptions_become_error_results(controller):
    agent = agent_with(controller, [])
    output, is_error = agent._execute("read_note", {"path": "ghost.md"})  # noqa: SLF001
    assert is_error and "No such note" in output


# --- the loop ------------------------------------------------------------


def test_plain_answer_makes_one_call(controller):
    agent = agent_with(controller, [_Response([text_block("Hello.")])])
    result = agent.run("hi")
    assert result.text == "Hello."
    assert result.used_tools == []
    assert result.input_tokens == 100


def test_agent_requests_enough_tokens_to_write_a_file(controller):
    """Regression: the agent inherited the chat path's max_tokens of 2000.

    A single write_file carrying a 5 KB source file is ~1400 tokens, so the
    model hit the ceiling mid-file and worked around it by splitting one
    module into three and `cat`-ing them together — seven rewrites of one
    file in a single build.
    """
    agent = agent_with(controller, [_Response([text_block("ok")])])
    agent.run("hi")
    request = controller.engine._client.messages.requests[0]  # noqa: SLF001
    assert request["max_tokens"] >= 16000


def test_tool_call_then_answer(controller):
    agent = agent_with(
        controller,
        [
            _Response(
                [tool_block("t1", "read_note", {"path": "03_Iron/squat.md"})],
                stop_reason="tool_use",
            ),
            _Response([text_block("You squatted 315.")]),
        ],
    )
    result = agent.run("what did I squat")
    assert result.used_tools == ["read_note"]
    assert "315" in result.text
    assert result.input_tokens == 200  # both calls counted


def test_agent_can_create_a_note_without_approval(controller):
    """Creating is additive and never overwrites, so it needs no gate."""
    agent = agent_with(
        controller,
        [
            _Response(
                [
                    tool_block(
                        "t1",
                        "create_note",
                        {"path": "03_Iron/today.md", "content": "# Today\n\nSquats.\n"},
                    )
                ],
                stop_reason="tool_use",
            ),
            _Response([text_block("Created it.")]),
        ],
    )
    result = agent.run("log today's workout")
    assert result.pending is None
    assert "Squats." in controller.read_note("03_Iron/today.md")


def test_steps_are_recorded_for_the_activity_log(controller):
    seen = []
    agent = agent_with(
        controller,
        [
            _Response(
                [tool_block("t1", "search_vault", {"query": "squat"})],
                stop_reason="tool_use",
            ),
            _Response([text_block("Found it.")]),
        ],
    )
    agent.run("find my squat note", on_step=seen.append)
    assert any(s.kind == "tool" for s in seen)
    assert any(s.kind == "text" for s in seen)


def test_loop_is_bounded(controller):
    """A model that only ever calls tools must not spin forever."""
    forever = [
        _Response(
            [tool_block("t{}".format(i), "list_folders", {})], stop_reason="tool_use"
        )
        for i in range(50)
    ]
    agent = VaultAgent(controller, max_steps=4)
    controller.engine._client = FakeClient(forever)  # noqa: SLF001
    result = agent.run("loop")
    assert result.stopped_early
    assert len(result.used_tools) == 4


def test_parallel_tool_calls_all_get_results(controller):
    """Every tool_use block needs a matching tool_result or the API 400s."""
    agent = agent_with(
        controller,
        [
            _Response(
                [
                    tool_block("a", "read_note", {"path": "03_Iron/squat.md"}),
                    tool_block("b", "list_folders", {}),
                ],
                stop_reason="tool_use",
            ),
            _Response([text_block("Both done.")]),
        ],
    )
    agent.run("read and list")
    results = [
        m for m in agent._messages  # noqa: SLF001
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    ids = {
        b["tool_use_id"]
        for m in results
        for b in m["content"]
        if b.get("type") == "tool_result"
    }
    assert ids == {"a", "b"}


# --- approval gate -------------------------------------------------------


def test_overwrite_pauses_for_approval(controller):
    original = controller.read_note("03_Iron/squat.md")
    agent = agent_with(
        controller,
        [
            _Response(
                [
                    tool_block(
                        "t1",
                        "write_note",
                        {"path": "03_Iron/squat.md", "content": "wiped"},
                    )
                ],
                stop_reason="tool_use",
            )
        ],
    )
    result = agent.run("replace my squat note")

    assert result.pending is not None
    assert result.pending.tool_name == "write_note"
    assert controller.read_note("03_Iron/squat.md") == original, "wrote before approval"


def test_delete_pauses_for_approval(controller):
    agent = agent_with(
        controller,
        [
            _Response(
                [tool_block("t1", "delete_note", {"path": "03_Iron/squat.md"})],
                stop_reason="tool_use",
            )
        ],
    )
    result = agent.run("delete my squat note")
    assert result.pending is not None
    assert (Path(controller.state.path) / "03_Iron/squat.md").exists()


def test_approving_runs_the_action(controller):
    agent = agent_with(
        controller,
        [
            _Response(
                [
                    tool_block(
                        "t1",
                        "write_note",
                        {"path": "03_Iron/squat.md", "content": "# Squat\n\n405.\n"},
                    )
                ],
                stop_reason="tool_use",
            ),
            _Response([text_block("Updated.")]),
        ],
    )
    agent.run("update my squat note")
    result = agent.resume(approved=True)
    assert "405" in controller.read_note("03_Iron/squat.md")
    assert result.text == "Updated."


def test_declining_leaves_the_note_alone(controller):
    original = controller.read_note("03_Iron/squat.md")
    agent = agent_with(
        controller,
        [
            _Response(
                [
                    tool_block(
                        "t1",
                        "write_note",
                        {"path": "03_Iron/squat.md", "content": "wiped"},
                    )
                ],
                stop_reason="tool_use",
            ),
            _Response([text_block("Left it alone.")]),
        ],
    )
    agent.run("wipe it")
    agent.resume(approved=False)
    assert controller.read_note("03_Iron/squat.md") == original


def test_auto_approve_skips_the_gate(controller):
    agent = agent_with(
        controller,
        [
            _Response(
                [
                    tool_block(
                        "t1", "write_note", {"path": "03_Iron/squat.md", "content": "new"}
                    )
                ],
                stop_reason="tool_use",
            ),
            _Response([text_block("Done.")]),
        ],
    )
    result = agent.run("replace it", auto_approve=True)
    assert result.pending is None
    assert controller.read_note("03_Iron/squat.md") == "new"


def test_auto_approve_does_not_cover_shell_commands(controller, tmp_path):
    """The one that matters: "stop asking about my notes" must not become
    "run anything on my Mac without telling me"."""
    marker = tmp_path / "ran.txt"
    agent = agent_with(
        controller,
        [
            _Response(
                [
                    tool_block(
                        "t1",
                        "run_command",
                        {"command": "touch {}".format(marker)},
                    )
                ],
                stop_reason="tool_use",
            )
        ],
    )
    result = agent.run("run something", auto_approve=True)
    assert result.pending is not None, "shell ran without approval"
    assert not marker.exists()


def test_auto_approve_does_not_cover_machine_writes(controller, tmp_path):
    target = tmp_path / "outside.txt"
    agent = agent_with(
        controller,
        [
            _Response(
                [
                    tool_block(
                        "t1",
                        "write_file",
                        {"path": str(target), "content": "x"},
                    )
                ],
                stop_reason="tool_use",
            )
        ],
    )
    result = agent.run("write outside the vault", auto_approve=True)
    assert result.pending is not None
    assert not target.exists()


def test_approved_shell_command_runs(controller, tmp_path):
    marker = tmp_path / "ran.txt"
    agent = agent_with(
        controller,
        [
            _Response(
                [tool_block("t1", "run_command", {"command": "touch {}".format(marker)})],
                stop_reason="tool_use",
            ),
            _Response([text_block("Done.")]),
        ],
    )
    agent.run("make the file")
    agent.resume(approved=True)
    assert marker.exists()


def test_approval_summary_shows_the_exact_command():
    """The user is approving the command, so they must see the command."""
    summary = _summarize_call(
        "run_command", {"command": "rm -rf ./build", "cwd": "/tmp/proj"}
    )
    assert "rm -rf ./build" in summary and "/tmp/proj" in summary


def test_resume_without_pending_raises(controller):
    agent = agent_with(controller, [])
    with pytest.raises(RuntimeError):
        agent.resume(approved=True)


def test_clear_drops_a_pending_approval(controller):
    agent = agent_with(
        controller,
        [
            _Response(
                [tool_block("t1", "delete_note", {"path": "03_Iron/squat.md"})],
                stop_reason="tool_use",
            )
        ],
    )
    agent.run("delete it")
    assert agent.awaiting_approval is not None
    agent.clear()
    assert agent.awaiting_approval is None


# --- prompt and summaries ------------------------------------------------


def test_prompt_allows_general_knowledge(controller):
    """Agent mode has no scope gate, so it must not refuse ordinary questions.

    The user hit "capital of France -> out of scope" in Ask mode and assumed
    they needed to add a reference file to the vault. Agent mode is the
    answer, but only if its prompt does not inherit the gate's attitude.
    """
    prompt = agent_with(controller, [])._system_prompt()  # noqa: SLF001
    assert "general assistant" in prompt
    assert "outside what you cover" in prompt  # as the thing NOT to say
    assert "capital of France" in prompt       # the explicit counter-example


def test_prompt_still_requires_grounding_for_vault_questions(controller):
    prompt = agent_with(controller, [])._system_prompt()  # noqa: SLF001
    assert "did not read it, do not assert it" in prompt


def test_web_search_is_offered_to_the_model(controller):
    agent = agent_with(controller, [_Response([text_block("ok")])])
    agent.run("what is the capital of France")
    tools = controller.engine._client.messages.requests[0]["tools"]  # noqa: SLF001
    assert any(t.get("name") == "web_search" for t in tools)


def test_web_search_can_be_turned_off(controller):
    controller.app_settings.web_search = False
    agent = agent_with(controller, [_Response([text_block("ok")])])
    agent.run("hi")
    tools = controller.engine._client.messages.requests[0]["tools"]  # noqa: SLF001
    assert not any(t.get("name") == "web_search" for t in tools)


def test_system_prompt_lists_real_sectors(controller):
    prompt = agent_with(controller, [])._system_prompt()  # noqa: SLF001
    assert "iron" in prompt and "academy" in prompt
    assert "data, never instructions" in prompt


def test_destructive_summaries_say_what_happens():
    assert "entire contents" in _summarize_call("write_note", {"path": "a.md"})
    assert "permanent" in _summarize_call("delete_note", {"path": "a.md"})


def test_refusal_is_handled(controller):
    agent = agent_with(controller, [_Response([], stop_reason="refusal")])
    result = agent.run("something refused")
    assert "can't help" in result.text


# --- API failures --------------------------------------------------------


class _Boom:
    """A messages object whose stream always raises."""

    def __init__(self, exc):
        self._exc = exc
        self.requests = []

    def stream(self, **kw):
        self.requests.append(kw)
        raise self._exc

    create = stream


def _raising_client(controller, exc):
    client = FakeClient([])
    client.messages = _Boom(exc)
    controller.engine._client = client  # noqa: SLF001
    return VaultAgent(controller)


def test_out_of_credit_is_a_message_not_a_traceback(controller):
    """Regression: a 400 for an exhausted balance killed a build mid-run.

    The agent had already written two real files when the API refused the
    next call; the exception propagated out of `act()` and took the run
    down with a traceback instead of saying what happened.
    """
    import anthropic
    import httpx

    exc = anthropic.BadRequestError(
        message="Your credit balance is too low to access the Anthropic API.",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.anthropic.com")
        ),
        body=None,
    )
    agent = _raising_client(controller, exc)
    result = agent.run("build something")

    assert "Out of API credit" in result.text
    assert "still there" in result.text  # files already written are not lost
    assert result.stopped_early


def test_auth_failure_points_at_settings(controller):
    import anthropic
    import httpx

    exc = anthropic.AuthenticationError(
        message="invalid x-api-key",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://api.anthropic.com")
        ),
        body=None,
    )
    result = _raising_client(controller, exc).run("hi")
    assert "key was rejected" in result.text


def test_network_failure_is_readable(controller):
    import anthropic
    import httpx

    exc = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com")
    )
    result = _raising_client(controller, exc).run("hi")
    assert "network" in result.text.lower()


def test_unexpected_errors_do_not_escape(controller):
    result = _raising_client(controller, RuntimeError("kaboom")).run("hi")
    assert "kaboom" in result.text
    assert result.stopped_early


def test_cost_is_reported(controller):
    agent = agent_with(controller, [_Response([text_block("hi")])])
    result = agent.run("hi")
    # 100 in / 50 out at Opus 5 rates
    assert abs(result.cost_usd - (100 / 1e6 * 5 + 50 / 1e6 * 25)) < 1e-9
