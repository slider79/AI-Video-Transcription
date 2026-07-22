"""Offline checks for the agent and its tools.

Run with:  python tests/test_agent.py

No API keys are needed. The Groq client and the two network-bound tool calls
(SerpApi and Gemini) are replaced with fakes, so these tests exercise the real
orchestration logic: does the agent call both tools, in the right order, feed
each result to the next step, save the transcript, and cite the source.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import VideoTranscriptionAgent  # noqa: E402
from tools import TranscriptionTool, VideoSearchTool, ToolError  # noqa: E402

failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------------
# A scripted stand-in for the Groq client.
# ---------------------------------------------------------------------------


def tool_call(call_id: str, name: str, **args):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def assistant_turn(content="", tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


class FakeGroq:
    """Replays a fixed list of responses, one per create() call, and records
    the messages it was given so we can assert the tool results flowed back."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.seen_messages = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen_messages.append(kwargs["messages"])
        return self._scripted.pop(0)


# Fakes for the two tools (dependency-injected into the agent).


class FakeSearch:
    name = "video_search"

    def __init__(self):
        self.calls = []

    def run(self, query):
        self.calls.append(query)
        return {
            "video_url": "https://www.youtube.com/watch?v=FAKE123",
            "title": "How Photosynthesis Works",
            "channel": "Science Channel",
            "length": "5:12",
        }


class FakeTranscribe:
    name = "transcribe_video"

    def __init__(self):
        self.calls = []

    def run(self, video_url, title=""):
        self.calls.append(video_url)
        return {
            "transcript": "Plants convert sunlight into energy.",
            "source_url": video_url,
            "saved_to": "knowledge_base/fake.md",
            "model": "gemini-2.5-flash",
            "characters": 36,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_calls_both_tools_in_order():
    print("\nAgent calls video_search then transcribe_video, then cites the source")
    search, transcribe = FakeSearch(), FakeTranscribe()

    groq = FakeGroq(
        [
            assistant_turn(tool_calls=[tool_call("c1", "video_search", query="photosynthesis")]),
            assistant_turn(tool_calls=[tool_call("c2", "transcribe_video",
                                                 video_url="https://www.youtube.com/watch?v=FAKE123",
                                                 title="How Photosynthesis Works")]),
            assistant_turn(content=(
                "Plants convert sunlight into energy.\n\n"
                "Source: https://www.youtube.com/watch?v=FAKE123"
            )),
        ]
    )

    agent = VideoTranscriptionAgent(
        client=groq, search_tool=search, transcription_tool=transcribe, verbose=False
    )
    answer = agent.run("Find and transcribe a video about photosynthesis")

    check(agent.used_tools_in_order() == ["video_search", "transcribe_video"],
          "both tools ran, search before transcribe")
    check(search.calls == ["photosynthesis"], "search received the query")
    check(transcribe.calls == ["https://www.youtube.com/watch?v=FAKE123"],
          "transcribe received the URL from the search result")
    check("Source:" in answer and "FAKE123" in answer, "final answer cites the source URL")


def test_transcribe_receives_search_url():
    print("\nThe transcription step sees the search result in the conversation")
    search, transcribe = FakeSearch(), FakeTranscribe()
    groq = FakeGroq(
        [
            assistant_turn(tool_calls=[tool_call("c1", "video_search", query="x")]),
            assistant_turn(tool_calls=[tool_call("c2", "transcribe_video",
                                                 video_url="https://www.youtube.com/watch?v=FAKE123")]),
            assistant_turn(content="done. Source: https://www.youtube.com/watch?v=FAKE123"),
        ]
    )
    agent = VideoTranscriptionAgent(
        client=groq, search_tool=search, transcription_tool=transcribe, verbose=False
    )
    agent.run("transcribe a video")

    # Before the transcribe step, the tool result from search must be present.
    second_call_messages = groq.seen_messages[1]
    tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
    check(any("FAKE123" in m["content"] for m in tool_msgs),
          "search result is fed back before transcription")


def test_agent_surfaces_tool_failure():
    print("\nA tool failure is reported, not papered over")

    class FailingSearch(FakeSearch):
        def run(self, query):
            raise ToolError("SerpApi request failed: 429 Too Many Requests")

    groq = FakeGroq(
        [
            assistant_turn(tool_calls=[tool_call("c1", "video_search", query="x")]),
            assistant_turn(content="I couldn't search: SerpApi request failed (429)."),
        ]
    )
    agent = VideoTranscriptionAgent(
        client=groq, search_tool=FailingSearch(), transcription_tool=FakeTranscribe(), verbose=False
    )
    answer = agent.run("find a video")

    check(agent.used_tools_in_order() == [], "no tool is recorded as successful")
    tool_msgs = [m for m in groq.seen_messages[1] if m.get("role") == "tool"]
    check(any("error" in m["content"] for m in tool_msgs), "error was returned to the model")
    check("couldn't" in answer.lower() or "error" in answer.lower(), "answer reflects the failure")


def test_search_tool_parses_serpapi(monkeypatch=None):
    print("\nVideoSearchTool parses a SerpApi YouTube response")
    import tools as tools_mod

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "video_results": [
                    {
                        "title": "Real Video",
                        "link": "https://www.youtube.com/watch?v=REAL",
                        "channel": {"name": "Some Channel"},
                        "length": "3:00",
                    }
                ]
            }

    tools_mod.requests.get = lambda *a, **k: FakeResponse()
    tool = VideoSearchTool(api_key="fake")
    result = tool.run("anything")
    check(result["video_url"] == "https://www.youtube.com/watch?v=REAL", "extracts the top video URL")
    check(result["channel"] == "Some Channel", "extracts channel name")


def test_search_tool_needs_key():
    print("\nVideoSearchTool refuses without an API key")
    # Clear any real key that a local .env may have loaded, so this is
    # deterministic on a developer machine and on a clean checkout alike.
    import os

    saved = os.environ.pop("SERPAPI_API_KEY", None)
    try:
        tool = VideoSearchTool(api_key=None)
        tool.run("x")
        check(False, "should have raised ToolError")
    except ToolError:
        check(True, "raises a clear ToolError when the key is missing")
    finally:
        if saved is not None:
            os.environ["SERPAPI_API_KEY"] = saved


def test_transcription_saves_to_knowledge_base():
    print("\nTranscriptionTool saves the transcript with source metadata")
    with tempfile.TemporaryDirectory() as tmp:
        tool = TranscriptionTool(api_key="fake", knowledge_base=Path(tmp))
        # Replace the network call with a canned transcript.
        tool._transcribe = lambda url: "This is the spoken transcript."

        result = tool.run("https://www.youtube.com/watch?v=ABC", title="My Video")
        saved = Path(result["saved_to"])

        check(saved.exists(), "a transcript file was written")
        text = saved.read_text(encoding="utf-8")
        check("This is the spoken transcript." in text, "transcript body was saved")
        check("https://www.youtube.com/watch?v=ABC" in text, "source URL is recorded in the file")
        check(result["source_url"] == "https://www.youtube.com/watch?v=ABC", "returns the source URL")


def test_transcription_retries_transient_errors():
    print("\nTranscriptionTool retries a transient 503, then succeeds")

    class Boom(Exception):
        def __init__(self, code):
            self.code = code
            super().__init__(f"{code} error")

    class FlakyModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls < 3:      # fail twice with a retryable 503
                raise Boom(503)
            return SimpleNamespace(text="Recovered transcript.")

    tool = TranscriptionTool(api_key="fake")
    tool._RETRYABLE_CODES = (503,)  # keep the test focused
    client = SimpleNamespace(models=FlakyModels())

    # Patch sleep so the test does not actually wait.
    import tools as tools_mod
    import time as _time
    original_sleep = _time.sleep
    _time.sleep = lambda *_: None
    try:
        result = tool._generate_with_retry(client, {}, attempts=4)
    finally:
        _time.sleep = original_sleep

    check(client.models.calls == 3, "retried twice, succeeded on the third attempt")
    check(result.text == "Recovered transcript.", "returned the recovered response")


def test_transcription_gives_up_on_permanent_error():
    print("\nTranscriptionTool does not retry a permanent 404")

    class Boom(Exception):
        def __init__(self, code):
            self.code = code
            super().__init__(f"{code} not found")

    class DeadModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise Boom(404)  # not retryable

    tool = TranscriptionTool(api_key="fake")
    client = SimpleNamespace(models=DeadModels())
    try:
        tool._generate_with_retry(client, {}, attempts=4)
        check(False, "should have raised ToolError")
    except ToolError:
        check(True, "raised ToolError without retrying")
    check(client.models.calls == 1, "a 404 is not retried")


def test_redact_secrets_removes_keys():
    print("\nredact_secrets strips key values and key-shaped tokens")
    import os
    from tools import redact_secrets

    os.environ["GROQ_API_KEY"] = "gsk_supersecretvalue123456"
    try:
        msg = "failed for url ...&api_key=abcd1234efgh5678 with token gsk_supersecretvalue123456"
        out = redact_secrets(msg)
        check("gsk_supersecretvalue123456" not in out, "exact key value is removed")
        check("abcd1234efgh5678" not in out, "api_key= query value is removed")
        check("[redacted]" in out, "redaction marker is present")
    finally:
        os.environ.pop("GROQ_API_KEY", None)


def test_serpapi_http_error_hides_key():
    print("\nA SerpApi HTTP error never carries the api_key")
    import tools as tools_mod
    import requests

    class FakeResp:
        status_code = 401
        url = "https://serpapi.com/search.json?search_query=x&api_key=LEAKYKEY999"
        reason = "Unauthorized"

        def raise_for_status(self):
            raise requests.HTTPError(
                f"401 Client Error: Unauthorized for url: {self.url}", response=self
            )

        def json(self):
            return {}

    tools_mod.requests.get = lambda *a, **k: FakeResp()
    tool = VideoSearchTool(api_key="LEAKYKEY999")
    try:
        tool.run("anything")
        check(False, "should have raised ToolError")
    except ToolError as exc:
        check("LEAKYKEY999" not in str(exc), "the key is not in the error message")
        check("401" in str(exc), "the status code is still reported")


def test_query_validation():
    print("\nThe API validates and caps the query")
    import importlib.util

    spec = importlib.util.spec_from_file_location("h", str(ROOT / "api" / "transcribe.py"))
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)

    check(api.clean_query(123)[1] is not None, "non-string is rejected")
    check(api.clean_query("   ")[1] is not None, "blank query is rejected")
    check(api.clean_query("x" * (api.MAX_QUERY_LEN + 1))[1] is not None, "over-long query is rejected")
    q, err = api.clean_query("  black   holes\x00 explained ")
    check(err is None, "a normal query passes")
    check(q == "black holes explained", "control chars stripped and whitespace collapsed")


def test_tool_schemas_are_wellformed():
    print("\nBoth tool schemas are valid function definitions")
    for schema, expected in ((VideoSearchTool.SCHEMA, "video_search"),
                             (TranscriptionTool.SCHEMA, "transcribe_video")):
        fn = schema.get("function", {})
        ok = (
            schema.get("type") == "function"
            and fn.get("name") == expected
            and "description" in fn
            and fn.get("parameters", {}).get("type") == "object"
        )
        check(ok, f"{expected} schema is well-formed")


if __name__ == "__main__":
    for test in (
        test_agent_calls_both_tools_in_order,
        test_transcribe_receives_search_url,
        test_agent_surfaces_tool_failure,
        test_search_tool_parses_serpapi,
        test_search_tool_needs_key,
        test_transcription_saves_to_knowledge_base,
        test_transcription_retries_transient_errors,
        test_transcription_gives_up_on_permanent_error,
        test_redact_secrets_removes_keys,
        test_serpapi_http_error_hides_key,
        test_query_validation,
        test_tool_schemas_are_wellformed,
    ):
        test()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")
