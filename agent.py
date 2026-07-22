"""A tool-calling AI agent that searches for a video and transcribes it.

The agent runs on Groq (an OpenAI-compatible chat model with tool calling). It
is given two tools: video_search (SerpApi) and transcribe_video (Gemini). For a
request like "find and transcribe a video about X", the model:

  1. calls video_search  -> gets a YouTube URL,
  2. calls transcribe_video with that URL -> gets the transcript (saved to disk),
  3. replies with the transcript and the source URL.

The agent never writes a transcript itself. Transcription only ever comes from
the Gemini tool, so a real tool call is always required to produce content.

Run:  python agent.py "your request here"
"""

from __future__ import annotations

import json
import os
import sys

from groq import Groq

try:
    from dotenv import load_dotenv

    # encoding="utf-8-sig" tolerates a byte-order mark, which some Windows
    # editors add to the first line and which would otherwise corrupt the
    # first variable's name (e.g. it becomes "﻿GROQ_API_KEY").
    load_dotenv(encoding="utf-8-sig")
except ImportError:
    pass

from tools import TranscriptionTool, VideoSearchTool, ToolError

# openai/gpt-oss-120b is used for the agent because it emits well-formed tool
# calls reliably on Groq and calls tools sequentially. llama-3.3-70b was tried
# first but intermittently produced a malformed tool-call format that Groq's
# validator rejects, and tended to call both tools at once, which breaks the
# dependency where transcription needs the URL that search returns.
DEFAULT_AGENT_MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 6  # safety bound on the tool-calling loop

SYSTEM_PROMPT = """You are a video research agent with exactly two tools:

1. video_search(query) - finds a YouTube video and returns its URL and metadata.
2. transcribe_video(video_url, title) - transcribes that video with Gemini and
   saves the transcript.

Follow this procedure for any request to find, transcribe, or get the contents
of a video:

- First call video_search to find the video.
- Then call transcribe_video with the exact video_url from the search result.
- Then reply to the user.

Hard rules:
- The transcript MUST come from the transcribe_video tool. Never write, invent,
  guess, paraphrase, or summarize the video's contents yourself. You have not
  watched the video; only the tool has.
- Present the transcript exactly as the tool returned it.
- End every reply with a "Source:" line giving the video URL from video_search.
- If a tool returns an error, tell the user plainly what failed. Do not fill the
  gap with made-up content.
"""


class VideoTranscriptionAgent:
    def __init__(
        self,
        groq_api_key: str | None = None,
        agent_model: str | None = None,
        search_tool: VideoSearchTool | None = None,
        transcription_tool: TranscriptionTool | None = None,
        client=None,
        verbose: bool = True,
    ):
        # `client` is injectable so the tool-calling loop can be tested without
        # a live Groq connection.
        self.client = client or Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))
        self.model = agent_model or os.environ.get("AGENT_MODEL", DEFAULT_AGENT_MODEL)
        self.search_tool = search_tool or VideoSearchTool()
        self.transcription_tool = transcription_tool or TranscriptionTool()
        self.verbose = verbose

        self.tools = [VideoSearchTool.SCHEMA, TranscriptionTool.SCHEMA]
        self._dispatch = {
            self.search_tool.name: self.search_tool.run,
            self.transcription_tool.name: self.transcription_tool.run,
        }
        # Records which tools ran, so callers can verify tools were actually used.
        self.tool_calls: list[dict] = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _run_tool(self, name: str, args: dict) -> dict:
        """Execute one tool call, converting tool failures into a payload the
        model can read rather than an exception that aborts the run."""
        func = self._dispatch.get(name)
        if func is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = func(**args)
            self.tool_calls.append({"tool": name, "args": args, "ok": True, "result": result})
            return result
        except ToolError as exc:
            self.tool_calls.append({"tool": name, "args": args, "ok": False, "error": str(exc)})
            return {"error": str(exc)}

    def last_result(self, tool_name: str) -> dict | None:
        """The most recent successful result for a tool, or None. Lets a caller
        pull structured data (URL, title, transcript) rather than parse text."""
        for call in reversed(self.tool_calls):
            if call["tool"] == tool_name and call.get("ok"):
                return call.get("result")
        return None

    def run(self, user_request: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]

        for _ in range(MAX_STEPS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                # One tool at a time, so transcribe_video always runs after
                # video_search and receives the real URL rather than a guess.
                parallel_tool_calls=False,
                temperature=0.2,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            # Record the assistant turn that requested the tools.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for call in message.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                self._log(f"  -> {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
                result = self._run_tool(name, args)

                if "error" in result:
                    self._log(f"     ! {result['error']}")
                elif name == self.transcription_tool.name:
                    self._log(f"     saved {result.get('characters', 0)} chars to {result.get('saved_to')}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )

        return (
            "Stopped after the maximum number of steps without a final answer. "
            "This usually means a tool kept failing; check the log above."
        )

    def used_tools_in_order(self) -> list[str]:
        """The sequence of successful tool names, for verification."""
        return [c["tool"] for c in self.tool_calls if c.get("ok")]


def main(argv: list[str]) -> int:
    request = " ".join(argv).strip() or _prompt_for_request()
    if not request:
        print("Nothing to do. Give a request, e.g.:")
        print('  python agent.py "find and transcribe a short video about black holes"')
        return 1

    for var in ("GROQ_API_KEY", "SERPAPI_API_KEY", "GEMINI_API_KEY"):
        if not os.environ.get(var):
            print(f"Warning: {var} is not set. The run will fail when that service is called.")

    agent = VideoTranscriptionAgent()
    print(f"\nRequest: {request}\n")
    answer = agent.run(request)

    print("\n" + "=" * 70)
    print(answer)
    print("=" * 70)
    print(f"\nTools used (in order): {' -> '.join(agent.used_tools_in_order()) or 'none'}")
    return 0


def _prompt_for_request() -> str:
    try:
        return input("What video should I find and transcribe? ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
