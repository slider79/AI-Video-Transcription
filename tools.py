"""The two tools the agent can call.

VideoSearchTool  - finds a YouTube video with SerpApi and returns its URL.
TranscriptionTool - sends that URL to Gemini, which transcribes the video,
                    and saves the transcript to the knowledge base.

Each tool exposes:
  * SCHEMA - the JSON tool definition sent to the Groq agent.
  * run(**args) - the Python implementation the agent's call is routed to.

Keeping the schema next to the implementation means the description the model
sees and the code it triggers can never drift apart.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests

KNOWLEDGE_BASE = Path(__file__).resolve().parent / "knowledge_base"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# Default transcription model. Gemini Flash accepts a YouTube URL directly and
# is fast and inexpensive, which suits transcription. "gemini-flash-latest" is
# an alias that always points at the current Flash model, so it does not get
# retired the way a pinned version (e.g. gemini-2.5-flash) eventually does.
# Override with GEMINI_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# Upper bound on transcript length. 8192 tokens comfortably covers a short
# video; raise it for longer material.
MAX_TRANSCRIPT_TOKENS = 8192


class ToolError(RuntimeError):
    """Raised when a tool cannot complete. The message is shown to the agent."""


# ---------------------------------------------------------------------------
# Tool 1: Video search (SerpApi)
# ---------------------------------------------------------------------------


@dataclass
class VideoResult:
    url: str
    title: str
    channel: str
    length: str


class VideoSearchTool:
    name = "video_search"

    SCHEMA = {
        "type": "function",
        "function": {
            "name": "video_search",
            "description": (
                "Search YouTube for a video matching a query and return its URL "
                "and basic metadata. Call this first, before transcribing, to find "
                "the video the user is asking about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search YouTube for, e.g. 'how photosynthesis works'.",
                    }
                },
                "required": ["query"],
            },
        },
    }

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        self.timeout = timeout

    def run(self, query: str) -> dict:
        if not self.api_key:
            raise ToolError("SERPAPI_API_KEY is not set; cannot search for videos.")

        params = {
            "engine": "youtube",
            "search_query": query,
            "api_key": self.api_key,
        }
        try:
            response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise ToolError(f"SerpApi request failed: {exc}") from exc

        if "error" in data:
            raise ToolError(f"SerpApi returned an error: {data['error']}")

        results = data.get("video_results") or []
        if not results:
            raise ToolError(f"No YouTube videos found for query: {query!r}")

        top = results[0]
        result = VideoResult(
            url=top.get("link", ""),
            title=top.get("title", "Unknown title"),
            channel=(top.get("channel") or {}).get("name", "Unknown channel"),
            length=top.get("length", "unknown"),
        )
        if not result.url:
            raise ToolError("Top search result had no usable video URL.")

        # Returned dict becomes the tool message the agent reads.
        return {
            "video_url": result.url,
            "title": result.title,
            "channel": result.channel,
            "length": result.length,
        }


# ---------------------------------------------------------------------------
# Tool 2: Transcription (Gemini)
# ---------------------------------------------------------------------------


def _slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:limit].strip("-") or "transcript"


class TranscriptionTool:
    name = "transcribe_video"

    SCHEMA = {
        "type": "function",
        "function": {
            "name": "transcribe_video",
            "description": (
                "Transcribe the spoken content of a YouTube video using Gemini's "
                "multimodal model, then save the transcript to the knowledge base. "
                "Call this after video_search, passing the video_url it returned. "
                "This produces the actual transcript; never write a transcript yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {
                        "type": "string",
                        "description": "The YouTube URL returned by video_search.",
                    },
                    "title": {
                        "type": "string",
                        "description": "The video title, used to name the saved file.",
                    },
                },
                "required": ["video_url"],
            },
        },
    }

    PROMPT = (
        "Transcribe the spoken audio of this video verbatim, in full. "
        "Output only the transcript text. Do not summarize, translate, add "
        "commentary, timestamps, or speaker labels unless the speaker states "
        "their own name. If a passage is inaudible, write [inaudible]."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        knowledge_base: Path = KNOWLEDGE_BASE,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.knowledge_base = knowledge_base

    def run(self, video_url: str, title: str = "") -> dict:
        if not self.api_key:
            raise ToolError("GEMINI_API_KEY is not set; cannot transcribe.")

        transcript = self._transcribe(video_url)
        saved_path = self._save(transcript, video_url, title)

        return {
            "transcript": transcript,
            "source_url": video_url,
            "saved_to": str(saved_path),
            "model": self.model,
            "characters": len(transcript),
        }

    # -- internals ---------------------------------------------------------

    def _transcribe(self, video_url: str) -> str:
        # Imported lazily so the module loads (and tests run) without the SDK.
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ToolError(
                "The google-genai package is not installed. Run: pip install google-genai"
            ) from exc

        client = genai.Client(api_key=self.api_key)
        request = dict(
            model=self.model,
            contents=types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=video_url)),
                    types.Part(text=self.PROMPT),
                ]
            ),
            config=types.GenerateContentConfig(max_output_tokens=MAX_TRANSCRIPT_TOKENS),
        )

        response = self._generate_with_retry(client, request)
        if not transcript:
            raise ToolError(
                "Gemini returned an empty transcript. The video may be private, "
                "age-restricted, region-locked, or have no speech."
            )
        return transcript

    # Codes worth retrying: server overload/errors and rate limiting. These are
    # transient; the request is likely to succeed on a second attempt.
    _RETRYABLE_CODES = (500, 502, 503, 504, 429)

    def _generate_with_retry(self, client, request, attempts: int = 4):
        """Call Gemini, retrying transient errors (503 high-demand, 429 rate
        limit, 5xx) with exponential backoff. Non-transient errors, such as a
        missing model or an invalid request, are raised immediately."""
        import time

        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return client.models.generate_content(**request)
            except Exception as exc:  # noqa: BLE001 - inspect, then retry or raise
                code = getattr(exc, "code", None)
                if code not in self._RETRYABLE_CODES or attempt == attempts - 1:
                    raise ToolError(f"Gemini transcription failed: {exc}") from exc
                last_exc = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(
                    f"     Gemini returned {code}; retrying in {wait}s "
                    f"(attempt {attempt + 2} of {attempts})...",
                    flush=True,
                )
                time.sleep(wait)

        # Unreachable, but keeps type checkers happy.
        raise ToolError(f"Gemini transcription failed: {last_exc}")

    def _save(self, transcript: str, video_url: str, title: str) -> Path:
        self.knowledge_base.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}_{_slugify(title or video_url)}.md"
        path = self.knowledge_base / name

        header = (
            f"# Transcript\n\n"
            f"- **Title:** {title or 'Unknown'}\n"
            f"- **Source:** {video_url}\n"
            f"- **Model:** {self.model}\n"
            f"- **Transcribed:** {_dt.datetime.now().isoformat(timespec='seconds')}\n\n"
            f"---\n\n"
        )
        path.write_text(header + transcript + "\n", encoding="utf-8")
        return path
