"""Vercel serverless function: run the agent for one request.

POST /api/transcribe  with JSON  {"query": "..."}
Returns JSON with the transcript, the source video, and the tools the agent
used, so the frontend can display a result and prove tool use.

On Vercel the filesystem is read-only except for /tmp, so the knowledge base
is redirected there before the tools module is imported.
"""

import json
import os
import sys

# Redirect transcript saves to the one writable location on Vercel.
os.environ.setdefault("KNOWLEDGE_BASE_DIR", "/tmp/knowledge_base")

# Make the project root importable (this file lives in api/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import BaseHTTPRequestHandler  # noqa: E402


REQUIRED_KEYS = ("GROQ_API_KEY", "SERPAPI_API_KEY", "GEMINI_API_KEY")


def run_agent(query: str) -> dict:
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        return {
            "ok": False,
            "error": f"Server is missing environment variable(s): {', '.join(missing)}.",
        }

    # Imported here so a missing dependency surfaces as a clean JSON error.
    from agent import VideoTranscriptionAgent

    agent = VideoTranscriptionAgent(verbose=False)
    answer = agent.run(query)

    search = agent.last_result("video_search")
    transcription = agent.last_result("transcribe_video")

    return {
        "ok": transcription is not None,
        "query": query,
        "answer": answer,
        "tools_used": agent.used_tools_in_order(),
        "video": search,
        "transcript": (transcription or {}).get("transcript"),
        "source": (transcription or {}).get("source_url")
        or (search or {}).get("video_url"),
        "error": None if transcription else "Transcription did not complete. See the answer field for details.",
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            query = (json.loads(raw or b"{}").get("query") or "").strip()
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"ok": False, "error": "Invalid JSON body."})

        if not query:
            return self._send(400, {"ok": False, "error": "Please provide a 'query'."})

        try:
            result = run_agent(query)
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the client
            return self._send(500, {"ok": False, "error": f"Agent error: {exc}"})

        self._send(200 if result.get("ok") else 502, result)

    def do_GET(self) -> None:
        # A simple health check.
        self._send(200, {"ok": True, "message": "POST a JSON body with a 'query' to transcribe a video."})
