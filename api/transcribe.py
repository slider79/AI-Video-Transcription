"""Vercel serverless function: run the agent for one request.

POST /api/transcribe  with JSON  {"query": "..."}
Returns JSON with the transcript, the source video, and the tools the agent
used, so the frontend can display a result and prove tool use.

Security posture:
  * No key is ever returned to the client. Every error string is passed
    through redact_secrets() as a last line of defence.
  * The query is validated and length-capped before it reaches any API.
  * CORS is locked down: cross-origin browser requests are refused unless the
    origin matches ALLOWED_ORIGIN. (This restricts browsers, not scripts, so it
    is not an abuse control on its own; see the README.)

On Vercel the filesystem is read-only except for /tmp, so the knowledge base
is redirected there before the tools module is imported.
"""

import json
import os
import sys

# Redirect transcript saves to the one writable location on Vercel.
os.environ.setdefault("KNOWLEDGE_BASE_DIR", "/tmp/knowledge_base")

# Make the project root importable (this file lives in api/).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
INDEX_HTML = os.path.join(ROOT_DIR, "index.html")

from http.server import BaseHTTPRequestHandler  # noqa: E402

from tools import redact_secrets  # noqa: E402


REQUIRED_KEYS = ("GROQ_API_KEY", "SERPAPI_API_KEY", "GEMINI_API_KEY")

MAX_QUERY_LEN = 300      # characters; a search query has no reason to be longer
MAX_BODY_BYTES = 4096    # reject oversized request bodies outright


def clean_query(raw: object) -> tuple[str | None, str | None]:
    """Validate and normalise the query. Returns (query, error)."""
    if not isinstance(raw, str):
        return None, "The 'query' field must be a string."
    # Drop control characters (keep normal whitespace), collapse, and trim.
    text = "".join(ch for ch in raw if ch == " " or ch == "\t" or ord(ch) >= 0x20)
    text = " ".join(text.split()).strip()
    if not text:
        return None, "Please provide a non-empty 'query'."
    if len(text) > MAX_QUERY_LEN:
        return None, f"Query is too long (max {MAX_QUERY_LEN} characters)."
    return text, None


def run_agent(query: str) -> dict:
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        # Names only, never values.
        return {"ok": False, "error": f"Server is missing environment variable(s): {', '.join(missing)}."}

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
        "source": (transcription or {}).get("source_url") or (search or {}).get("video_url"),
        "error": None if transcription else "Transcription did not complete. See the answer field for details.",
    }


def _allowed_origin(request_origin: str | None) -> str | None:
    """Echo the request origin only if it matches the configured allowlist.

    ALLOWED_ORIGIN may be a single origin or a comma-separated list. If it is
    unset, no cross-origin header is sent, so same-origin requests (the frontend
    calling its own /api) still work while other sites are refused by the browser.
    """
    configured = os.environ.get("ALLOWED_ORIGIN", "").strip()
    if not configured or not request_origin:
        return None
    allowed = {o.strip() for o in configured.split(",") if o.strip()}
    if "*" in allowed:
        return "*"
    return request_origin if request_origin in allowed else None


class handler(BaseHTTPRequestHandler):
    # ---- response helpers -------------------------------------------------

    def _base_headers(self, content_len: int) -> None:
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(content_len))
        # Harden the response.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        origin = _allowed_origin(self.headers.get("Origin"))
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send(self, status: int, payload: dict) -> None:
        # Redact defensively: nothing leaving this function may carry a key.
        if payload.get("error"):
            payload["error"] = redact_secrets(str(payload["error"]))
        if payload.get("answer"):
            payload["answer"] = redact_secrets(str(payload["answer"]))
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._base_headers(len(body))
        self.end_headers()
        self.wfile.write(body)

    # ---- routes -----------------------------------------------------------

    def do_OPTIONS(self) -> None:
        # CORS preflight.
        self.send_response(204)
        origin = _allowed_origin(self.headers.get("Origin"))
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Vary", "Origin")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self._send(400, {"ok": False, "error": "Invalid Content-Length."})

        if length > MAX_BODY_BYTES:
            return self._send(413, {"ok": False, "error": "Request body too large."})

        try:
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"ok": False, "error": "Invalid JSON body."})

        query, err = clean_query(body.get("query") if isinstance(body, dict) else None)
        if err:
            return self._send(400, {"ok": False, "error": err})

        try:
            result = run_agent(query)
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the client
            return self._send(500, {"ok": False, "error": f"Agent error: {redact_secrets(str(exc))}"})

        self._send(200 if result.get("ok") else 502, result)

    def do_GET(self) -> None:
        # Serve the frontend at the root. In Vercel's single-entrypoint Python
        # mode static files are not auto-served, so the handler serves the page
        # itself. This is harmless if Vercel does serve it statically, since
        # then this path is never reached.
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                with open(INDEX_HTML, "rb") as fh:
                    html = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(html)
                return
            except OSError:
                pass
        self._send(200, {"ok": True, "message": "POST a JSON body with a 'query' to transcribe a video."})

    def log_message(self, *args) -> None:  # noqa: D401 - silence default request logging
        # Default logging prints the request line, which could echo a query.
        return
