"""Diorama gateway: the phone talks to this Windows server, which forwards to the WSL level API.

The phone cannot reach WSL directly (it sits behind WSL's NAT), and a level
takes a minute or two, longer than a phone request should stay open. So:

    POST /upload    (X-API-KEY, multipart photo [+ style])  -> {"status": "accepted"}
                    the photo is forwarded to WSL's /infer in the background
    GET  /download  (X-API-KEY) -> {"status": "processing"}
                                or {"status": "success", "images": {...}, "metadata": {...}}
                                or {"status": "error", "message": "..."}
    GET  /health

One level at a time: the app uploads its photos one by one and polls for each.

Settings (environment variables; the older SKATEP_* names still work):
    DIORAMA_API_KEY       key the app sends as X-API-KEY (default 0000)
    DIORAMA_WSL_URL       the WSL level API (default http://127.0.0.1:9000/infer)
    DIORAMA_WSL_TIMEOUT   seconds to wait for one level (default 300)
"""

import os
import threading

import requests
from flask import Flask, jsonify, request


def _setting(name: str, default: str) -> str:
    return os.getenv(f"DIORAMA_{name}") or os.getenv(f"SKATEP_{name}", default)


API_KEY = _setting("API_KEY", "0000")
WSL_URL = _setting("WSL_URL", "http://127.0.0.1:9000/infer")
WSL_TIMEOUT = float(_setting("WSL_TIMEOUT", "300"))

app = Flask(__name__)


class Job:
    """The latest level request and its result."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = {"status": "processing"}

    def set(self, state: dict) -> None:
        with self._lock:
            self._state = state

    def get(self) -> dict:
        with self._lock:
            return self._state


job = Job()


def build_level(photo: bytes, filename: str, content_type: str, style: str | None) -> None:
    print(f"→ WSL {WSL_URL} (style: {style or 'default'})")
    try:
        response = requests.post(
            WSL_URL,
            files={"photo": (filename, photo, content_type)},
            data={"style": style} if style is not None else None,
            timeout=WSL_TIMEOUT,
        )
    except requests.RequestException as exc:
        job.set({"status": "error", "message": f"Level API unreachable: {exc}"})
        print(f"✗ {exc}")
        return
    if response.status_code != 200:
        job.set({"status": "error", "message": f"Level API error {response.status_code}: {response.text[:500]}"})
        print(f"✗ HTTP {response.status_code}")
        return
    try:
        level = response.json()
    except ValueError:
        job.set({"status": "error", "message": "Level API returned an unreadable response"})
        print("✗ response is not JSON")
        return
    job.set({"status": "success", "images": level.get("images", {}), "metadata": level.get("metadata", {})})
    print("✓ level ready")


@app.before_request
def require_key():
    if request.endpoint != "health" and request.headers.get("X-API-KEY") != API_KEY:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    return None


@app.post("/upload")
def upload():
    photo = request.files.get("photo")
    if photo is None:
        return jsonify({"status": "error", "message": "No photo field in request"}), 400
    job.set({"status": "processing"})
    threading.Thread(
        target=build_level,
        args=(photo.read(), photo.filename or "photo.jpg", photo.content_type or "image/jpeg", request.form.get("style")),
        daemon=True,
    ).start()
    return jsonify({"status": "accepted"})


@app.get("/download")
def download():
    return jsonify(job.get())


@app.get("/health")
def health():
    return jsonify({"status": "ok", "level_api": WSL_URL})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
