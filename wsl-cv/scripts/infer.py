"""SkateP level API: the Unity app (through win-server) sends a photo, gets a level back.

    GET  /health    liveness check
    POST /infer     multipart: photo (image file), style (optional art style)

Response:
    images.preview_base64      PNG: the three layers composited, track drawn on top
    images.foreground_base64   PNG: near layer (drawn in front of the skater)
    images.gameplay_base64     PNG: mid layer
    images.background_base64   PNG: far layer
    metadata                   track_points.json: points, track_color, level
                               (mission, surfaces, occluders), art_style

Each request works in its own temporary folder, so requests never see each
other's files. A requested art style starts in Muse Image as soon as the photo
arrives and runs alongside depth, segmentation and level design; without one,
Muse Image fills the background behind the front layers while the track is
traced.

Run from wsl-cv/:
    uvicorn scripts.infer:app --host 0.0.0.0 --port 9000
"""

import base64
import io
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layers import LAYER_FILES  # noqa: E402
from muse_image import STYLES, fill_enabled, fill_photo_background, front_objects, load_setting, prefetch_style, stylize_run  # noqa: E402
from perception import OUTPUT_WIDTH, fit_width, output_paths  # noqa: E402
from pipeline import build_level  # noqa: E402

DEFAULT_STYLE = os.getenv("MUSE_IMAGE_STYLE", "").strip().lower()
NO_STYLE = {"", "none", "original", "photo"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("skatep")

app = FastAPI(title="SkateP level API", description="Photo in, playable 2.5D skate level out.")


def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def preview_png(layer_paths: list[Path], points: list[list[float]], max_side: int = 768) -> str:
    """The layers composited far to near with the track on top, as base64 PNG ("" on failure)."""
    try:
        far, mid, near = (Image.open(p).convert("RGBA") for p in reversed(layer_paths))
        image = Image.alpha_composite(Image.alpha_composite(far, mid.resize(far.size)), near.resize(far.size))
        width, height = image.size
        line = [(x * (width - 1), y * (height - 1)) for x, y in points]
        if len(line) > 1:
            # The showcase's lime track over a dark outline, readable on any photo.
            draw = ImageDraw.Draw(image)
            draw.line(line, fill=(0, 0, 0, 150), width=max(5, width // 110), joint="curve")
            draw.line(line, fill=(196, 255, 58, 255), width=max(3, width // 200), joint="curve")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        log.warning("Could not compose the preview", exc_info=True)
        return ""


def _restyle(photo: Path, level_dir: Path, objects_path: str, style: str) -> dict[str, Any] | None:
    """The Muse Image style's manifest entry, or None to keep the photo's art."""
    try:
        manifest = stylize_run(
            str(photo),
            str(level_dir),
            [style],
            setting=load_setting(objects_path),
            removed=front_objects(str(level_dir), objects_path),
        )
    except Exception:
        log.exception("Muse Image restyle failed; returning the photo's layers")
        return None
    entry = next((e for e in manifest["styles"] if e["name"] == style), None)
    if entry is None:
        log.error("Muse Image restyle failed (%s); returning the photo's layers", manifest["errors"].get(style))
    return entry


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/infer")
def infer(photo: UploadFile = File(...), style: str | None = Form(None)) -> dict[str, Any]:
    style = (DEFAULT_STYLE if style is None else style).strip().lower()
    style = "" if style in NO_STYLE else style
    if style and style not in STYLES:
        raise HTTPException(400, f"Unknown style '{style}'; choose from {sorted(STYLES)} or 'original'.")
    image = cv2.imdecode(np.frombuffer(photo.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "The upload is not a readable image.")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="skatep-") as work, ThreadPoolExecutor(max_workers=2) as muse_image:
        work = Path(work)
        photo_path, level_dir, inference_dir = work / "photo.jpg", work / "level", work / "inference"
        cv2.imwrite(str(photo_path), image)
        objects_path = output_paths(str(photo_path), str(inference_dir))["objects"]

        # Muse Image needs only the photo: start the restyle now.
        early: Future | None = None
        if style:
            early = muse_image.submit(prefetch_style, style, fit_width(image, OUTPUT_WIDTH), str(level_dir))
        background: list[Future] = []

        def fill_background() -> None:
            if not style and fill_enabled():
                background.append(
                    muse_image.submit(
                        fill_photo_background,
                        str(photo_path),
                        str(level_dir),
                        setting=load_setting(objects_path),
                        removed=front_objects(str(level_dir), objects_path),
                    )
                )

        try:
            metadata = build_level(str(photo_path), str(inference_dir), str(level_dir), after_layers=fill_background)
        except Exception as exc:
            log.exception("Level generation failed")
            raise HTTPException(500, f"Level generation failed: {exc}") from exc

        for future in background:
            try:
                log.info("Background fill: %s", future.result())
            except Exception:
                log.exception("Muse Image background fill failed; keeping the inpainted background")

        layer_paths = [level_dir / name for name in LAYER_FILES]
        if early is not None:
            try:
                early.result()
            except Exception:
                log.warning("Early Muse Image call failed; the restyle retries it", exc_info=True)
            art = _restyle(photo_path, level_dir, objects_path, style)
            if art is not None:
                layer_paths = [level_dir / art["dir"] / name for name in LAYER_FILES]
                metadata["track_color"] = art["track_color"] or metadata["track_color"]
                metadata["art_style"] = {
                    "name": art["name"],
                    "label": art["label"],
                    "model": "muse-image",
                    "edge_f1": art["alignment"]["edge_f1"],
                }

        foreground, gameplay, background_layer = layer_paths
        response = {
            "images": {
                "preview_base64": preview_png(layer_paths, metadata.get("points", [])),
                "foreground_base64": _b64_file(foreground),
                "gameplay_base64": _b64_file(gameplay),
                "background_base64": _b64_file(background_layer),
            },
            "metadata": metadata,
        }
    level = metadata.get("level", {})
    log.info("Level ready in %.1fs (style=%s): %s", time.perf_counter() - started, style or "photo", level.get("mission"))
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
