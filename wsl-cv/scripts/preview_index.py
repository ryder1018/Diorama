"""Build showcase/preview.html: a light page for picking photos by their track and layers.

Reads every showcase/<name>/ run that has a track (a `run_photos.sh --preview`
run or a full build), writes small WebP copies into showcase/<name>/preview/
and one page that references them. Nothing 3D, so dozens of photos stay fast.

Usage:
    python scripts/preview_index.py [showcase_dir] [--force]
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MAX_SIDE = 960
LAYER_NAMES = ["near", "mid", "far"]


def _photo_for(folder: str, name: str) -> str | None:
    for ext in PHOTO_EXTS:
        path = os.path.join(folder, name + ext)
        if os.path.isfile(path):
            return path
    return None


def _fit(image: np.ndarray, interpolation=cv2.INTER_AREA) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(width, height))
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=interpolation)


def _write_webp(path: str, image: np.ndarray, quality: int = 82) -> None:
    cv2.imwrite(path, image, [cv2.IMWRITE_WEBP_QUALITY, quality])


def _tail(path: str, lines: int = 6) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-lines:]).strip()


def _surface_overlay(folder: str, outputs: str, name: str, level: dict, size: tuple[int, int]) -> np.ndarray:
    """RGBA overlay: rideable ground in green, objects the skater passes behind in orange."""
    width, height = size
    overlay = np.zeros((height, width, 4), np.uint8)
    track_path = os.path.join(folder, "layer_00_mask.npy")
    if os.path.isfile(track_path):
        track = cv2.resize((np.load(track_path) > 127).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
        overlay[track] = (90, 220, 120, 70)
    labels_path = os.path.join(outputs, f"{name}_depth_mask.npy")
    objects_path = os.path.join(outputs, f"{name}_sam31_objects.json")
    if level.get("occluders") and os.path.isfile(labels_path) and os.path.isfile(objects_path):
        labels = cv2.resize(np.load(labels_path).astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
        with open(objects_path, encoding="utf-8") as f:
            ids = {o["id"]: o.get("label_id") for o in json.load(f)["objects"]}
        for occ in level["occluders"]:
            if ids.get(occ["id"]) is not None:
                overlay[labels == ids[occ["id"]]] = (40, 150, 255, 80)
    return overlay


def _layer_contents(outputs: str, name: str, folder: str) -> list[list[str]]:
    """Object labels in each layer (near, mid, far), e.g. ['keyboard', 'mug ×2']."""
    layers_path = os.path.join(folder, "layers.json")
    objects_path = os.path.join(outputs, f"{name}_sam31_objects.json")
    if not (os.path.isfile(layers_path) and os.path.isfile(objects_path)):
        return []
    with open(layers_path, encoding="utf-8") as f:
        layers = json.load(f)["layers"]
    with open(objects_path, encoding="utf-8") as f:
        by_label_id = {o["label_id"]: o["label"] for o in json.load(f)["objects"] if o.get("label_id") is not None}
    contents = []
    for layer in layers:
        counts: dict[str, int] = {}
        for region in layer["region_ids"]:
            label = by_label_id.get(region)
            if label:
                counts[label] = counts.get(label, 0) + 1
        contents.append([label if n == 1 else f"{label} ×{n}" for label, n in counts.items()])
    return contents


def _build_assets(folder: str, photo: str, level: dict, name: str, force: bool) -> tuple[int, int] | None:
    out = os.path.join(folder, "preview")
    stamp = os.path.join(out, "track.webp")
    source = os.path.join(folder, "track_points.json")
    image = cv2.imread(photo)
    if image is None:
        return None
    small = _fit(image)
    height, width = small.shape[:2]
    if not force and os.path.isfile(stamp) and os.path.getmtime(stamp) >= os.path.getmtime(source):
        return width, height
    os.makedirs(out, exist_ok=True)
    _write_webp(os.path.join(out, "photo.webp"), small)
    _write_webp(os.path.join(out, "track.webp"), _surface_overlay(folder, os.path.join(folder, "outputs"), name, level, (width, height)), 90)
    for index in range(len(LAYER_NAMES)):
        layer = cv2.imread(os.path.join(folder, f"merged_layer_{index + 1:02d}.png"), cv2.IMREAD_UNCHANGED)
        if layer is not None:
            _write_webp(os.path.join(out, f"layer_{index + 1}.webp"), cv2.resize(layer, (width, height), interpolation=cv2.INTER_AREA))
    return width, height


def collect(showcase_dir: str, force: bool) -> list[dict]:
    entries = []
    for name in sorted(os.listdir(showcase_dir)):
        folder = os.path.join(showcase_dir, name)
        photo = _photo_for(folder, name) if os.path.isdir(folder) else None
        if not photo:
            continue
        entry = {"name": name, "built": os.path.isfile(os.path.join(folder, "showcase.html"))}
        track_path = os.path.join(folder, "track_points.json")
        if not os.path.isfile(track_path):
            log = next((p for p in (os.path.join(folder, f) for f in ("preview.log", "run.log")) if os.path.isfile(p)), None)
            entry.update(status="failed", log=_tail(log) if log else "no track yet")
            entries.append(entry)
            continue
        with open(track_path, encoding="utf-8") as f:
            track = json.load(f)
        level = track.get("level", {})
        size = _build_assets(folder, photo, level, name, force)
        if size is None:
            continue
        validation = level.get("validation") or {}
        entry.update(
            status="ok",
            w=size[0],
            h=size[1],
            points=track.get("points", []),
            occluded=track.get("occluded_segments", []),
            mission=level.get("mission", ""),
            surfaces=level.get("surfaces", []),
            behind=[o["label"] for o in level.get("occluders", [])],
            muse=level.get("generator", "").startswith("muse-spark"),
            attempts=validation.get("attempts"),
            passed=validation.get("passed"),
            notes=level.get("design_notes", ""),
            layers=_layer_contents(os.path.join(folder, "outputs"), name, folder),
        )
        entries.append(entry)
    return entries


def build_preview(showcase_dir: str, force: bool = False) -> str:
    entries = collect(showcase_dir, force)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview_template.html"), encoding="utf-8") as f:
        template = f.read()
    data = json.dumps(entries, ensure_ascii=False).replace("</", "<\\/")
    page = template.replace("__DATA__", data).replace("__STAMP__", str(int(time.time())))
    out = os.path.join(showcase_dir, "preview.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    ok = sum(1 for e in entries if e["status"] == "ok")
    print(f"  ✔ Preview saved to {out} ({ok} track(s), {len(entries) - ok} failed)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("showcase_dir", nargs="?", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "showcase"))
    parser.add_argument("--force", action="store_true", help="Rewrite the WebP copies even if they are current")
    args = parser.parse_args()
    build_preview(args.showcase_dir, args.force)
