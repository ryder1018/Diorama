"""Muse Spark depth: how far from the camera each SAM 3.1 region is.

The layers need each region's distance, not a depth value per pixel, so the
depth comes from Muse Spark instead of a depth network:

  1. Muse Spark sees the photo with every region outlined and numbered, and a
     table of the regions (where they are, how big, where they meet the ground),
     and estimates each region's distance in meters.
  2. The estimates are checked against the ground plane: of two objects
     standing on the riding surface, the one whose base is clearly lower in the
     photo is nearer. Violations go back to Muse Spark for repair, up to N
     attempts.
  3. layers.py splits the regions into near / mid / far thirds by distance. For
     the showcase's 3D relief, a depth image is drawn from the estimates: every
     object at its distance, the ground sloping away from the camera.

Writes <base>_depth_order.json and <base>_depth_16bit.png (larger = farther).
"""

import json
import os
import time
from typing import Any

import cv2
import numpy as np

from config import env_number
from muse_spark import call_muse_spark, image_data_url, model_id

GRID = 1000

INSTRUCTIONS = """You estimate how far each region of a photo is from the camera, in meters.

The photo comes with a copy where every segmented region is outlined and labelled with its
number. Region 0 is everything that was not segmented (often sky, walls or distant scenery).
Give every region number exactly once with its distance from the camera in meters.

Judge each region by where most of it is. A region that stretches from near to far, such as
the ground or a long wall, goes by its middle. Objects at the same distance (two things side
by side) get the same value. Use every cue in the photo: what covers what, where objects
stand on the ground (on flat ground, a base lower in the photo is nearer), real-world sizes,
perspective and haze. The table gives each region's box and where it touches the riding
surface, on a 0..1000 grid with y growing downward."""

DEPTH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["regions", "notes"],
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["region", "meters"],
                "properties": {"region": {"type": "integer"}, "meters": {"type": "number"}},
            },
        },
        "notes": {"type": "string"},
    },
}


def _ground_contacts(labels: np.ndarray, ground: np.ndarray) -> dict[int, float]:
    """Row (pixels) where each region's base meets the ground, for regions standing on it."""
    height, width = labels.shape
    reach = max(3, int(height * 0.02))
    # Ground pixels just below each pixel (within `reach` rows).
    below = np.zeros_like(ground)
    for shift in range(1, reach + 1):
        below[:-shift] |= ground[shift:]
    contacts: dict[int, float] = {}
    for region in np.unique(labels):
        if region <= 0:
            continue
        mask = labels == region
        if (mask & ground).sum() > 0.5 * mask.sum():
            continue  # part of the ground itself
        has = mask.any(axis=0)
        bottom = height - 1 - np.argmax(mask[::-1], axis=0)
        cols = np.flatnonzero(has)
        touching = cols[below[bottom[cols], cols]]
        if touching.size >= max(2, 0.2 * cols.size):
            contacts[int(region)] = float(np.median(bottom[touching]))
    return contacts


def _region_table(labels: np.ndarray, objects: list[dict[str, Any]], ground_ids: set[int], contacts: dict[int, float]) -> list[dict[str, Any]]:
    height, width = labels.shape
    names = {o["label_id"]: o["id"] for o in objects if o.get("label_id") is not None}
    table = []
    for region in np.unique(labels):
        if region < 0:
            continue
        ys, xs = np.nonzero(labels == region)
        row: dict[str, Any] = {
            "region": int(region),
            "name": "not segmented" if region == 0 else names.get(int(region), f"region {region}"),
            "box": [round(xs.min() / width * GRID), round(ys.min() / height * GRID), round(xs.max() / width * GRID), round(ys.max() / height * GRID)],
            "area_percent": round(100 * ys.size / labels.size, 1),
        }
        if region in ground_ids:
            row["riding_surface"] = True
        if region in contacts:
            row["base_on_surface_y"] = round(contacts[region] / height * GRID)
        table.append(row)
    return table


def _numbered_view(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The photo with each region outlined and its number written inside it."""
    view = (image * 0.8).astype(np.uint8)
    edges = cv2.morphologyEx(labels.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    view[edges] = (255, 255, 255)
    scale = max(0.5, image.shape[1] / 1400)
    for region in np.unique(labels):
        if region <= 0:
            continue
        mask = (labels == region).astype(np.uint8)
        inside = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(inside), inside.shape)
        text = str(int(region))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x0, y0 = int(np.clip(x - tw // 2, 0, image.shape[1] - tw)), int(np.clip(y + th // 2, th, image.shape[0] - 1))
        cv2.rectangle(view, (x0 - 3, y0 - th - 3), (x0 + tw + 3, y0 + 4), (0, 0, 0), -1)
        cv2.putText(view, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, (60, 255, 200), 2, cv2.LINE_AA)
    return view


def check_distances(answer: list[dict[str, Any]], regions: list[int], contacts: dict[int, float], height: int) -> list[str]:
    """Problems with the estimates: missing, unknown or repeated regions, and ground-plane contradictions."""
    listed = [item["region"] for item in answer]
    errors = []
    missing = [r for r in regions if r not in listed]
    unknown = sorted({r for r in listed if r not in regions})
    repeated = sorted({r for r in listed if listed.count(r) > 1})
    if missing:
        errors.append(f"Regions {missing} are missing; give every region exactly once.")
    if unknown:
        errors.append(f"Regions {unknown} do not exist.")
    if repeated:
        errors.append(f"Regions {repeated} are listed more than once.")
    if any(item["meters"] <= 0 for item in answer):
        errors.append("Every distance must be a positive number of meters.")
    if errors:
        return errors
    meters = {item["region"]: item["meters"] for item in answer}
    gap = 0.1 * height
    standing = sorted(contacts, key=lambda r: -contacts[r])
    for i, near in enumerate(standing):
        for far in standing[i + 1 :]:
            if contacts[near] - contacts[far] > gap and meters[near] >= meters[far]:
                errors.append(
                    f"Region {near} stands on the riding surface at y={round(contacts[near] / height * GRID)}, clearly lower "
                    f"in the photo than region {far} (y={round(contacts[far] / height * GRID)}), so it is nearer, "
                    f"but it is {meters[near]} m away against {meters[far]} m."
                )
    return errors[:6]


def geometric_distances(regions: list[int], labels: np.ndarray, contacts: dict[int, float]) -> dict[int, float]:
    """Fallback: the lower a region reaches in the photo, the nearer (ground contact first, else its bottom row)."""
    height = labels.shape[0]
    distances = {}
    for region in regions:
        if region == 0:
            distances[region] = 100.0
            continue
        rows = np.nonzero((labels == region).any(axis=1))[0]
        bottom = contacts.get(region, float(rows.max()) if rows.size else 0.0)
        distances[region] = 1.0 + 20.0 * (1.0 - bottom / (height - 1))
    return distances


def depth_image(labels: np.ndarray, distances: dict[int, float], ground_ids: set[int], contacts: dict[int, float]) -> np.ndarray:
    """A depth map (log meters, larger = farther) drawn from the estimates, for the 3D relief."""
    height, width = labels.shape
    depth = np.zeros((height, width), np.float32)
    for region, meters in distances.items():
        depth[labels == region] = np.log1p(meters)
    # The ground slopes away: nearest at the photo's bottom edge, as far as the
    # farthest object standing on it at its top edge.
    near = np.log1p(min(distances.values()))
    far = np.log1p(max([distances[r] for r in contacts if r in distances] + [distances.get(r, 1.0) for r in ground_ids]))
    rows = np.arange(height, dtype=np.float32)[:, None]
    for region in ground_ids:
        mask = labels == region
        if not mask.any():
            continue
        top = float(np.nonzero(mask.any(axis=1))[0].min())
        t = np.clip((height - 1 - rows) / max(height - 1 - top, 1.0), 0, 1)
        slope = np.broadcast_to(near + t * (far - near), (height, width))
        depth[mask] = slope[mask]
    return depth


def estimate_distances(
    image_path: str,
    labels_path: str,
    objects_path: str,
    ground_ids: list[int] | None,
) -> dict[str, Any]:
    """Muse Spark's distance for every region; writes it and the depth image next to the label map."""
    image = cv2.imread(image_path)
    labels = np.load(labels_path).astype(np.int32)
    with open(objects_path, encoding="utf-8") as f:
        objects = json.load(f)["objects"]
    height = labels.shape[0]
    ground = set(ground_ids or [])
    contacts = _ground_contacts(labels, np.isin(labels, list(ground)))
    regions = [int(r) for r in np.unique(labels) if r >= 0]
    table = _region_table(labels, objects, ground, contacts)

    conversation: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Estimate each region's distance from the camera:\n" + json.dumps(table)},
                {"type": "input_image", "image_url": image_data_url(image)},
                {"type": "input_image", "image_url": image_data_url(_numbered_view(image, labels))},
            ],
        }
    ]
    attempts: list[dict[str, Any]] = []
    generator = model_id()
    try:
        for attempt in range(1, max(1, env_number("MUSE_SPARK_MAX_ATTEMPTS", 3)) + 1):
            started = time.perf_counter()
            raw, usage = call_muse_spark(INSTRUCTIONS, conversation, "region_depth", DEPTH_SCHEMA, os.getenv("MUSE_SPARK_REASONING_EFFORT", "medium"))
            answer = json.loads(raw)
            errors = check_distances(answer["regions"], regions, contacts, height)
            attempts.append({"attempt": attempt, "regions": answer["regions"], "notes": answer["notes"], "errors": errors, "usage": usage, "seconds": round(time.perf_counter() - started, 2)})
            print(f"  ✔ Muse Spark depth, attempt {attempt}: {len(errors)} violation(s)")
            for error in errors:
                print(f"    - {error}")
            if not errors:
                break
            conversation.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": raw}]})
            conversation.append(
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "The estimates failed these checks:\n- " + "\n- ".join(errors) + "\nReturn corrected estimates."}]}
            )
    except Exception as exc:
        print(f"  ! Muse Spark depth failed ({exc}); estimating from height in the photo.")

    # Keep the attempt with the fewest ground-plane violations among the complete ones.
    complete = [a for a in attempts if not check_distances(a["regions"], regions, {}, height)]
    if complete:
        best = min(complete, key=lambda a: len(a["errors"]))
        distances = {item["region"]: float(item["meters"]) for item in best["regions"]}
    else:
        distances, generator = geometric_distances(regions, labels, contacts), "geometric"

    base = labels_path[: -len("_depth_mask.npy")]
    depth = depth_image(labels, distances, ground, contacts)
    cv2.imwrite(f"{base}_depth_16bit.png", (depth / max(float(depth.max()), 1e-6) * 65535).astype(np.uint16))
    result = {
        "generator": generator,
        "seconds": round(sum(a["seconds"] for a in attempts), 2),
        "meters": {str(r): m for r, m in sorted(distances.items())},
        "contacts": {str(r): round(y / height * GRID) for r, y in contacts.items()},
        "attempts": attempts,
    }
    with open(f"{base}_depth_order.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
