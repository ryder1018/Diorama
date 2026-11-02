"""Step 4: trace the riding line along the top edge of the ground.

The raw edge is noisy (grass blades, paving joints, SAM's mask border), so it
is smoothed, pulled a little toward the ground's overall slope, reduced to its
turning points and re-sampled as a cubic spline. The result keeps real slopes
and steps but not the jitter. Writes track_points.json:

  {"version", "timestamp", "source_image", "aspect_ratio", "track_color",
   "points": [[x, y], ...]}            x, y in 0..1 of the photo, left to right

level_designer.finalize_level() later adds the level metadata to it.
"""

import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np
import scipy.ndimage
from scipy.interpolate import CubicSpline

DEFAULT_COLOR = "#9e5752"


def top_edge(ground: np.ndarray) -> np.ndarray:
    """Topmost ground row per column; empty columns take the line between their neighbours."""
    height, width = ground.shape
    has = ground.any(axis=0)
    if not has.any():
        return np.full(width, height - 1, dtype=np.float64)
    first = np.argmax(ground, axis=0).astype(np.float64)
    columns = np.arange(width)
    return np.interp(columns, columns[has], first[has])


def smooth_line(
    edge: np.ndarray,
    height: int,
    sigma: float = 18.0,
    slope_weight: float = 0.25,
    simplify_px: float = 18.0,
    spacing_px: float = 18.0,
    top_margin_px: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel x, y of the smoothed line, one point every `spacing_px`."""
    width = edge.size
    columns = np.arange(width, dtype=np.float64)
    slope = np.polyval(np.polyfit(columns, edge, 1), columns)
    line = (1 - slope_weight) * scipy.ndimage.gaussian_filter1d(edge, sigma) + slope_weight * slope

    # Keep only the turning points, then bend a spline through them.
    polyline = np.stack([columns, line], axis=1).astype(np.int32).reshape(-1, 1, 2)
    keys = cv2.approxPolyDP(polyline, simplify_px, False).reshape(-1, 2).astype(np.float64)
    top = min(top_margin_px, height - 1)
    keys[:, 0] = np.clip(keys[:, 0], 0, width - 1)
    keys[:, 1] = np.clip(keys[:, 1], top, height - 1)
    # The ends of an open polyline are always kept, so the line spans the full width.
    xs, first = np.unique(keys[:, 0], return_index=True)
    ys = keys[first, 1]

    samples = np.linspace(xs[0], xs[-1], max(10, int((xs[-1] - xs[0]) / spacing_px)))
    curve = np.clip(CubicSpline(xs, ys, bc_type="natural")(samples), top, height - 1)
    # The spline can overshoot at a clipped key; a light blur hides the corner.
    curve = np.clip(scipy.ndimage.gaussian_filter1d(curve, 1.5, mode="nearest"), top, height - 1)
    return samples, curve


def track_color(image_bgr: np.ndarray, points: list[list[float]]) -> str | None:
    """Average colour of the photo along the track (5x5 px around each point), as #rrggbb."""
    height, width = image_bgr.shape[:2]
    patches = []
    for x, y in points:
        px = int(np.clip(x, 0, 1) * (width - 1))
        py = int(np.clip(y, 0, 1) * (height - 1))
        patch = image_bgr[max(0, py - 2) : py + 3, max(0, px - 2) : px + 3]
        if patch.size:
            patches.append(patch.reshape(-1, 3).mean(axis=0))
    if not patches:
        return None
    b, g, r = (int(c) for c in np.mean(patches, axis=0))
    return f"#{r:02x}{g:02x}{b:02x}"


def trace_track(ground_mask_path: str, out_json: str, image_path: str, top_margin_px: float = 80.0) -> dict:
    """Trace the line along the ground mask and write track_points.json; returns its content."""
    ground = np.load(ground_mask_path) > 127
    height, width = ground.shape
    xs, ys = smooth_line(top_edge(ground), height, top_margin_px=top_margin_px)
    points = [[round(x / (width - 1), 4), round(y / (height - 1), 4)] for x, y in zip(xs, ys)]

    photo = cv2.imread(image_path)
    track = {
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_image": os.path.basename(image_path),
        "aspect_ratio": round(width / height, 4),
        "track_color": (track_color(photo, points) if photo is not None else None) or DEFAULT_COLOR,
        "points": points,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(track, f, indent=4)
    print(f"  ✔ Track traced: {len(points)} points")
    return track
