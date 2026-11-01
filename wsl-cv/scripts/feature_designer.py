"""Muse Spark trick designer: jumps and grinds along a finished track.

Showcase only: it reads a finished run (track_points.json, the SAM 3.1 label
map) and writes features.json next to it, which showcase.py turns into jump
arcs and grind rails. track_points.json and the Unity payload are unchanged.

  1. Muse Spark sees the photo with the track drawn on it, plus every SAM 3.1
     object that meets the track (where, how tall, in front of the skater or
     not), and proposes up to four tricks, or none when nothing on the line
     suits one.
  2. Every trick is checked against the masks: a jump must clear the whole
     object it jumps (the arc height is computed from the mask), a grind must
     follow the object's real top edge. Violations go back to Muse Spark,
     up to N attempts.
  3. Tricks that still fail after the last attempt are dropped; the rest are
     saved.

Usage:
    python scripts/feature_designer.py <image> <inference_dir> <send2unity_dir>
"""

import argparse
import json
import os
import time
from typing import Any

import cv2
import numpy as np

from level_designer import _get_float_env, _get_int_env, _top_edge
from muse_spark import call_muse_spark, image_data_url, model_id

GRID = 1000.0
# Same scene units as the showcase: the photo is H tall.
SCENE_H = 6.0
MAX_TRICKS = 4

INSTRUCTIONS = """You add tricks to a 2.5D side-scrolling skateboarding level made from a photo.
The skater rolls from left to right along the bright green line drawn on the photo (the track).
You get the objects that meet the track. Coordinates use a 0..1000 grid (x=0 left).

Two kinds of trick:
- "jump": an ollie over an object that stands on the track ("stands_on_track": true; a mug on
  a desk, a box on a floor, a hydrant on a sidewalk). Objects that only overlap the line in the
  photo but stand in front of or behind it cannot be jumped. from_x is the take-off, to_x the
  landing; the span must cover the
  whole object ("track_x") with run-up room on both sides, and be 40–300 long. Tall objects
  need wider spans, and some are simply too tall to ollie.
- "grind": a slide along a ledge, curb, rail or bench whose top edge runs along the track
  ("edge_along_track_x" says where). from_x..to_x is the stretch of that edge you ride, 60–600 long.

Rules:
- Only use objects from the list. Objects marked "in_front" are between the camera and the
  track: the skater passes behind them, so they cannot be jumped or ground.
- Tricks must not overlap; leave room to land between them. At most four.
- Pick what a skater would actually go for. Big, far-away or silly objects (a whole car, a
  person, a tree) are not tricks. Zero tricks is a good answer when nothing on the line suits one.

"name" is a short trick name for the player (e.g. "Ollie the coffee mug", "Curb grind").
"mission" rewrites the current mission in one short, fun sentence that also names the tricks;
leave it empty when there are no tricks. "design_notes" briefly explains the choice."""

TRICKS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tricks", "mission", "design_notes"],
    "properties": {
        "tricks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "object", "from_x", "to_x", "name"],
                "properties": {
                    "type": {"type": "string", "enum": ["jump", "grind"]},
                    "object": {"type": "string"},
                    "from_x": {"type": "number"},
                    "to_x": {"type": "number"},
                    "name": {"type": "string"},
                },
            },
        },
        "mission": {"type": "string"},
        "design_notes": {"type": "string"},
    },
}


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end] index runs where flags is True."""
    runs: list[tuple[int, int]] = []
    for x in np.flatnonzero(flags):
        if runs and x == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], int(x))
        else:
            runs.append((int(x), int(x)))
    return runs


class TrackScene:
    """The finished level measured against the SAM 3.1 masks."""

    def __init__(self, image_path: str, inference_dir: str, send2unity_dir: str):
        base = os.path.splitext(os.path.basename(image_path))[0]
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        self.height, self.width = self.image.shape[:2]
        labels = np.load(os.path.join(inference_dir, f"{base}_depth_mask.npy"))
        if labels.shape != (self.height, self.width):
            labels = cv2.resize(labels.astype(np.int32), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        self.labels = labels
        with open(os.path.join(inference_dir, f"{base}_sam31_objects.json"), encoding="utf-8") as f:
            self.objects = {o["id"]: o for o in json.load(f)["objects"] if o.get("label_id") is not None}
        with open(os.path.join(send2unity_dir, "track_points.json"), encoding="utf-8") as f:
            self.track = json.load(f)
        level = self.track.get("level") or {}
        self.surfaces = set(level.get("surfaces", []))
        self.occluders = {o["id"] for o in level.get("occluders", [])}
        self.mission = level.get("mission", "")

        points = self.track["points"]
        xs = np.array([p[0] for p in points]) * (self.width - 1)
        ys = np.array([p[1] for p in points]) * (self.height - 1)
        self.track_y = np.interp(np.arange(self.width), xs, ys, left=np.nan, right=np.nan)
        self.x0, self.x1 = int(np.ceil(xs.min())), int(np.floor(xs.max()))

        self.corridor = int(self.height * _get_float_env("LEVEL_SKATER_HEIGHT", 0.15))
        self.edge_tol = self.height * _get_float_env("TRICK_EDGE_TOLERANCE", 0.035)
        self.max_air = _get_float_env("TRICK_MAX_AIR", 0.4)  # of the photo height
        self._geometry: dict[str, dict[str, Any]] = {}

    def gx(self, px: float) -> int:
        return round(px / (self.width - 1) * GRID)

    def px(self, gx: float) -> int:
        return int(round(np.clip(gx, 0, GRID) / GRID * (self.width - 1)))

    def geometry(self, oid: str) -> dict[str, Any]:
        """Where an object meets the track, how tall it stands there, where its edge follows it."""
        if oid in self._geometry:
            return self._geometry[oid]
        mask = self.labels == self.objects[oid]["label_id"]
        top = _top_edge(mask)
        has = mask.any(axis=0)
        bottom = np.where(has, self.height - 1 - np.argmax(mask[::-1], axis=0), np.nan).astype(np.float32)
        rows = np.arange(self.height)[:, None]
        ty = np.where(np.isfinite(self.track_y), self.track_y, -1e9)[None, :]
        # Columns where the object reaches into the skater's corridor just above the track.
        band = mask & (rows >= ty - self.corridor) & (rows <= ty + 0.02 * self.height)
        contact = band.any(axis=0) & np.isfinite(self.track_y)
        valid = np.isfinite(top) & np.isfinite(self.track_y)
        along = valid & (np.abs(top - np.nan_to_num(self.track_y)) <= self.edge_tol)
        # Stands on the track: its base sits on the line (not on the floor in front of it,
        # where its top can still poke up to the line in the photo).
        cols = np.flatnonzero(contact)
        base = float(np.median(bottom[cols] - self.track_y[cols])) if cols.size else np.inf
        stands = abs(base) <= self.height * _get_float_env("TRICK_BASE_TOLERANCE", 0.06)
        geo = {"mask_top": top, "contact": contact, "along": along, "stands": stands, "base": base}
        self._geometry[oid] = geo
        return geo

    def describe(self) -> list[dict[str, Any]]:
        """The objects that meet the track, in grid units for Muse Spark."""
        described = []
        for oid, obj in self.objects.items():
            if oid in self.surfaces:
                continue
            geo = self.geometry(oid)
            cols = np.flatnonzero(geo["contact"])
            if cols.size < 0.01 * self.width:
                continue
            rise = (self.track_y[cols] - geo["mask_top"][cols]) / (self.height - 1) * GRID
            edge = [r for r in _runs(geo["along"]) if r[1] - r[0] >= 0.04 * self.width]
            entry = {
                "id": oid,
                "label": obj["label"],
                "skateable": obj["skateable"],
                "in_front": oid in self.occluders,
                "stands_on_track": geo["stands"],
                "track_x": [self.gx(cols[0]), self.gx(cols[-1])],
                "height_above_track": round(float(np.nanmax(rise))),
            }
            if edge:
                longest = max(edge, key=lambda r: r[1] - r[0])
                entry["edge_along_track_x"] = [self.gx(longest[0]), self.gx(longest[1])]
            described.append(entry)
        return sorted(described, key=lambda o: o["track_x"][0])

    def jump_height(self, oid: str, a: int, b: int) -> float:
        """Apex (fraction of photo height) the arc needs to clear the object between a and b."""
        geo = self.geometry(oid)
        cols = np.flatnonzero(geo["contact"][a : b + 1]) + a
        clearance = 0.02 * self.height
        need = 0.0
        for x in cols:
            s = (x - a) / max(b - a, 1)
            rise = self.track_y[x] - geo["mask_top"][x] + clearance
            need = max(need, rise / max(4 * s * (1 - s), 1e-3))
        return float(need / (self.height - 1))

    def narrowest_jump(self, oid: str) -> tuple[int, int] | None:
        """The narrowest span (pixels), centered on the object, that clears it within the limits."""
        cols = np.flatnonzero(self.geometry(oid)["contact"])
        c0, c1 = int(cols[0]), int(cols[-1])
        center = (c0 + c1) / 2
        step = max(1, self.width // 200)
        for half in range((c1 - c0) // 2 + step, int(0.15 * self.width) + 1, step):
            a, b = int(center - half), int(center + half)
            if a < self.x0 or b > self.x1:
                break
            if a < c0 and b > c1 and self.jump_height(oid, a, b) <= self.max_air:
                return a, b
        return None

    def check_trick(self, trick: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
        """Violations for one trick, plus the resolved feature when there are none."""
        oid, kind = trick["object"], trick["type"]
        a, b = sorted((self.px(trick["from_x"]), self.px(trick["to_x"])))
        name = f"{kind} on '{oid}' (x={self.gx(a)}–{self.gx(b)})"
        if oid not in self.objects:
            return [f"{name}: unknown object id; use ids from the object list."], None
        if oid in self.surfaces:
            return [f"{name}: '{oid}' is the riding surface itself; tricks need an obstacle on it."], None
        if oid in self.occluders:
            return [f"{name}: '{oid}' is in front of the track (the skater passes behind it), so it cannot be ridden."], None
        if a < self.x0 or b > self.x1:
            return [f"{name}: the track only runs from x={self.gx(self.x0)} to x={self.gx(self.x1)}."], None

        geo = self.geometry(oid)
        span = (b - a) / self.width
        errors = []
        height = None
        if kind == "jump":
            cols = np.flatnonzero(geo["contact"])
            if cols.size == 0:
                return [f"{name}: '{oid}' does not meet the track; there is nothing to jump there."], None
            c0, c1 = int(cols[0]), int(cols[-1])
            if not geo["stands"]:
                where = "in front of" if geo["base"] > 0 else "behind"
                return [f"{name}: '{oid}' does not stand on the track; its base is {where} the line, so the skater never meets it."], None
            if not 0.04 <= span <= 0.3:
                errors.append(f"{name}: a jump spans 40–300 on the grid; this one spans {self.gx(b) - self.gx(a)}.")
            if c0 <= a or c1 >= b:
                errors.append(
                    f"{name}: '{oid}' meets the track at x={self.gx(c0)}–{self.gx(c1)}; take off before "
                    f"{self.gx(c0)} and land after {self.gx(c1)}, with run-up room on both sides."
                )
            elif not errors:
                height = self.jump_height(oid, a, b)
                if height > self.max_air:
                    fit = self.narrowest_jump(oid)
                    hint = (
                        f"a span of at least x={self.gx(fit[0])}–{self.gx(fit[1])} clears it"
                        if fit
                        else "no jump within 300 clears it, so drop it or pick a lower object"
                    )
                    errors.append(
                        f"{name}: clearing '{oid}' over this span needs an apex of {height:.0%} of the photo "
                        f"height (max {self.max_air:.0%}); {hint}."
                    )
        else:
            if not self.objects[oid]["skateable"]:
                errors.append(f"{name}: '{oid}' is scenery ({self.objects[oid]['label']}), not a ledge, curb or rail.")
            if not 0.06 <= span <= 0.6:
                errors.append(f"{name}: a grind spans 60–600 on the grid; this one spans {self.gx(b) - self.gx(a)}.")
            follows = float(geo["along"][a : b + 1].mean())
            if follows < 0.6:
                runs = [r for r in _runs(geo["along"]) if r[1] - r[0] >= 0.04 * self.width]
                hint = (
                    "its top edge meets the track at "
                    + ", ".join(f"x={self.gx(r[0])}–{self.gx(r[1])}" for r in runs)
                    if runs
                    else "its top edge never runs along the track"
                )
                errors.append(f"{name}: only {follows:.0%} of the span follows the top edge of '{oid}'; {hint}.")
        if errors:
            return errors, None

        feature = {
            "type": kind,
            "object": oid,
            "label": self.objects[oid]["label"],
            "name": trick["name"].strip() or f"{kind} {self.objects[oid]['label']}",
            "from": [round(a / (self.width - 1), 4), round(float(self.track_y[a]) / (self.height - 1), 4)],
            "to": [round(b / (self.width - 1), 4), round(float(self.track_y[b]) / (self.height - 1), 4)],
        }
        if height is not None:
            feature["height"] = round(height, 4)
        return [], feature

    def check_set(self, features: list[dict[str, Any]]) -> list[str]:
        """Violations between tricks: count and overlap."""
        errors = []
        if len(features) > MAX_TRICKS:
            errors.append(f"{len(features)} tricks; at most {MAX_TRICKS}.")
        ordered = sorted(features, key=lambda f: f["from"][0])
        gap = 0.02
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt["from"][0] < prev["to"][0] + gap:
                errors.append(
                    f"'{prev['name']}' (to x={round(prev['to'][0] * GRID)}) and '{nxt['name']}' "
                    f"(from x={round(nxt['from'][0] * GRID)}) overlap; leave room to land between them."
                )
        return errors

    def overlay(self) -> np.ndarray:
        """The photo with the track drawn on it, as Muse Spark sees it."""
        vis = self.image.copy()
        points = self.track["points"]
        path = np.array([[int(x * (self.width - 1)), int(y * (self.height - 1))] for x, y in points], dtype=np.int32)
        cv2.polylines(vis, [path], False, (60, 255, 60), max(3, self.width // 200), cv2.LINE_AA)
        return vis


def design_tricks(image_path: str, inference_dir: str, send2unity_dir: str) -> dict[str, Any]:
    """Let Muse Spark place jumps and grinds on the track; writes features.json."""
    scene = TrackScene(image_path, inference_dir, send2unity_dir)
    candidates = scene.describe()
    out_path = os.path.join(send2unity_dir, "features.json")
    result: dict[str, Any] = {
        "generator": model_id(),
        "features": [],
        "mission": "",
        "design_notes": "",
        "candidates": candidates,
        "attempts": [],
    }
    if not candidates:
        result["generator"] = "none"
        result["design_notes"] = "No object meets the track, so there is nothing to jump or grind."
        print("  ✔ Tricks: no object meets the track; skipping Muse Spark.")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    conversation: list[dict[str, Any]] = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Current mission: {scene.mission}\nObjects that meet the track:\n" + json.dumps(candidates),
                },
                {"type": "input_image", "image_url": image_data_url(scene.overlay())},
            ],
        }
    ]

    attempts: list[dict[str, Any]] = []
    max_attempts = max(1, _get_int_env("MUSE_SPARK_MAX_ATTEMPTS", 3))
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        raw_text, usage = call_muse_spark(
            INSTRUCTIONS,
            conversation,
            "trick_plan",
            TRICKS_SCHEMA,
            os.getenv("MUSE_SPARK_REASONING_EFFORT", "medium"),
        )
        plan = json.loads(raw_text)
        checked = [scene.check_trick(trick) for trick in plan["tricks"]]
        features = [feature for _, feature in checked if feature]
        errors = [error for trick_errors, _ in checked for error in trick_errors] + scene.check_set(features)
        attempts.append(
            {
                "attempt": attempt,
                "plan": plan,
                "tricks": [
                    {**trick, "ok": not trick_errors} for trick, (trick_errors, _) in zip(plan["tricks"], checked)
                ],
                "errors": errors,
                "usage": usage,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        names = ", ".join(f"{t['type']} {t['object']}" for t in plan["tricks"]) or "none"
        print(f"  ✔ Muse Spark tricks attempt {attempt}: {names} ({len(errors)} violation(s))")
        for error in errors:
            print(f"    - {error}")
        if not errors:
            break

        conversation.append(
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": raw_text}]}
        )
        conversation.append(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "The tricks failed verification against the segmentation masks:\n- "
                        + "\n- ".join(errors)
                        + "\nReturn a corrected set (fix or drop the failing tricks).",
                    }
                ],
            }
        )

    # Keep the verified tricks of the last attempt; drop any that still overlap.
    last = attempts[-1]
    kept: list[dict[str, Any]] = []
    for feature in sorted((f for _, f in (scene.check_trick(t) for t in last["plan"]["tricks"]) if f), key=lambda f: f["from"][0]):
        if not scene.check_set(kept + [feature]):
            kept.append(feature)
    result.update(
        features=kept,
        mission=last["plan"]["mission"].strip() if kept else "",
        design_notes=last["plan"]["design_notes"],
        attempts=attempts,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    summary = ", ".join(f"{f['type']} {f['object']}" for f in kept) or "none"
    print(f"  ✔ Tricks saved: {summary}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Let Muse Spark add jumps and grinds to a finished run.")
    parser.add_argument("image_path", help="The (resized) input image the pipeline ran on")
    parser.add_argument("inference_dir", help="Step 1 output directory")
    parser.add_argument("send2unity_dir", help="Step 2/3 output directory")
    args = parser.parse_args()
    design_tricks(args.image_path, args.inference_dir, args.send2unity_dir)


if __name__ == "__main__":
    main()
