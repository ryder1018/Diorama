"""Diorama pipeline: one photo in, a playable 2.5D level (and its showcase) out.

  1. perception.py       SAM 3.1 objects (Muse Spark picks the concepts)
  2. level_designer.py   Muse Spark picks the riding surface, verified against the masks
  3. depth_order.py      Muse Spark estimates each region's distance, checked against the ground
  4. layers.py           near / mid / far layers by distance, plus the ground mask
  5. track.py            the line along the ground's top edge; level_designer adds
                         the occluders and the mission
  6. feature_designer.py Muse Spark jumps and grinds (showcase only)
  7. muse_image.py       art styles and the background behind the front layers
  8. showcase.py         the self-contained 3D replay page

Every model call goes to Meta's models through the Meta Model API.
--preview stops after step 5; --resume starts at step 6 from an earlier preview.

Usage:
    python scripts/pipeline.py <photo> --out_dir <dir> --send2unity_dir <dir> [--styles comic,snow] [--preview | --resume]
"""

import argparse
import os
from collections.abc import Callable

from feature_designer import design_tricks
from depth_order import estimate_distances
from layers import cut_layers
from level_designer import finalize_level, plan_track, repair_track_occlusions
from muse_image import front_objects, load_setting, parse_styles, styles_from_env, stylize_run
from perception import analyze_photo, output_paths
from showcase import build_showcase
from track import trace_track


def _step(number: str, title: str) -> None:
    print(f"\n── Step {number}: {title} " + "─" * max(4, 60 - len(title)))


def build_level(
    image_path: str,
    out_dir: str,
    level_dir: str,
    after_layers: Callable[[], None] | None = None,
) -> dict:
    """Steps 1-5. Writes the Unity level into level_dir; returns track_points.json's content.

    after_layers runs once the layers exist, before tracing (the API starts
    its background fill there).
    """
    os.makedirs(level_dir, exist_ok=True)

    _step("1", "Objects (Muse Spark concepts, SAM 3.1)")
    paths = analyze_photo(image_path, out_dir)

    _step("2", "Riding surface (Muse Spark)")
    try:
        plan = plan_track(image_path, paths["labels"], paths["objects"])
    except Exception as exc:
        print(f"  ! Muse Spark surface choice failed ({exc}); falling back to the geometric guess.")
        plan = None
    ground_ids = plan["label_ids"] if plan else None

    _step("3", "Depth (Muse Spark)")
    depth = estimate_distances(image_path, paths["labels"], paths["objects"], ground_ids)
    distances = {int(region): meters for region, meters in depth["meters"].items()}

    _step("4", "2.5D layers")
    cut_layers(image_path, paths["labels"], level_dir, distances, ground_ids)
    if after_layers is not None:
        after_layers()

    _step("5", "Track + level")
    ground_path = os.path.join(level_dir, "layer_00_mask.npy")
    repair_track_occlusions(ground_path, paths["labels"])
    trace_track(ground_path, os.path.join(level_dir, "track_points.json"), image_path)
    return finalize_level(image_path, paths["labels"], paths["objects"], level_dir, plan)


def finish_level(image_path: str, out_dir: str, level_dir: str, styles: list[str] | None = None) -> None:
    """Steps 6-8: tricks, Muse Image art, the 3D showcase."""
    objects_path = output_paths(image_path, out_dir)["objects"]

    _step("6", "Tricks (Muse Spark, showcase only)")
    features_path = os.path.join(level_dir, "features.json")
    if os.path.exists(features_path):
        os.remove(features_path)
    try:
        design_tricks(image_path, out_dir, level_dir)
    except Exception as exc:
        print(f"  ! Muse Spark trick design failed ({exc}); the showcase rides without tricks.")

    styles = styles_from_env() if styles is None else styles
    _step("7", f"Art (Muse Image: {', '.join(styles) or 'background only'})")
    try:
        stylize_run(
            image_path,
            level_dir,
            styles,
            photo_background=True,
            setting=load_setting(objects_path),
            removed=front_objects(level_dir, objects_path),
        )
    except Exception as exc:
        print(f"  ! Muse Image failed ({exc}); the level keeps the photo's art and inpainted background.")

    _step("8", "3D showcase")
    build_showcase(image_path, out_dir, level_dir, os.path.join(level_dir, "showcase.html"))


def run_pipeline(
    image_path: str,
    out_dir: str = "./outputs/inference_results",
    level_dir: str = "./Send2Unity",
    styles: list[str] | None = None,
    preview: bool = False,
    resume: bool = False,
) -> None:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if resume and os.path.exists(os.path.join(level_dir, "track_points.json")):
        print("Resuming from the existing track and layers.")
    else:
        build_level(image_path, out_dir, level_dir)
    if preview:
        print(f"\nPreview ready (track + layers) in {level_dir}")
        return
    finish_level(image_path, out_dir, level_dir, styles)
    print(f"\nDone. Level and showcase are in {level_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn one photo into a Diorama level and its 3D showcase.")
    parser.add_argument("image_path", help="The photo (resized in place to the working width)")
    parser.add_argument("--out_dir", default="./outputs/inference_results", help="Label map, object list and depth")
    parser.add_argument("--send2unity_dir", default="./Send2Unity", help="The level: layers, track, styles, showcase")
    parser.add_argument("--styles", default=None, help="Muse Image styles: comma-separated, 'all' or 'none' (default: $MUSE_IMAGE_STYLES or all)")
    parser.add_argument("--preview", action="store_true", help="Stop after the track and layers (steps 1-5)")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing track and layers; run steps 6-8 only")
    args = parser.parse_args()
    run_pipeline(
        args.image_path,
        out_dir=args.out_dir,
        level_dir=args.send2unity_dir,
        styles=parse_styles(args.styles) if args.styles is not None else None,
        preview=args.preview,
        resume=args.resume,
    )
