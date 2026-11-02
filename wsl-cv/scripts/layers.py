"""Step 3: cut the photo into 2.5D layers and the ground mask.

Every SAM 3.1 region, and the unsegmented rest of the photo, joins the near,
mid or far layer by its mean DA3 depth: the regions are split into thirds.
The skater rides between the mid and near layers, so near objects are the
ones it passes behind.

  merged_layer_01.png   near   RGBA, soft-edged object cut-outs
  merged_layer_02.png   mid    RGBA
  merged_layer_03.png   far    opaque; what the front layers hide is inpainted
                               (Muse Image later replaces this fill)
  layer_00_mask.npy     the riding surface (255) the track is traced along
  layers.json           which SAM region went into which layer

Background pixels hugging an object's outline belong to neither: they are
where SAM's mask edge and the real edge disagree, so the far layer inpaints
them instead of showing a halo.
"""

import json
import os

import cv2
import numpy as np
import scipy.ndimage

LAYER_FILES = ("merged_layer_01.png", "merged_layer_02.png", "merged_layer_03.png")
LAYER_NAMES = ("near", "mid", "far")
SEAM_PX = 4
GROW_PX = 2
FEATHER_PX = 2


def edge_seam(labels: np.ndarray) -> np.ndarray:
    """Unsegmented pixels within SEAM_PX of an object."""
    objects = (labels != 0).astype(np.uint8)
    # Each pass of a 5x5 kernel grows the objects by 2 px.
    return (labels == 0) & (cv2.dilate(objects, np.ones((5, 5), np.uint8), iterations=SEAM_PX // 2) > 0)


def assign_layers(labels: np.ndarray, depth: np.ndarray, count: int = len(LAYER_FILES)) -> dict[int, int]:
    """Layer index (0 = near) for every region id >= 0, by thirds of mean depth."""
    ids = [int(i) for i in np.unique(labels) if i >= 0]
    if len(ids) <= 1:
        return {i: 0 for i in ids}
    means = np.asarray(scipy.ndimage.mean(depth, labels, ids), dtype=np.float64)
    cuts = np.quantile(means, [k / count for k in range(1, count)])
    layer = np.searchsorted(cuts, means, side="right")
    # The thirds can land exactly on a region; such a region is ambiguous, so
    # it stays on the middle layer, which neither hides the skater nor is
    # painted over by the background fill.
    on_cut = layer != np.searchsorted(cuts, means, side="left")
    layer = np.minimum(np.where(on_cut, count // 2, layer), count - 1)
    return dict(zip(ids, layer.tolist()))


def guess_ground(labels: np.ndarray) -> list[int]:
    """Fallback when Muse Spark gave no surface: the region spanning the most width lowest down."""
    height, width = labels.shape
    best, best_score = None, -1.0
    for region in np.unique(labels):
        if region <= 0:
            continue
        ys, xs = np.nonzero(labels == region)
        score = np.unique(xs).size / width + ys.max() / (height - 1)
        if score > best_score:
            best, best_score = int(region), score
    return [best] if best is not None else []


def _soft_alpha(mask: np.ndarray) -> np.ndarray:
    grown = cv2.dilate(mask, np.ones((2 * GROW_PX + 1,) * 2, np.uint8))
    return cv2.GaussianBlur(grown, (0, 0), sigmaX=FEATHER_PX / 3.0)


def cut_layers(
    image_path: str,
    depth_path: str,
    labels_path: str,
    out_dir: str,
    ground_ids: list[int] | None = None,
) -> dict:
    """Write the three layer images, the ground mask and layers.json; returns the layers.json content."""
    photo = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    labels = np.load(labels_path).astype(np.int32)
    if photo is None or depth is None:
        raise FileNotFoundError(f"Missing photo or depth map for {image_path}")
    os.makedirs(out_dir, exist_ok=True)

    labels[edge_seam(labels)] = -1
    layer_of = assign_layers(labels, depth)

    present = set(layer_of)
    ground = [i for i in (ground_ids or []) if i in present]
    if ground:
        print(f"  ✔ Ground: Muse Spark surface region(s) {ground}")
    else:
        ground = guess_ground(labels)
        print(f"  ! No Muse Spark surface; using the widest low region {ground}")
    ground_mask = np.isin(labels, ground).astype(np.uint8) * 255
    np.save(os.path.join(out_dir, "layer_00_mask.npy"), ground_mask)

    bgra = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    last = len(LAYER_FILES) - 1
    for index, (file, name) in enumerate(zip(LAYER_FILES, LAYER_NAMES)):
        members = [i for i, layer in layer_of.items() if layer == index]
        mask = np.isin(labels, members).astype(np.uint8) * 255
        if index == last:
            image = cv2.cvtColor(cv2.inpaint(photo, 255 - mask, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2BGRA)
        else:
            image = bgra.copy()
            image[:, :, 3] = _soft_alpha(mask)
        cv2.imwrite(os.path.join(out_dir, file), image)
        print(f"  ✔ {file}: {name}, {len(members)} region(s)")

    info = {
        "layers": [
            {"file": file, "region_ids": sorted(i for i, layer in layer_of.items() if layer == index and i > 0)}
            for index, file in enumerate(LAYER_FILES)
        ],
        "track_region_ids": sorted(ground),
    }
    with open(os.path.join(out_dir, "layers.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info
