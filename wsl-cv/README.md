# SkateP backend (`wsl-cv/`)

Turns one photo into a playable 2.5D skate level: depth-sorted layers, a riding line, the objects the skater passes behind, Muse Spark tricks, Muse Image art styles, and a 3D showcase of the whole run. Every model call goes to Meta's models through the Meta Model API, so it needs no GPU; installation is in the root [`README.md`](../README.md#installation). Set `MODEL_API_KEY` (or put it in the repo-root `.env`, which the scripts load).

## Pipeline

| Step | Module | What happens |
|---|---|---|
| 1 | `perception.py`, `sam31_segmentation.py` | Muse Spark lists the concepts worth segmenting; SAM 3.1 masks them. Writes `<base>_depth_mask.npy` (label map) and `<base>_sam31_objects.json`. |
| 2 | `level_designer.plan_track` | Muse Spark picks the riding surface; the choice is verified against the masks and repaired if needed. |
| 3 | `depth_order.py` | Muse Spark estimates each region's distance in meters from the photo and a numbered copy of the regions. Checked against the ground plane: of two objects standing on the riding surface, the one whose base is clearly lower in the photo must be nearer; violations are repaired like the surface choice. Writes `<base>_depth_order.json` and, for the showcase relief, `<base>_depth_16bit.png`. |
| 4 | `layers.py` | Regions split into near / mid / far by distance (thirds; ties go nearer); writes `merged_layer_01..03.png`, the ground mask `layer_00_mask.npy` and `layers.json`. |
| 5 | `track.py`, `level_designer` | The ground edge is continued behind standing objects, traced into `track_points.json`, and given its occluders and mission (`finalize_level`). |
| 6 | `feature_designer.py` | Muse Spark jumps and grinds, verified against the masks (`features.json`, showcase only). |
| 7 | `muse_image.py` | Art styles and the background hidden behind the front layers. |
| 8 | `showcase.py` | The self-contained 3D replay page. |

`scripts/pipeline.py` runs them in order: `--preview` stops after step 5, `--resume` starts at step 6 from an earlier preview. The API (`scripts/infer.py`) runs steps 1–5 and, when a style is asked for, step 7.

```bash
python scripts/pipeline.py photo.jpg --out_dir run/outputs --send2unity_dir run [--styles comic,snow] [--preview | --resume]
```

### Trying Photos
Drop photos into `wsl-cv/photos/` and run `./run_photos.sh` (inside the project env). Each new photo gets `showcase/<name>/showcase.html`, and `showcase/index.html` links them all; already-built photos are skipped (`--force` rebuilds, `--video` also renders MP4s). For a single photo anywhere on disk: `./try_photo.sh <photo> [name] [--video]`.

To pick photos quickly, `./run_photos.sh --preview` runs only SAM 3.1 and the Muse Spark track and depth (about 1.5 minutes per photo; no tricks, Muse Image or 3D page) and opens `showcase/preview.html`: each photo with its track line, rideable ground and the objects the skater passes behind, or its near / track / mid / far layers in an exploded 3D stack. Mark photos with ☆ pick (← → browse, P picks, 1 2 3 switch views in the enlarged view), copy the generated `./run_photos.sh <names>` command, and run it: picked photos are finished from their previewed track (`pipeline.py --resume`), so the showcase rides the line you chose. Names given to `run_photos.sh` limit any mode to those photos.

`./run_photos.sh --restyle` adds Muse Image art styles to every showcase that is already built, reusing its SAM 3.1 / depth / track outputs (only the Muse Image calls run; cached styles are kept unless `--force`).

### Muse Image Art Styles
After the layers and the track exist, `scripts/muse_image.py` asks Muse Image (`muse-image-1.0`, Responses API, $0.01/image) to repaint the photo in each style (`comic`, `cyberpunk`, `snow`, `pixel`, `watercolor`; the prompts are in `STYLES`). The styles run in parallel, and each takes 15–45 s. Every prompt also tells the model to keep the composition (same camera, objects, outlines).

Because the masks are reused, the composition is measured instead of assumed:
1. `size: "auto"` keeps the input's aspect ratio. The tool only accepts `1024x1024`, `1536x1024`, `1024x1536` or `auto`; arbitrary `WxH` values are rejected with HTTP 400.
2. The result is resized to the photo and registered with SIFT + RANSAC (a similarity transform). The transform is applied only when it is plausible and improves the fit.
3. The fit is scored where it matters: points are sampled along the outlines the foreground/gameplay layers were cut on. At each point a 64 px gradient patch of the photo is phase-correlated against the styled image to get the local shift (`alignment.outline_offset`: `median_px`, `within_3px`). Across the 8 sample levels × 5 styles, the median drift is 0.5–2.5 px, with 54–99 % of points within 3 px. The registration scales stay within 1–4 % of 1.0. A plain edge F1 (`edge_f1_raw` / `edge_f1`) is also stored, but it penalizes new texture (bricks, rain, neon), so read it as a style-change score, not an alignment score. As a sanity check, shifting a restyle by 8 px moves the median to about 5 px and drops it to about 35 % within 3 px.
4. The layers are re-cut from the styled pixels with the exact alphas of `merged_layer_01/02.png`. The background is filled in by a second Muse Image turn on the styled image (see Background fill below). All styles therefore share one foreground / gameplay / background split, one track and one set of occluders.

Outputs: `<send2unity_dir>/styles/<name>/{styled_raw,styled,fill_raw,merged_layer_01..03}.png` and `<send2unity_dir>/styles.json` (alignment, timing, token usage, a per-style `track_color` and `background` source). Server errors and timeouts are retried (`MUSE_IMAGE_RETRIES`, default 2). A style that still fails, for example one blocked by Muse Image's content filter, is left out, and the run keeps the others.

**Background fill:** `layers.py` fills the background layer with `cv2.inpaint`, which smears across large holes. Muse Image replaces it: everything the foreground and gameplay layers cover (grown by a few pixels) is painted solid magenta, and Muse Image is asked to fill the magenta by continuing whatever surrounds it in the same place, adding no new objects. Most of the frame can be magenta, so the prompt also carries Muse Spark's one-sentence `setting` of the photo (e.g. "a bright indoor room with grey and yellow foam floor tiles and mint-green walls"), written in the same call that proposes the SAM 3.1 concepts and stored in `*_sam31_objects.json`; older runs get one from a separate Muse Spark call (`--objects`). The prompt names no landscape features: an earlier version that listed mountains and vegetation turned desks and rooms into forests. The result is registered like a restyle and only the filled pixels are used; the rest keeps the original pixels. Because the hidden objects are erased from the input, they cannot come back as ghosts, including objects SAM 3.1 never named. Muse Image ignores `input_image_mask` (tested: accepted, no effect), hence the key color. Each style is filled from its own image, and its fill prompt spells out that style's look and puts it above the setting's lighting (the setting is written from the photo, so "a bright lawn under a blue sky" otherwise turned the fill of a cyberpunk night or a snow scene back into a sunny photo when most of the frame was magenta), and the photo gets one too (`<send2unity_dir>/background/`, `background.json`). Fills are cached per image and holes. A hole keeps its object's outline, which invited the model to paint the object back (a hydrant-shaped hole became a hydrant), so the prompt also names the removed objects (the front-layer SAM 3.1 objects that Muse Spark did not list as `fixed`, i.e. not ground, walls or buildings) and asks for background only; an object can still come back now and then. Two checks run on every fill: no more than 2 % of the area may stay magenta, and the colors just outside the holes, which the fill was told to keep, may not shift by more than 20 (Lab, `MUSE_IMAGE_FILL_MAX_DRIFT`; good fills stay under ~13, while fills that redrew the picture in another look, such as a cyberpunk scene coming back as a daylight photo, measured 27–79). A failed fill is made again (`MUSE_IMAGE_FILL_ATTEMPTS`, default 2 calls) and then falls back to `cv2.inpaint`; `MUSE_IMAGE_FILL=0` turns fills off.

```bash
python scripts/muse_image.py <image> <send2unity_dir> [--styles comic,snow|all|none] [--force] [--no-photo-background] [--objects <run>_sam31_objects.json]
```
`pipeline.py` runs every style by default (`--styles` or `MUSE_IMAGE_STYLES` narrows it; `none` skips it). The `/infer` API restyles only when asked: send the form field `style=<name>`, or set `MUSE_IMAGE_STYLE=<name>` on the server. The returned layers are then the styled ones, `metadata.track_color` follows the art, and `metadata.art_style` records the style. The restyle call starts as soon as the photo arrives, alongside SAM 3.1 and Muse Spark, and the background fill follows it (for an unstyled level it runs alongside track extraction). In tests, with Muse Spark verifying on the first attempt, a level took about 75–105 s. If Muse Image fails, the photo's layers are returned.

### Muse Spark Tricks (showcase only)
After the track exists, `scripts/feature_designer.py` asks Muse Spark where a skater would jump or grind on it. It sees the photo with the track drawn on it and every SAM 3.1 object that meets the track (where, how tall, whether its base stands on the line, whether it is in front of the skater), and proposes up to four tricks, or none. Each trick is checked against the masks:
- **jump**: the object must stand on the track (its base on the line, not on the floor in front of it) and not be in front of the skater; the span (40–300 on the 0..1000 grid) must cover every column where the object meets the track. The arc apex needed to clear the object's real top edge is computed from the mask and must stay under 40 % of the photo height (`TRICK_MAX_AIR`); when it does not, the feedback names the narrowest span that would clear it, or says that none does.
- **grind**: the object must be skateable and its top edge must follow the track (within `TRICK_EDGE_TOLERANCE`, 3.5 % of the height) along at least 60 % of the span; the feedback names where the edge actually meets the track.
- Tricks may not overlap.

Violations go back to Muse Spark for repair (up to `MUSE_SPARK_MAX_ATTEMPTS`); tricks that still fail are dropped. When no object meets the track, Muse Spark is not called. The result is `<send2unity_dir>/features.json` (tricks, a mission that names them, every attempt); `track_points.json` and the `/infer` response for Unity are unchanged. `pipeline.py` runs it as step 5; `./run_photos.sh --tricks` (re)runs it on every built showcase and rebuilds the pages. On the 9 sample levels it placed a jump on 2, a grind on 2 and nothing on 5 (plazas, lawns and streets where only buildings, trees or people meet the line).

```bash
python scripts/feature_designer.py <image> <inference_dir> <send2unity_dir>
```

### 3D Pipeline Showcase
Every `scripts/pipeline.py` run also writes `<send2unity_dir>/showcase.html`: a self-contained three.js page that replays the run (photo → Muse Spark concepts → SAM 3.1 objects → Muse Spark depth relief, each object labelled with its distance → 2.5D layers → Muse Image restyles → Muse Spark design/verify loop → Muse Spark tricks → ride). The Tricks step appears only when `features.json` exists and Muse Spark was asked: each rejected set draws in red with its violations and fades, then the verified jumps (cyan arcs, as high as the mask check measured) and grinds (orange) draw; in the ride the skater pops over the arcs and throws sparks on the grinds. Open it in a browser (it loads three.js from a CDN). Controls: space play/pause, R restart, 1–8 jump to a step, drag to orbit, C back to the cinematic camera, H hide the UI for screen recording, M sound on/off, F foreground fade on/off (by default the foreground fades to 35% while the skater is behind it, like Unity's ForegroundOcclusionFader; off keeps it opaque so real objects fully hide the skater; `?fade=0` starts with it off).

**Parallax planes:** the page splits the foreground and gameplay layers into up to two planes each where Muse Spark's distances jump (3–5 planes in all; the background stays one opaque plane, and Unity still gets the three layers). A small index map says which pixels of a layer belong to which plane, so every art style splits the same way; each plane reaches 3 px into its neighbours, so parallax does not open cracks between them. In the Ride step every plane, the track and the skater are scaled by their distance to the camera, so from the camera the planes fit back together into the photo while camera moves still slide them apart.

**Skater:** in the Ride step a rider stands on the board: a sideways stance facing the camera, built from capsules and posed each frame with 2-bone IK (two kick pushes at the start, knee bob and arm sway while cruising, a crouch-pop over jump spans), with a lime outline so it reads on any art style. It is drawn between the gameplay and foreground layers like the board, so foreground objects hide it.

**Art style switcher:** the bar at the top (Photo · Comic · Cyberpunk · …) swaps the art on the depth relief and on all three layers at any time, with a wipe transition. `S` cycles through the styles. During the Restyle step the timeline tours every style, then keeps one for the track and ride steps: the first style by default, or the one in `?style=<name>` (`?style=original` keeps the photo). Picking a style overrides the tour until `R`. The gallery (`showcase/index.html`) links each level once per style.

**Soundtrack:** the page synthesizes its own music and sound effects with the Web Audio API, so there are no audio files and no third-party audio licenses. A lo-fi beat (96 BPM, Fmaj7 · Em7 · Dm7 · Cmaj9) builds up step by step: pad first, then drums and bass, a muffled beat during the Track step, and a drop on the first downbeat of the Ride. The sound effects are timed from the level's own data: a shutter for the photo, a rumble as the depth relief rises, one note per Muse Spark concept and per SAM 3.1 object, a swoosh along each restyle wipe, a buzz for each rejected Muse Spark attempt and a chime for the verified one, then rolling wheels, ollie pops, landings and grind scrapes, muffled while the skater is behind a foreground object. The whole score is rendered once with an `OfflineAudioContext` (~1–5 s) and then plays locked to the timeline (scrubbing, jumping to a step and pausing all stay in sync). Browsers only start audio after a click or a key press; the speaker button or `M` toggles it.

To render the same sequence to MP4 locally (headless Chromium + ffmpeg, ~1 s per 1080p frame). The MP4 gets the same soundtrack as an AAC track; `--no-audio` leaves it out:
```bash
pip install playwright && python -m playwright install chromium   # once
python scripts/render_showcase.py <send2unity_dir>/showcase.html --out showcase.mp4 [--style cyberpunk] [--no-audio] [--no-fade]
```

## Modules

*   `scripts/pipeline.py`: the steps above, as a CLI and as `build_level` / `finish_level` for the API.
*   `scripts/infer.py`: the level API (FastAPI). Each request runs in its own temporary folder.
*   `scripts/perception.py`: step 1.
*   `scripts/depth_order.py`: step 3.
*   `scripts/sam31_segmentation.py`: Muse Spark proposes this photo's concepts (skateable + scenery) and a one-line setting; SAM 3.1 segments each concept through the Meta Model API; the masks become one label map, small objects first so handrails keep their own region.
*   `scripts/muse_spark.py`: shared Muse Spark client (Responses API, JSON-schema output; model `$MUSE_SPARK_MODEL_ID`, default `muse-spark-1.3-contributor`).
*   `scripts/level_designer.py`: Muse Spark surface choice with mask verification, track repair behind standing objects, occluders (foreground objects in the skater's corridor, also written as Unity's `occluded_segments`), mission, `level_vis.png`.
*   `scripts/layers.py`, `scripts/track.py`: steps 3 and 4.
*   `scripts/feature_designer.py`, `scripts/muse_image.py`: steps 5 and 6.
*   `scripts/showcase.py` + `showcase_template.html`, `scripts/render_showcase.py`: the showcase page and its MP4 renderer.
*   `scripts/showcase_index.py`, `scripts/preview_index.py` + `preview_template.html`: the gallery and the photo picker.
*   `scripts/config.py`: reads the tuning environment variables.

## API

```bash
uvicorn scripts.infer:app --host 0.0.0.0 --port 9000      # from wsl-cv/
curl -X POST http://127.0.0.1:9000/infer -F "photo=@photo.jpg" -F "style=comic" -o level.json
```

`POST /infer` takes the multipart fields `photo` and, optionally, `style` (`comic`, `cyberpunk`, `snow`, `pixel`, `watercolor`, or `original`; default `$MUSE_IMAGE_STYLE`, else the photo's own pixels). A level takes about 75–105 s. The response:

```json
{
  "images": {
    "preview_base64": "PNG: layers composited, track drawn on top",
    "foreground_base64": "PNG: near layer, drawn in front of the skater",
    "gameplay_base64": "PNG: mid layer",
    "background_base64": "PNG: far layer (opaque)"
  },
  "metadata": {
    "version": "1.0",
    "timestamp": "2026-11-02T03:58:23.000000Z",
    "source_image": "photo.jpg",
    "aspect_ratio": 1.3333,
    "track_color": "#81786f",
    "points": [[0.0, 0.7909], [0.0182, 0.7804], "... x, y in 0..1, left to right"],
    "occluded_segments": [[0, 14]],
    "level": {
      "mission": "Shred the desk mat behind the keyboard!",
      "surfaces": ["desk mat_1"],
      "occluders": [{"id": "keyboard_2", "label": "keyboard", "x_range": [0.0, 0.2708]}],
      "render_order": ["background", "gameplay", "track", "foreground"],
      "generator": "muse-spark-1.3-contributor",
      "design_notes": "...",
      "validation": {"passed": true, "attempts": 1, "errors": []}
    },
    "art_style": {"name": "comic", "label": "Comic", "model": "muse-image", "edge_f1": 0.64}
  }
}
```

`art_style` is present only for a restyled level. `GET /health` answers `{"status": "ok"}`.

## Configuration

| Variable | Default | |
|---|---|---|
| `MODEL_API_KEY` | (required) | Meta Model API key |
| `MUSE_IMAGE_STYLES` / `MUSE_IMAGE_STYLE` | all / none | styles for `pipeline.py` / the API |
| `MUSE_SPARK_MAX_ATTEMPTS` | 3 | verify-and-repair rounds for the surface, the depth and the tricks |
| `PIPELINE_OUTPUT_WIDTH`, `PIPELINE_INFERENCE_WIDTH` | 1024, 768 | working width of the level / of SAM 3.1's input |
| `LEVEL_SKATER_HEIGHT` | 0.15 | skater height (fraction of the photo) used for occluders and tricks |

