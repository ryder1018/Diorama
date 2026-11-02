#!/usr/bin/env bash
# Build a showcase for every photo dropped into photos/, then open the gallery.
#
#   ./run_photos.sh            # only photos without a showcase yet
#   ./run_photos.sh cup street # only these photos (names as in showcase/)
#   ./run_photos.sh --preview  # quick pass: SAM 3.1 + Muse Spark track, depth and layers only
#                              # (no tricks, Muse Image or 3D page), then open showcase/preview.html
#                              # to pick photos; building a picked one later keeps its previewed track.
#   ./run_photos.sh --force    # rebuild all of them
#   ./run_photos.sh --video    # also render an MP4 for each processed photo
#   ./run_photos.sh --restyle  # (re)generate Muse Image art styles and background fills for every built
#                              # showcase, reusing its SAM/track outputs;
#                              # cached styles are kept unless --force is given.
#                              # MUSE_IMAGE_STYLES=comic,snow picks styles.
#   ./run_photos.sh --tricks   # (re)place Muse Spark jumps/grinds (features.json) on every built
#                              # showcase, reusing its track, and rebuild the page.
#
# Photos stay where they are; each one gets showcase/<name>/ (name = file name
# reduced to [A-Za-z0-9_-]) and showcase/index.html links them all.
# Run inside the project's Python env (see the root README).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PHOTOS="$HERE/photos"
PYTHON="${PYTHON:-python}"
export PYTHONNOUSERSITE=1
FORCE=0
VIDEO_ARG=""
RESTYLE=0
TRICKS=0
PREVIEW=0
ONLY=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --video) VIDEO_ARG="--video" ;;
    --restyle) RESTYLE=1 ;;
    --tricks) TRICKS=1 ;;
    --preview) PREVIEW=1 ;;
    -*) echo "unknown option: $arg" >&2; exit 1 ;;
    *) ONLY+=("$arg") ;;
  esac
done
wanted() {
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local n; for n in "${ONLY[@]}"; do [ "$n" = "$1" ] && return 0; done
  return 1
}

# MODEL_API_KEY for SAM 3.1 / Muse Spark lives in the repo root .env.
if [ -z "${MODEL_API_KEY:-}" ] && [ -f "$HERE/../.env" ]; then
  set -a; . "$HERE/../.env"; set +a
fi
mkdir -p "$PHOTOS"
done_count=0
failed=()
shopt -s nullglob nocaseglob

if [ "$PREVIEW" -eq 1 ]; then
  for photo in "$PHOTOS"/*.{jpg,jpeg,png,webp,bmp}; do
    base="$(basename "${photo%.*}")"
    ext="${photo##*.}"
    name="$(printf '%s' "$base" | tr -cs 'A-Za-z0-9_-' '_' | sed 's/^_*//; s/_*$//')"
    wanted "$name" || continue
    dir="$HERE/showcase/$name"
    if [ "$FORCE" -eq 0 ] && [ -f "$dir/track_points.json" ]; then
      echo "skip  $name (already has a track; --force to redo)"
      continue
    fi
    if [ -f "$dir/showcase.html" ]; then
      echo "skip  $name (already built; preview would replace its track)"
      continue
    fi
    echo "look  $name  <-  $(basename "$photo")"
    mkdir -p "$dir"
    rm -f "$dir/track_points.json"
    cp "$photo" "$dir/$name.$ext"
    if (cd "$HERE" && "$PYTHON" scripts/pipeline.py "$dir/$name.$ext" --out_dir "$dir/outputs" --send2unity_dir "$dir" --preview) \
        > "$dir/preview.log" 2>&1 && [ -f "$dir/track_points.json" ]; then
      grep -E "Level saved" "$dir/preview.log" | sed 's/^ */      /'
      done_count=$((done_count + 1))
    else
      echo "      FAILED, see showcase/$name/preview.log"
      failed+=("$name")
    fi
  done
  "$PYTHON" "$HERE/scripts/preview_index.py" "$HERE/showcase"
  echo "Previewed $done_count photo(s)${failed:+, failed: ${failed[*]}}."
  if [ "${NO_OPEN:-0}" != "1" ] && command -v explorer.exe >/dev/null 2>&1; then
    explorer.exe "$(wslpath -w "$HERE/showcase/preview.html")" || true
  fi
  exit 0
fi

if [ "$TRICKS" -eq 1 ]; then
  for dir in "$HERE"/showcase/*/; do
    name="$(basename "$dir")"
    wanted "$name" || continue
    photo=""
    for candidate in "$dir$name".{jpg,jpeg,png,webp,bmp}; do photo="$candidate"; break; done
    if [ -z "$photo" ] || [ ! -d "${dir}outputs" ] || [ ! -f "${dir}track_points.json" ]; then
      echo "skip  $name (no pipeline outputs)"
      continue
    fi
    echo "trick $name"
    if (cd "$HERE" && "$PYTHON" scripts/feature_designer.py "$photo" "${dir}outputs" "$dir" \
        && "$PYTHON" scripts/showcase.py "$photo" "${dir}outputs" "$dir") > "${dir}tricks.log" 2>&1; then
      grep -E "✔|✘|^    - " "${dir}tricks.log" | sed 's/^ */      /'
      done_count=$((done_count + 1))
    else
      echo "      FAILED, see showcase/$name/tricks.log"
      failed+=("$name")
    fi
  done
  "$PYTHON" "$HERE/scripts/showcase_index.py" "$HERE/showcase"
  echo "Placed tricks on $done_count showcase(s)${failed:+, failed: ${failed[*]}}."
  exit 0
fi

if [ "$RESTYLE" -eq 1 ]; then
  FORCE_ARG=""
  [ "$FORCE" -eq 1 ] && FORCE_ARG="--force"
  for dir in "$HERE"/showcase/*/; do
    name="$(basename "$dir")"
    wanted "$name" || continue
    photo=""
    for candidate in "$dir$name".{jpg,jpeg,png,webp,bmp}; do photo="$candidate"; break; done
    if [ -z "$photo" ] || [ ! -d "${dir}outputs" ] || [ ! -f "${dir}merged_layer_01.png" ]; then
      echo "skip  $name (no pipeline outputs to restyle)"
      continue
    fi
    echo "style $name"
    if (cd "$HERE" && "$PYTHON" scripts/muse_image.py "$photo" "$dir" --styles "${MUSE_IMAGE_STYLES:-all}" $FORCE_ARG \
        --objects "${dir}outputs/${name}_sam31_objects.json" \
        && "$PYTHON" scripts/showcase.py "$photo" "${dir}outputs" "$dir") > "${dir}restyle.log" 2>&1; then
      grep -E "✔|✘" "${dir}restyle.log" | tail -n 6 | sed 's/^ */      /'
      done_count=$((done_count + 1))
    else
      echo "      FAILED, see showcase/$name/restyle.log"
      failed+=("$name")
    fi
  done
  "$PYTHON" "$HERE/scripts/showcase_index.py" "$HERE/showcase"
  echo "Restyled $done_count showcase(s)${failed:+, failed: ${failed[*]}}."
  exit 0
fi

for photo in "$PHOTOS"/*.{jpg,jpeg,png,webp,bmp}; do
  base="$(basename "${photo%.*}")"
  name="$(printf '%s' "$base" | tr -cs 'A-Za-z0-9_-' '_' | sed 's/^_*//; s/_*$//')"
  wanted "$name" || continue
  dir="$HERE/showcase/$name"
  if [ "$FORCE" -eq 0 ] && [ -f "$dir/showcase.html" ]; then
    echo "skip  $name (already built; --force to rebuild)"
    continue
  fi
  log="$dir/run.log"
  # A --preview run left the track that was picked: finish from it instead of
  # segmenting again (SAM 3.1 / Muse Spark may draw a different one).
  previewed=""
  for candidate in "$dir/$name".{jpg,jpeg,png,webp,bmp}; do previewed="$candidate"; break; done
  if [ "$FORCE" -eq 0 ] && [ -n "$previewed" ] && [ -f "$dir/track_points.json" ]; then
    echo "build $name  (from its preview track)"
    run_build() {
      (cd "$HERE" && "$PYTHON" scripts/pipeline.py "$previewed" --out_dir "$dir/outputs" --send2unity_dir "$dir" --resume) \
        && { [ -z "$VIDEO_ARG" ] || (cd "$HERE" && "$PYTHON" scripts/render_showcase.py "$dir/showcase.html" --out "$dir/showcase.mp4"); }
    }
  else
    echo "build $name  <-  $(basename "$photo")"
    run_build() { NO_OPEN=1 PYTHON="$PYTHON" "$HERE/try_photo.sh" "$photo" "$name" $VIDEO_ARG; }
  fi
  mkdir -p "$dir"
  if run_build > "$log" 2>&1 && grep -q "Showcase saved" "$log"; then
    grep -E "Level saved" "$log" | sed 's/^ */      /'
    done_count=$((done_count + 1))
  else
    echo "      FAILED, see showcase/$name/run.log"
    failed+=("$name")
  fi
done

"$PYTHON" "$HERE/scripts/showcase_index.py" "$HERE/showcase"
"$PYTHON" "$HERE/scripts/preview_index.py" "$HERE/showcase" > /dev/null
echo "Built $done_count new showcase(s)${failed:+, failed: ${failed[*]}}."
if [ "${NO_OPEN:-0}" != "1" ] && command -v explorer.exe >/dev/null 2>&1; then
  explorer.exe "$(wslpath -w "$HERE/showcase/index.html")" || true
fi
