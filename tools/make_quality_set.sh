#!/bin/zsh
# Builds the quality A/B set for `vsr-bench --quality`: 1080p reference frames from the Xiph
# derf test sequences and their 960x540 inputs, both as a clean Lanczos downscale and after an
# H.264 round trip at a streaming bitrate.
#
#   tools/make_quality_set.sh OUT_DIR [FRAMES]
#
# OUT_DIR/ref/<clip>_<n>.png     1920x1080 reference
# OUT_DIR/clean/<clip>_<n>.png   960x540 Lanczos downscale of the reference
# OUT_DIR/h264/<clip>_<n>.png    the same downscale encoded with libx264 at 1.5 Mbit/s, decoded
#
# Only the first FRAMES (default 60) frames of each sequence are streamed; every 6th frame from
# frame 6 on is kept. Needs ffmpeg with libx264.
set -euo pipefail
OUT=${1:?usage: make_quality_set.sh OUT_DIR [FRAMES]}
FRAMES=${2:-60}
BASE=https://media.xiph.org/video/derf/y4m
CLIPS=(crowd_run_1080p50 park_joy_1080p50 ducks_take_off_1080p50 in_to_tree_1080p50
       old_town_cross_1080p50 pedestrian_area_1080p25 sunflower_1080p25
       riverbed_1080p25 station2_1080p25)
mkdir -p "$OUT"/{ref,clean,h264,src}
# BT.709 limited range in, full-range RGB out, as a browser presents SDR video.
RGB="scale=out_color_matrix=bt709:out_range=full:in_color_matrix=bt709:in_range=tv,format=rgb24"
KEEP="select='not(mod(n\,6))*gte(n\,6)'"
for c in $CLIPS; do
  name=${c%_1080p*}
  src="$OUT/src/$name.y4m"
  [[ -s $src ]] || ffmpeg -nostdin -loglevel error -i "$BASE/$c.y4m" -frames:v "$FRAMES" \
    -vf "crop=1920:1080,format=yuv420p" -y "$src"
  ffmpeg -nostdin -loglevel error -i "$src" -vf "$KEEP,$RGB" -fps_mode passthrough \
    -start_number 0 -y "$OUT/ref/${name}_%02d.png"
  ffmpeg -nostdin -loglevel error -i "$src" \
    -vf "$KEEP,scale=960:540:flags=lanczos,$RGB" -fps_mode passthrough \
    -start_number 0 -y "$OUT/clean/${name}_%02d.png"
  ffmpeg -nostdin -loglevel error -i "$src" -vf "scale=960:540:flags=lanczos" \
    -c:v libx264 -preset medium -b:v 1500k -maxrate 2000k -bufsize 3000k -pix_fmt yuv420p \
    -y "$OUT/src/${name}_540p.mp4"
  ffmpeg -nostdin -loglevel error -i "$OUT/src/${name}_540p.mp4" -vf "$KEEP,$RGB" \
    -fps_mode passthrough -start_number 0 -y "$OUT/h264/${name}_%02d.png"
  echo "$name: $(ls "$OUT/ref" | grep -c "^${name}_") frames"
done
