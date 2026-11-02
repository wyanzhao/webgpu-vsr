#!/bin/sh
# Regenerates the VP9 clips used by the overlay/session e2e tests.
# Playwright's Chromium build has no proprietary codecs, so the clips are VP9/WebM.
set -eu
cd "$(dirname "$0")"
enc="-c:v libvpx-vp9 -deadline realtime -cpu-used 8 -pix_fmt yuv420p"
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=960x540:rate=30:duration=6" $enc -b:v 600k clip_540p30.webm
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=1280x720:rate=60:duration=6" $enc -b:v 900k clip_720p60.webm
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "smptehdbars=size=640x272:rate=24:duration=6" $enc -b:v 300k clip_scope_24.webm
