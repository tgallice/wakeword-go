#!/usr/bin/env bash
# Downloads the microWakeWord v2 models from the esphome/micro-wake-word-models repository,
# at a pinned commit, into testdata/models/v2/.
set -euo pipefail
COMMIT="${MWW_MODELS_COMMIT:-05b65922cc433c9df13e98e32a7fe520758c837e}"
BASE="https://raw.githubusercontent.com/esphome/micro-wake-word-models/${COMMIT}/models/v2"
DEST="$(cd "$(dirname "$0")/../.." && pwd)/testdata/models/v2"
mkdir -p "$DEST"
for m in alexa hey_jarvis hey_mycroft okay_nabu vad; do
  for ext in tflite json; do
    curl -fsSL -o "$DEST/$m.$ext" "$BASE/$m.$ext"
    echo "ok $m.$ext"
  done
done
