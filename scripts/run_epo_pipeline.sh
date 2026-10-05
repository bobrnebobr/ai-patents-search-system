#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

printf '\n[%s] Starting EPO dataset pipeline\n' "$(date '+%Y-%m-%d %H:%M:%S')"
uv run python scripts/download_epo_ops_xml.py --target-gib 1 --delay-seconds 5
uv run python scripts/combine_epo_xml.py \
  --input-dir data/raw/epo \
  --output data/exports/epo_patents_1g.xml
uv run python scripts/upload_to_rustfs.py \
  --file data/exports/epo_patents_1g.xml \
  --object-key epo/epo_patents_1g.xml
printf '[%s] EPO dataset pipeline completed\n' "$(date '+%Y-%m-%d %H:%M:%S')"
