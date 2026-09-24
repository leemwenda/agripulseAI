#!/usr/bin/env bash
# Download FMD cattle image datasets for AgriPulse.
# Run this on YOUR machine (not in a sandboxed environment) — it needs
# outbound internet access to zenodo.org and, optionally, kaggle.com.
#
# Usage:
#   chmod +x download_fmd_datasets.sh
#   ./download_fmd_datasets.sh [destination_dir]
#
# Default destination: ~/agripulse-ai/disease-detection/data/fmd_raw

set -euo pipefail

DEST="${1:-$HOME/agripulse-ai/disease-detection/data/fmd_raw}"
mkdir -p "$DEST/zenodo" "$DEST/kaggle"

command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Zenodo — FMD Cattle (Osborn, University of Surrey), CC-BY 4.0
#    https://zenodo.org/records/7779246
#    Downloaded via the Zenodo API so it always grabs the current files,
#    rather than hardcoding a filename that might change.
# ---------------------------------------------------------------------------
echo "=== Zenodo: FMD Cattle dataset (record 7779246) ==="
RECORD_ID="7779246"
META_JSON="$DEST/zenodo/_meta.json"
curl -sL "https://zenodo.org/api/records/${RECORD_ID}" -o "$META_JSON"

python3 - "$META_JSON" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for entry in data.get("files", []):
    print(f"{entry['key']}\t{entry['links']['self']}")
PYEOF
python3 - "$META_JSON" <<'PYEOF' > "$DEST/zenodo/_filelist.tsv"
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for entry in data.get("files", []):
    print(f"{entry['key']}\t{entry['links']['self']}")
PYEOF

while IFS=$'\t' read -r name url; do
    [ -z "$name" ] && continue
    echo "  downloading $name ..."
    curl -L --fail -o "$DEST/zenodo/$name" "$url"
done < "$DEST/zenodo/_filelist.tsv"

echo "  unzipping archives ..."
for z in "$DEST"/zenodo/*.zip; do
    [ -e "$z" ] || continue
    unzip -q -o "$z" -d "${z%.zip}"
done
echo "  done -> $DEST/zenodo"
echo ""

# ---------------------------------------------------------------------------
# 2. Kaggle — FMD Cattle Dataset + FMD Mouth Disease (150 images)
#    Requires a Kaggle API token: https://www.kaggle.com/settings -> "Create
#    New Token" -> save kaggle.json to ~/.kaggle/kaggle.json (chmod 600).
#    Skips cleanly if the kaggle CLI or token isn't set up.
# ---------------------------------------------------------------------------
echo "=== Kaggle datasets (optional) ==="
if ! command -v kaggle >/dev/null 2>&1; then
    echo "  kaggle CLI not installed — skipping. Install with: pip install kaggle"
elif [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "  ~/.kaggle/kaggle.json not found — skipping. See:"
    echo "  https://www.kaggle.com/settings -> API -> Create New Token"
else
    echo "  downloading wasimfaraz/fmd-cattle-dataset ..."
    kaggle datasets download -d wasimfaraz/fmd-cattle-dataset -p "$DEST/kaggle" --unzip

    echo "  downloading wasimfaraz/fmd-mouth-disease-in-cattle-dataset-150 ..."
    kaggle datasets download -d wasimfaraz/fmd-mouth-disease-in-cattle-dataset-150 -p "$DEST/kaggle" --unzip
fi

echo ""
echo "All done. Raw data is under: $DEST"
echo "Next: run crop_cows.py and check_leakage.py on $DEST before training."
