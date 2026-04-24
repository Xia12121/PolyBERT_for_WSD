#!/usr/bin/env bash
# Download the Raganato et al. 2017 WSD evaluation framework
# (SemCor train + Senseval-2/3, SemEval-2007/2013/2015 test sets).
set -euo pipefail

DEST="${1:-data}"
mkdir -p "$DEST"
cd "$DEST"

URL="http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip"
ZIP="WSD_Evaluation_Framework.zip"

if [ ! -f "$ZIP" ]; then
    echo "Downloading $URL …"
    curl -L -o "$ZIP" "$URL"
fi

if [ ! -d "WSD_Evaluation_Framework" ]; then
    echo "Unzipping…"
    unzip -q "$ZIP"
fi

echo "Done. Layout:"
find WSD_Evaluation_Framework -maxdepth 3 -type d
