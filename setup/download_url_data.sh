#!/usr/bin/env bash
set -uo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/url"
mkdir -p "$DEST"
cd "$DEST"

fetch() {
    local name="$1" url="$2" out="$3"
    if wget -q -O "$out" "$url"; then
        echo "ok      $name"
    else
        rm -f "$out"
        echo "failed  $name ($url)"
    fi
}

fetch openphish https://openphish.com/feed.txt openphish.txt
fetch tranco https://tranco-list.eu/top-1m.csv.zip tranco.zip && [ -f tranco.zip ] && unzip -o -q tranco.zip && rm tranco.zip
fetch phishtank http://data.phishtank.com/data/online-valid.csv.bz2 phishtank.csv.bz2 && [ -f phishtank.csv.bz2 ] && bunzip2 -f phishtank.csv.bz2

ls -lh "$DEST"
