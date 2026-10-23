#!/usr/bin/env bash
# Usage: scan.sh <repo> <tag> <asset>; downloads, scans, records, then removes its own work folder.
set -u
S="$(cd "$(dirname "$0")" && pwd)"; repo=$1; tag=$2; asset=$3
d="$S/work-$repo"
python -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$d"
mkdir -p "$d/x"
out="$S/$repo.txt"; : > "$out"
gh release download "$tag" -R "kurtvalcorza/$repo" -p "$asset" -D "$d" || { echo DOWNLOAD_FAIL >> "$out"; exit 1; }
{ echo "repo=$repo tag=$tag asset=$asset"; sha256sum "$d/$asset" | cut -d' ' -f1 | sed 's/^/sha256=/'
  echo "-- members"; unzip -l "$d/$asset" | tail -n +4 | head -n -2 | awk '{print $1, $4}'
} >> "$out"
unzip -q "$d/$asset" -d "$d/x"
find "$d/x" -type f | while read -r f; do
  if unzip -tq "$f" >/dev/null 2>&1 && [[ "$f" != *.zip ]]; then
    echo "-- nested zip: ${f#$d/x/}" >> "$out"; unzip -l "$f" | tail -n +4 | head -n -2 | awk '{print "   ", $1, $4}' >> "$out"
    mkdir -p "$f.x"; unzip -q "$f" -d "$f.x"
  fi
done
echo "-- file types (non-text)" >> "$out"
find "$d/x" -type f | while read -r f; do printf '%s: %s\n' "${f#$d/x/}" "$(file -b "$f" | cut -c1-60)"; done | grep -v -E "JSON|ASCII text|UTF-8" >> "$out"
echo "-- picklescan 1.0.5" >> "$out"
uvx -q picklescan==1.0.5 -g -p "$d/x" >> "$out" 2>&1; echo "picklescan_exit=$?" >> "$out"
mapfile -t jl < <(find "$d/x" -type f -name '*.joblib')
if [ ${#jl[@]} -gt 0 ]; then
  echo "-- restricted-unpickler global inventory (joblib_globals.py)" >> "$out"
  uv run -q --no-project --with joblib --with numpy python -I "$S/joblib_globals.py" "${jl[@]}" 2>&1 | sed "s#$d/x/##" >> "$out"
fi
python -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$d"
