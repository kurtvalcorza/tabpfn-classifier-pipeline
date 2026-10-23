#!/usr/bin/env bash
# Usage: scan.sh <repo> <tag> <asset>
# Downloads one release asset, extracts it, scans every pickle payload once, and writes <repo>.txt.
# PyTorch zip checkpoints (.pt/.ckpt/.pth/.bin) are listed but not extracted: picklescan reads their
# embedded data.pkl itself. Other nested zips (for example model.tabpfn_fit) are extracted so their
# members are scanned. joblib files get a restricted-unpickler inventory (joblib_globals.py), because
# picklescan stops at the raw array bytes joblib writes between pickle instructions.
# Exit status: the highest status of picklescan and the inventory (0 = clean, 1 = dangerous, 2 = incomplete).
set -u
S="$(cd "$(dirname "$0")" && pwd)"; repo=$1; tag=$2; asset=$3
d="$S/work-$repo"
python -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$d"
mkdir -p "$d/x"
out="$S/$repo.txt"; : > "$out"
gh release download "$tag" -R "kurtvalcorza/$repo" -p "$asset" -D "$d" || { echo DOWNLOAD_FAIL >> "$out"; exit 3; }
{ echo "repo=$repo tag=$tag asset=$asset"; sha256sum "$d/$asset" | cut -d' ' -f1 | sed 's/^/sha256=/'
  echo "-- members"; unzip -l "$d/$asset" | tail -n +4 | head -n -2 | awk '{print $1, $4}'
} >> "$out"
unzip -q "$d/$asset" -d "$d/x"
find "$d/x" -type f | while read -r f; do
  if unzip -tq "$f" >/dev/null 2>&1; then
    echo "-- nested zip: ${f#$d/x/}" >> "$out"; unzip -l "$f" | tail -n +4 | head -n -2 | awk '{print "   ", $1, $4}' >> "$out"
    case "$f" in
      *.pt|*.ckpt|*.pth|*.bin) echo "   (PyTorch zip: scanned in place by picklescan, not extracted)" >> "$out" ;;
      *) mkdir -p "$f.x"; unzip -q "$f" -d "$f.x" ;;
    esac
  fi
done
echo "-- picklescan 1.0.5" >> "$out"
uvx -q picklescan==1.0.5 -g -p "$d/x" >> "$out" 2>&1; ps=$?; echo "picklescan_exit=$ps" >> "$out"
status=$ps
mapfile -t jl < <(find "$d/x" -type f -name '*.joblib')
if [ ${#jl[@]} -gt 0 ]; then
  echo "-- restricted-unpickler global inventory (joblib_globals.py)" >> "$out"
  mkdir -p "$d/nested"
  uv run -q --no-project --python 3.13 --with joblib==1.6.0 --with numpy==2.5.4 python -I "$S/joblib_globals.py" --dump-nested "$d/nested" "${jl[@]}" >> "$out" 2>&1
  inv=$?; echo "inventory_exit=$inv" >> "$out"
  [ $inv -gt 0 ] && [ $status -lt 2 ] && status=2
  if [ -n "$(ls -A "$d/nested")" ]; then
    echo "-- picklescan 1.0.5 of nested torch payloads" >> "$out"
    uvx -q picklescan==1.0.5 -g -p "$d/nested" >> "$out" 2>&1; np=$?; echo "nested_picklescan_exit=$np" >> "$out"
    [ $np -gt $status ] && status=$np
  fi
  # joblib files the inventory parsed to the end no longer count as an incomplete scan
  if [ $ps -eq 2 ] && [ $inv -eq 0 ] && ! grep -q "^   stopped" "$out"; then
    others=$(grep "^ERROR: parsing pickle" "$out" | grep -vc "\.joblib:")
    [ "$others" -eq 0 ] && [ $status -eq 2 ] && status=0
  fi
fi
echo "scan_status=$status" >> "$out"
python -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$d"
exit $status
