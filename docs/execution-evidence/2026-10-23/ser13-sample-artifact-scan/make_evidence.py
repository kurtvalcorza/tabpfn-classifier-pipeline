"""Copy a scan.sh report into committed evidence, changing only local paths (to <scan-dir>).
Copies scan.sh always and joblib_globals.py only when the report contains its inventory.
Usage: make_evidence.py REPORT OUT_DIR"""
import pathlib
import re
import shutil
import sys

report, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
here = report.parent
text = report.read_text(encoding="utf-8")
text = re.sub(r"[A-Za-z]:[\\/][^\s:]*?work-[a-z-]+[\\/](x|nested)[\\/]", r"<scan-dir>/\1/", text)
assert "KURTVA" not in text and "Users" not in text, "local path left in evidence"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "scan-report.txt").write_bytes(text.encode("utf-8"))
tools = ["scan.sh", "make_evidence.py"]
if "-- restricted-unpickler global inventory" in text:
    tools.append("joblib_globals.py")
for tool in tools:
    shutil.copyfile(here / tool, out_dir / tool)
print(out_dir / "scan-report.txt", len(text.splitlines()), "lines;", ", ".join(tools))
