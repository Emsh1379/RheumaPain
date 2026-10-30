"""Download RheumaPain v3 from Mendeley Data and verify every file against manifest.json.

Usage: python data/download_data.py <destination_folder> [--only PATTERN]
The dataset is written to <destination_folder>/RheumaPain/...; set RP_DATA=<destination_folder> afterwards.
Already downloaded files with a matching checksum are skipped, so the script can be rerun after interruptions.
"""
import argparse
import hashlib
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(entry, root):
    dest = os.path.join(root, entry["path"])
    if os.path.exists(dest) and sha256(dest) == entry["sha256"]:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(entry["url"], headers={"User-Agent": "Mozilla/5.0 (rheumapain-benchmark)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
            while block := r.read(1 << 20):
                out.write(block)
    except Exception as exc:  # report and continue with the other files
        return f"FAIL {entry['path']}: {exc}"
    if entry["sha256"] and sha256(dest) != entry["sha256"]:
        return f"CHECKSUM MISMATCH {entry['path']}"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dest", help="folder that will contain RheumaPain/")
    ap.add_argument("--only", default="", help="download only paths containing this text (for testing)")
    args = ap.parse_args()
    with open(os.path.join(HERE, "manifest.json"), encoding="utf-8-sig") as f:
        manifest = [m for m in json.load(f) if args.only in m["path"]]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda m: fetch(m, args.dest), manifest))
    problems = [r for r in results if r not in ("ok", "skip")]
    print(f"{len(manifest)} files: ok={results.count('ok')} skip={results.count('skip')} problems={len(problems)}")
    for p in problems:
        print(p)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
