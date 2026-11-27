"""
Stage 12: the gzip copies of the released CSV files.

Every CSV under data/ is published gzip-compressed, as <name>.csv.gz (the
motion tables are about half as large, the masks about a hundredth). The
manifest and every pipeline step keep the plain CSV files; only the uploads
(upload_hf.py, upload_zenodo.py) publish the compressed copies, so the same
paths reach every host. Metadata tables stay plain CSV.

A copy is kept in Data/Release/_gz/ under the source's own path
(Data/Release/_gz/<source path>.gz) and rewritten only when the source changed
since (size or modification time, recorded in _gz/_index.json). Compression is
deterministic (no file name or time in the gzip header), so an unchanged CSV
always yields the same bytes and is not sent again.

Usage (from the repository root; the uploads call it themselves):
  python pipeline/12_release/gz_store.py --version 0.3.12 [--jobs 8]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REL = Path("Data/Release")
STORE = REL / "_gz"
INDEX = STORE / "_index.json"


def compressed(release_path: str) -> bool:
    """True for a released file that is published gzip-compressed."""
    return release_path.startswith("data/") and release_path.endswith(".csv")


def published_path(release_path: str) -> str:
    return release_path + ".gz" if compressed(release_path) else release_path


def store_path(source: Path) -> Path:
    return STORE / (source.as_posix().lstrip("/").replace(":", "") + ".gz")


def _stamp(source: Path) -> str:
    st = source.stat()
    return f"{st.st_size}|{st.st_mtime_ns}"


def _compress(task: tuple[str, str]) -> str:
    src, dst = map(Path, task)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as fi, open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
    os.replace(tmp, dst)
    return str(src)


def ensure(manifest: list[dict], jobs: int = 8) -> dict[str, Path]:
    """Release path -> file to publish: the gzip copy for a compressed path, else the source.
    Writes the copies that are missing or older than their source."""
    index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {}
    out, todo = {}, []
    for m in manifest:
        src = Path(m["source_path"])
        if not compressed(m["release_path"]):
            out[m["release_path"]] = src
            continue
        dst = store_path(src)
        out[m["release_path"]] = dst
        if index.get(src.as_posix()) != _stamp(src) or not dst.exists():
            todo.append((str(src), str(dst)))
    if todo:
        print(f"  compressing {len(todo)} CSV files ({jobs} jobs) -> {STORE}", flush=True)
        STORE.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(jobs) as ex:
            for i, src in enumerate(ex.map(_compress, todo, chunksize=4), 1):
                index[Path(src).as_posix()] = _stamp(Path(src))
                if i % 200 == 0 or i == len(todo):
                    print(f"  compressed {i}/{len(todo)}", flush=True)
                    INDEX.write_text(json.dumps(index), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    manifest = list(csv.DictReader(open(REL / f"v{a.version}" / "manifest.csv", encoding="utf-8")))
    files = ensure(manifest, a.jobs)
    raw = sum(int(m["bytes"]) for m in manifest)
    pub = sum(f.stat().st_size for f in files.values())
    print(f"v{a.version}: {len(files)} files, {raw / 1e9:.1f} GB as built, {pub / 1e9:.1f} GB published")


if __name__ == "__main__":
    main()
