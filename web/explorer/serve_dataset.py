#!/usr/bin/env python3
"""Browse the whole SKY-Piano release in the explorer, on this computer.

Serves the standalone full-dataset explorer (web/explorer/dist-dataset) at / and a release under /dataset/:

  python web/explorer/serve_dataset.py --data PATH        # a downloaded copy of the Hugging Face dataset
                                                          #   (PATH holds metadata/ and data/; its CSV
                                                          #   files are .csv.gz and are served unpacked)
  python web/explorer/serve_dataset.py --manifest Data/Release/v0.3.9/manifest.csv
                                                          # a release built by pipeline/12_release, before upload
then open the printed address.

Build it once first: `cd web/explorer && npm install && npm run build:dataset`. This is its own app,
separate from the public explorer under the project page: it opens on the dataset list and links nowhere else.

The dataset routes, all read-only:
  /dataset/index.json            the metadata tables (performances, takes, participants, repertoire, pieces) as JSON
  /dataset/<release path>        a released file, with HTTP Range support for audio and video
  .../hand.csv?every=4           any motion CSV with only every 4th frame (120 Hz -> 30 Hz for the browser)
  .../body_segments.csv?src=body_imputed&every=4
                                 the 13-segment body the motion viewer draws, computed from body.csv or
                                 body_imputed.csv with pipeline/09_visual3d (cached under the system temp folder)
Only files of the release are served; nothing is written next to the data.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import mimetypes
import re
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DIST = HERE / "dist-dataset"         # npm run build:dataset (base "/")
PUBLIC = HERE / "public"             # keyboard.png for the motion viewer
BASE = "/"
TABLES = ("performances", "takes", "participants", "repertoire", "pieces")
CACHE = Path(tempfile.gettempdir()) / "skypiano_explorer_cache"
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("audio/wav", ".wav")
mimetypes.add_type("audio/midi", ".mid")


class Release:
    """Release path -> file on disk, from a downloaded dataset folder or a Stage 12 manifest."""

    def __init__(self, data=None, manifest=None):
        self.files, self.tables = {}, {}
        if data:
            root = Path(data).resolve()
            meta = root / "metadata" if (root / "metadata").is_dir() else root
            for t in TABLES:
                if (meta / f"{t}.csv").exists():
                    self.tables[t] = meta / f"{t}.csv"
            self.root = root
        else:
            m = Path(manifest).resolve()
            for r in csv.DictReader(open(m, encoding="utf-8")):
                src = Path(r["source_path"])
                self.files[r["release_path"]] = src if src.is_absolute() else REPO / src
            for t in TABLES:
                if (m.parent / f"{t}.csv").exists():
                    self.tables[t] = m.parent / f"{t}.csv"
            self.root = None
        if "performances" not in self.tables:
            sys.exit("no performances.csv found: pass the dataset folder (with metadata/) or a release manifest.csv")

    def path(self, rel):
        rel = rel.strip("/")
        if ".." in Path(rel).parts:
            return None
        if self.root is not None:
            p = self.root / rel
            if not p.is_file() and p.suffix == ".csv":      # published as .csv.gz from 0.3.12 on
                p = p.with_name(p.name + ".gz")
            return p if p.is_file() else None
        p = self.files.get(rel)
        return p if p is not None and p.is_file() else None

    def index(self):
        out = {}
        for t, p in self.tables.items():
            with open(p, encoding="utf-8") as f:
                out[t] = list(csv.DictReader(f))
        return out


def csv_text(f):
    """A released CSV as text, unpacked when it is a .csv.gz."""
    return gzip.decompress(f.read_bytes()).decode("utf-8") if f.suffix == ".gz" else f.read_text(encoding="utf-8")


def every_nth(text, n):
    if n <= 1:
        return text
    lines = text.splitlines()
    return "\n".join(lines[:1] + lines[1::n]) + "\n"


_seg_lock = threading.Lock()


def body_segments(src):
    """13-segment body CSV for one body marker CSV (pipeline/09_visual3d), cached by file identity."""
    st = src.stat()
    key = hashlib.sha1(f"{src}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]
    out = CACHE / f"{key}.csv"
    if not out.exists():
        with _seg_lock:
            if not out.exists():
                sys.path.insert(0, str(REPO / "pipeline" / "09_visual3d" / "steps"))
                from s01_segment_body_csv import write_body_segments_csv
                CACHE.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".part")
                if src.suffix == ".gz":
                    plain = CACHE / f"{key}_in.csv"
                    plain.write_text(csv_text(src), encoding="utf-8")
                    src = plain
                write_body_segments_csv(str(src), str(tmp), with_centers=False)
                if src.name.endswith("_in.csv"):
                    src.unlink()
                tmp.replace(out)
    return out.read_text(encoding="utf-8")


def make_handler(rel: Release):
    index_json = json.dumps(rel.index()).encode()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urlparse(self.path)
            p, q = unquote(u.path), parse_qs(u.query)
            sub = p[len(BASE):]
            if sub.startswith("dataset/"):
                return self.dataset(sub[len("dataset/"):], q)
            f = (DIST / sub).resolve()
            if sub and f.is_file() and DIST in f.parents:
                return self.file(f)
            if sub == "keyboard.png":
                return self.file(PUBLIC / sub)
            return self.file(DIST / "index.html")        # client-side routes

        def dataset(self, sub, q):
            if sub == "index.json":
                return self.send(200, index_json, "application/json")
            every = int(q.get("every", ["1"])[0] or 1)
            name = sub.rsplit("/", 1)[-1]
            if name == "body_segments.csv":
                src_name = q.get("src", ["body"])[0]
                if src_name not in ("body", "body_imputed"):
                    return self.send(400, b"src must be body or body_imputed", "text/plain")
                src = rel.path(sub.rsplit("/", 1)[0] + f"/{src_name}.csv")
                if src is None:
                    return self.send(404, b"not found", "text/plain")
                return self.send(200, every_nth(body_segments(src), every).encode(), "text/csv")
            f = rel.path(sub)
            if f is None:
                return self.send(404, b"not found", "text/plain")
            if f.suffix == ".gz" or (f.suffix == ".csv" and every > 1):
                return self.send(200, every_nth(csv_text(f), every).encode(), "text/csv")
            return self.file(f)

        def file(self, f):
            size = f.stat().st_size
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            s, e = 0, size - 1
            if rng and (rng[1] or rng[2]):
                s = int(rng[1]) if rng[1] else max(0, size - int(rng[2]))
                e = min(int(rng[2]) if rng[1] and rng[2] else size - 1, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(e - s + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {s}-{e}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(f, "rb") as fh:
                fh.seek(s)
                left = e - s + 1
                try:
                    while left > 0:
                        chunk = fh.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="downloaded dataset folder (holds metadata/ and data/)")
    g.add_argument("--manifest", help="Data/Release/v{X}/manifest.csv of a release built by pipeline/12_release")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    if not (DIST / "index.html").exists():
        sys.exit("web/explorer/dist-dataset is missing: run `npm install && npm run build:dataset` in web/explorer first")
    rel = Release(a.data, a.manifest)
    n = len(rel.index()["performances"])
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(rel))
    print(f"{n} performances; open http://{a.host}:{a.port}/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
