"""
Stage 12: send a release version to Zenodo, as a restricted record.

Zenodo receives 1.0.0, the complete dataset; pre-1.0 versions are released on
Hugging Face only (upload_hf.py), where a version can be updated in place.

Zenodo holds at most 100 files per record, so the version's data goes up as
one zip per participant (SKY-Piano_v{version}_P{id}.zip, holding that
participant's data/P{id}/ tree as on Hugging Face), next to the metadata
tables, summary.json and LICENSE.txt. The zips are written under
Data/Release/v{version}/zenodo/ and removed once uploaded (--keep-zips keeps
them).

The record is CC BY-NC-ND 4.0 with the Additional Permission for Research Use
(Data/Release/LICENSE.txt, quoted verbatim in the description and the access
conditions). Access is restricted: the record and its description are public
once published, the files are shared on request.

Without --publish the record stays a draft with a reserved DOI, so the DOI
can be cited before release; --publish publishes the draft the version's
zenodo.json names. The record's id and DOI are kept in
Data/Release/v{version}/zenodo.json.

The token is ZENODO_TOKEN from the environment or from .env at the repository
root (git-ignored).

Usage (from the repository root):
  python pipeline/12_release/upload_zenodo.py --version 0.1.0 [--dry-run]
  python pipeline/12_release/upload_zenodo.py --version 0.1.0 --publish
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
REL = Path("Data/Release")
LICENSE = REL / "LICENSE.txt"
API = "https://zenodo.org/api"
META_TABLES = ("performances.csv", "participants.csv", "repertoire.csv")
STORED = {".wav", ".mid"}                     # already compact; csv files are deflated

TITLE = "SKY-Piano: A Multimodal Piano Performance Dataset"
CREATORS = ["Bae, Joonhyung", "Park, Dawon", "Kwon, Taegyun", "Choi, Yoon-Seok", "Hur, Hyeon",
            "Obata, Satoshi", "Kai, Shigeru", "Wada, Yohei", "Takahashi, Yu", "Maezawa, Akira",
            "Park, Jaebum", "Park, Jonghwa", "Nam, Juhan"]
LINKS = {"https://huggingface.co/datasets/SKY-Piano/SKY-Piano": "isAlternateIdentifier",
         "https://joonhyungbae.github.io/skypiano/": "isDocumentedBy",
         "https://github.com/joonhyungbae/skypiano": "isSupplementedBy"}
TASKS = ("technique1", "technique2", "technique_extra", "graded", "free")


def token() -> str:
    if "ZENODO_TOKEN" in os.environ:
        return os.environ["ZENODO_TOKEN"]
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ZENODO_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("no ZENODO_TOKEN")


def description(version: str, summary: dict, perf: list[dict], terms: str) -> str:
    rows = "".join(
        f"<tr><td>{t}</td><td>{len({r['participant_id'] for r in perf if r['task'] == t})}</td>"
        f"<td>{sum(r['task'] == t for r in perf)}</td></tr>" for t in TASKS if any(r["task"] == t for r in perf))
    return f"""<p>A multimodal piano performance dataset: audio, MIDI, and hand and body motion capture,
synchronized per performance. This is version {version}, a pre-1.0 release; later versions add imputed
motion, fingering labels, Visual3D kinematics, score alignment and face-anonymized video.</p>
<p>Files: one zip per participant (<code>SKY-Piano_v{version}_P&lt;id&gt;.zip</code>) holding
<code>data/P&lt;id&gt;/&lt;task&gt;/&lt;piece&gt;/r&lt;n&gt;/</code> with <code>audio.wav</code>,
<code>midi.mid</code>, <code>hand.csv</code> and <code>body.csv</code> (motion at 120 fps, frame 0 = time 0
of the audio and MIDI); free repertoire is given by whole take (<code>free/take&lt;nn&gt;/</code>).
<code>performances.csv</code>, <code>participants.csv</code> and <code>repertoire.csv</code> describe every
performance, participant (ids 1-19) and piece. The same release is on Hugging Face:
https://huggingface.co/datasets/SKY-Piano/SKY-Piano</p>
<table><tr><th>Task</th><th>Participants</th><th>Performances</th></tr>{rows}</table>
<p>{summary['performances']} performances, {summary['bytes'] / 1e9:.1f} GB.</p>
<h3>License</h3>
<p>CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/), with the following Additional
Permission for Research Use (also in LICENSE.txt):</p>
<pre>{html.escape(terms)}</pre>"""


def zips(version: str, manifest: list[dict], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    by: dict[str, list[dict]] = {}
    for m in manifest:
        by.setdefault(m["release_path"].split("/")[1], []).append(m)
    paths = []
    for pid, files in sorted(by.items()):
        z = out / f"SKY-Piano_v{version}_{pid}.zip"
        if not z.exists():
            tmp = z.with_suffix(".part")
            with zipfile.ZipFile(tmp, "w") as zf:
                for m in files:
                    kind = zipfile.ZIP_STORED if Path(m["release_path"]).suffix in STORED else zipfile.ZIP_DEFLATED
                    zf.write(m["source_path"], m["release_path"], compress_type=kind)
            tmp.rename(z)
        print(f"  {z.name}: {z.stat().st_size / 1e9:.2f} GB")
        paths.append(z)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vdir = REL / f"v{args.version}"
    state_file = vdir / "zenodo.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
    auth = {"access_token": token()}

    if args.publish:
        if not state.get("id"):
            sys.exit(f"no draft for v{args.version}; run without --publish first")
        r = requests.post(f"{API}/deposit/depositions/{state['id']}/actions/publish", params=auth, timeout=120)
        r.raise_for_status()
        state.update(doi=r.json().get("doi"), published=True)
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"published: https://doi.org/{state['doi']}")
        return

    manifest = list(csv.DictReader(open(vdir / "manifest.csv", encoding="utf-8")))
    perf = list(csv.DictReader(open(vdir / "performances.csv", encoding="utf-8")))
    summary = json.loads((vdir / "summary.json").read_text(encoding="utf-8"))
    terms = LICENSE.read_text(encoding="utf-8").strip()
    metadata = {
        "title": TITLE, "upload_type": "dataset", "version": args.version,
        "description": description(args.version, summary, perf, terms),
        "creators": [{"name": n} for n in CREATORS],
        "license": "cc-by-nc-nd-4.0",
        "access_right": "restricted",
        "access_conditions": "Access is granted under CC BY-NC-ND 4.0 with the Additional Permission for "
                             "Research Use below; the access-request process is being finalized.\n\n" + terms,
        "keywords": ["piano", "motion capture", "MIDI", "music performance", "multimodal"],
        "related_identifiers": [{"identifier": u, "relation": rel} for u, rel in LINKS.items()],
        "prereserve_doi": True,
    }
    if args.dry_run:
        print(json.dumps({k: v for k, v in metadata.items() if k != "description"}, indent=1)[:1500])
        print(f"dry run: {len({m['release_path'].split('/')[1] for m in manifest})} zips + "
              f"{len(META_TABLES) + 2} files")
        return

    if state.get("id"):
        dep = requests.get(f"{API}/deposit/depositions/{state['id']}", params=auth, timeout=60).json()
    else:
        r = requests.post(f"{API}/deposit/depositions", params=auth, json={}, timeout=60)
        r.raise_for_status()
        dep = r.json()
    r = requests.put(f"{API}/deposit/depositions/{dep['id']}", params=auth, json={"metadata": metadata}, timeout=60)
    if not r.ok:
        sys.exit(f"metadata rejected: {r.status_code} {r.text[:500]}")
    dep = r.json()
    doi = dep["metadata"].get("prereserve_doi", {}).get("doi")
    state.update(id=dep["id"], doi=doi, bucket=dep["links"]["bucket"], published=False)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"draft {dep['id']}, reserved DOI {doi}")

    have = {f["filename"]: f["filesize"] for f in dep.get("files", [])}
    uploads = zips(args.version, manifest, vdir / "zenodo")
    uploads += [vdir / n for n in META_TABLES] + [vdir / "summary.json", LICENSE]
    for p in uploads:
        if have.get(p.name) == p.stat().st_size:
            print(f"  {p.name}: already there")
            continue
        with open(p, "rb") as f:
            r = requests.put(f"{state['bucket']}/{p.name}", data=f, params=auth, timeout=None)
        r.raise_for_status()
        print(f"  uploaded {p.name}")
    if not args.keep_zips:
        for z in (vdir / "zenodo").glob("*.zip"):
            z.unlink()
    print(f"draft ready: https://zenodo.org/deposit/{dep['id']} (DOI {doi}, not published)")


if __name__ == "__main__":
    main()
