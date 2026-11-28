"""
Stage 12: send a release version to its Hugging Face dataset repository.

Reads the manifest that build_manifest.py wrote (Data/Release/v{version}/)
and uploads each listed file from where it lies in Data/ -- nothing is copied,
except that every CSV under data/ is published as <name>.csv.gz from the gzip
copies of gz_store.py (written there first when missing or stale).
Only files whose content differs from the repository's are sent; files the
repository holds that the version no longer lists are deleted. The version's
metadata tables (performances, participants, repertoire) and summary.json go
to metadata/, and a dataset card is written as README.md. The commit is tagged v{version}.

The repository is created private and gated (access requests approved by
hand, or automatically with --gated auto). Making it public is a separate,
deliberate step (--public), allowed only once the dataset license text exists
at Data/Release/LICENSE.txt (CC BY-NC-ND 4.0 with the Additional Permission for
Research Use, as approved; never edited here). It is uploaded as LICENSE.txt and
quoted in full in the card's License section.

Checks before anything is sent: every source file exists, no released path or
metadata row belongs to a session out of release scope
(paper_builders/participant_map.py), and the metadata is plain ASCII.

Content hashes are cached in Data/Release/_sha256_cache.json (keyed by path,
size and modification time), so a later version hashes only changed files.

The token is HF_TOKEN from the environment, or from .env at the repository
root (git-ignored), or the `hf auth login` credentials.

Usage (from the repository root):
  python pipeline/12_release/upload_hf.py --version 0.1.0 --repo OWNER/NAME [--dry-run]
  python pipeline/12_release/upload_hf.py --version 0.1.0 --repo OWNER/NAME --public
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_builders"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from participant_map import EXCLUDED_FROM_RELEASE  # noqa: E402
from gz_store import ensure as gz_ensure, published_path  # noqa: E402

REL = Path("Data/Release")
LICENSE = REL / "LICENSE.txt"
SHA_CACHE = REL / "_sha256_cache.json"
KEEP_REMOTE = {".gitattributes"}
BATCH = 400                      # file operations per commit

BIBTEX = """@inproceedings{bae2026skypiano,
  title     = {{SKY-Piano}: A Multimodal Piano Performance Dataset},
  author    = {Bae, Joonhyung and Park, Dawon and Kwon, Taegyun and
               Choi, Yoon-Seok and Hur, Hyeon and Obata, Satoshi and
               Kai, Shigeru and Wada, Yohei and Takahashi, Yu and
               Maezawa, Akira and Park, Jaebum and Park, Jonghwa and
               Nam, Juhan},
  booktitle = {Proceedings of the 27th International Society for Music
               Information Retrieval Conference (ISMIR)},
  address   = {Abu Dhabi, UAE},
  year      = {2026}
}"""


def load_env() -> None:
    env = ROOT / ".env"
    if "HF_TOKEN" in os.environ or not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()


def sha256(path: Path, cache: dict) -> str:
    st = path.stat()
    key = f"{path.as_posix()}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        cache[key] = h.hexdigest()
    return cache[key]


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


META_TABLES = ("performances.csv", "participants.csv", "repertoire.csv")
TAKE_TABLES = ("takes.csv", "pieces.csv")      # versions with take folders (0.3.0 on)
TASKS = ("technique1", "technique2", "technique_extra", "graded", "free")


def card(version: str, summary: dict, perf: list[dict], license_text: str | None, n_bytes: int) -> str:
    terms = license_text.strip() if license_text else "The license is not yet attached to this version."
    rows = "\n".join(
        f"| {t} | {len({r['participant_id'] for r in perf if r['task'] == t})} "
        f"| {sum(r['task'] == t for r in perf)} |" for t in TASKS if any(r["task"] == t for r in perf))
    return f"""---
pretty_name: SKY-Piano
license: cc-by-nc-nd-4.0
tags:
- music
- piano
- motion-capture
- midi
extra_gated_prompt: >-
  SKY-Piano holds recordings of real performers made under informed consent.
  Access is granted under its license: CC BY-NC-ND 4.0 with the Additional
  Permission for Research Use (see the License section and LICENSE.txt).
  Access requests are reviewed every Monday.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I agree to use the dataset under its license terms: checkbox
---

# SKY-Piano (version {version})

A multimodal piano performance dataset: audio, MIDI, hand and body motion
capture and per-note fingering, synchronized per performance. This is a pre-1.0
release; later versions add Visual3D kinematics, score alignment and
face-anonymized video. Each version is a tag of this repository (`v0.1.0`,
`v0.1.1`, ...).

## Access

Access is gated: request it on this page and agree to the license terms.
Requests are reviewed every Monday.

## Contents of {version}

| Task | Participants | Performances |
|---|---:|---:|
{rows}

{summary['files']} files, {n_bytes / 1e9:.1f} GB. Every CSV file under `data/` is
gzip-compressed (`.csv.gz`); pandas reads it as is (`pd.read_csv("hand.csv.gz")`),
and so does Python's `gzip` module or `gunzip` on the command line.

Tasks: `technique1` the numbered exercises of Fink's *Mastering Piano
Technique*; `technique2` the drills (scale, arpeggio, octave repeat, octave
scale, broken octave arpeggio; moderate and fast) and their applied pieces;
`technique_extra` a scale and arpeggio programme recorded with the
professional cohort; `graded` the Set 1 graded pieces; `free` free repertoire.

## Layout

```
data/P<id>/<task>/<piece>/r<n>/audio.wav    audio
                               midi.mid     MIDI
                               hand.csv.gz  hand motion, 120 fps
                               body.csv.gz  body joints (Visual3D-style), 120 fps, unfilled
                               fingering.csv.gz  per-note hand and finger, from the raw motion
                               fingering_human.csv.gz  per-note hand and finger labeled by hand (three performances)
                               hand_suspect_mask.csv.gz  1 = observed but implausible (ghost or swapped marker)
                               hand_imputed.csv.gz       hand motion with short gaps filled (where listed)
                               hand_imputed_mask.csv.gz  1 = value filled by the model
                               fingering_imputed.csv.gz  fingering labeled from hand_imputed.csv
                               body_imputed.csv.gz       body joints, marker gaps filled (lower body <= 5 s, upper <= 1 s)
                               body_imputed_mask.csv.gz  1 = value present only because of the filling
                               body_suspect_mask.csv.gz  1 = body.csv value uses a marker judged an outlier
                               top.mp4      overhead video of keyboard and hands (no face in view, not masked; where available)
                               left.mp4, right.mp4, right_low.mp4  side views, faces masked (where available)
data/P<id>/takes/<date>_<take>/audio.wav   every recorded take, whole (see below)
                               midi.mid, hand.csv.gz, body.csv.gz (and the body fill files)
metadata/performances.csv   one row per performance: path, participant, task,
                            piece, round, recording session and source take,
                            window in the take, first onset and release
metadata/takes.csv          one row per take: path, participant, session, source
                            take, modalities, durations, MIDI notes, muted audio
metadata/pieces.csv         one row per piece window in a take: start and end
                            in the take, piece, task, status (confirmed, retake,
                            practice variant, not a piece, rejected), best = the
                            performance released in its piece folder
metadata/participants.csv   participant id (1-19), group, sessions
metadata/repertoire.csv     piece id, task, title
metadata/summary.json       counts per session, and what was left out and why
```

`<id>` is the participant (1-19); `r<n>` numbers a participant's performances
of a piece in play order. The piece folders hold one representative
performance per participant, piece and round; the take folders hold every
recorded take whole, on the same clock as the piece cuts, and `pieces.csv`
locates every piece window in them, with `best` marking the representative one.
In the takes' audio, every stretch where the MIDI shows no key or sustain pedal
sounding (from 0.1 s before a note to 0.5 s after it stops sounding) is muted,
so talk between pieces is removed; a take recorded without MIDI keeps its audio
as recorded (`audio_muted` in `takes.csv`). Two amateur participants were
recorded without audio and MIDI; their takes carry motion only. Frame 0 of the motion files is time 0 of the audio and MIDI. Motion samples
the capture system lost are left empty in this version. `fingering.csv` has one
row per note of `midi.mid` (onset, offset in seconds from the file's start,
pitch, velocity, hand L/R, finger 1-5, tier A geometric / B press-depth refined
/ C learned fill, ambiguity flag, candidates, confidence). For two participants
of the 2022 amateur session (P10, P11) the fingertip markers are missing in most
recordings; their fingering is almost entirely Tier C (model-filled, ambiguity
= no_candidate) and should be treated as a symbolic prior rather than a
measurement. `fingering_human.csv` holds the fingering of the paper's audit,
labeled note by note from the video and motion, for three graded performances
(P04 `clementi_op36_no1`, P06 `schubert_impromptu_op90_no4`, P07
`mozart_k545_1`, each `r1`): the same notes in the same order as
`fingering.csv`, with hand L/R and finger 1-5. `hand_suspect_mask.csv`
has the columns of `hand.csv`; 1 marks an observed value that is implausible --
more than 150 mm from the median of the same hand's markers in that frame, or a
one-frame excursion of more than 50 mm out and back -- typically a ghost
reflection or a swapped marker label. Those values are kept in `hand.csv` as
recorded; drop them where 1.

`hand_imputed.csv` (professional cohort only, the pianists the gap-filling
model was trained on) is `hand.csv` with short gaps filled by a learned
imputation model; observed values are unchanged and `hand_imputed_mask.csv`
marks every filled value. To stay precise, only gaps of at most 29 frames
(0.24 s) are filled, and a fill is left empty when the frame has fewer than 5
observed hand markers, when it breaks the recording's bone lengths by more
than 5 mm / 10%, or when it moves faster than that marker ever does in the
recording; markers never observed in a recording, implausible observations
(`hand_suspect_mask.csv`), and the arm and keyboard reference markers are not
filled. On recordings held out from training, masked gaps of 10-29 frames are
filled with a mean error of about 1.4 mm (1.3% of values off by more than
10 mm).

`body.csv` holds body joints in the Visual3D segment convention (mm, columns
`<joint>_x|y|z`): head_top, head_base (between the two lateral head markers),
head_front (forehead marker, anterior reference), neck (between the
acromions), pelvis (centre of the four iliac spine markers), and per side
(`r_`, `l_`) shoulder, elbow, wrist, hand, hip, knee, ankle and foot. A joint
is computed only from a full set of its markers, so it is empty in frames
where one of them was lost; hand joints exist only where the body suit carried
a hand marker (2022 sessions). `body_suspect_mask.csv` marks joint values that
use a marker sample judged an outlier (a frozen or jumping marker, or one that
breaks the performer's rigid marker distances). `body_imputed.csv` holds the
same joints after those samples are removed and marker gaps are filled with
training-free methods (spline, rigid-body fill from the markers of the same
segment, with the hand capture's wrist, elbow and shoulder markers helping the
arm), up to 5 s for the lower body and 1 s for the upper body;
`body_imputed_mask.csv` marks values that exist only because of the filling.
On held-out gaps the filled joints are off by 1.4 / 3.7 / 9.6 mm on average
for 0.5 / 1 / 2 s gaps. Take folders other than the 2024 free-repertoire takes
carry the unfilled `body.csv` only.

## License

The SKY-Piano dataset is released under CC BY-NC-ND 4.0
(https://creativecommons.org/licenses/by-nc-nd/4.0/) with the following
Additional Permission for Research Use (also in LICENSE.txt):

```
{terms}
```

## Citation

```bibtex
{BIBTEX}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--repo", required=True, help="OWNER/NAME of the dataset repository")
    ap.add_argument("--gated", choices=["manual", "auto"], default="manual")
    ap.add_argument("--public", action="store_true", help="make the repository public (needs the license text)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    vdir = REL / f"v{args.version}"
    manifest = list(csv.DictReader(open(vdir / "manifest.csv", encoding="utf-8")))
    tables = {name: (vdir / name).read_bytes() for name in META_TABLES + TAKE_TABLES if (vdir / name).exists()}
    summary = json.loads((vdir / "summary.json").read_text(encoding="utf-8"))
    perf = list(csv.DictReader(tables["performances.csv"].decode("utf-8").splitlines()))

    # checks
    missing = [m["source_path"] for m in manifest if not Path(m["source_path"]).exists()]
    bad_scope = [r["path"] for r in perf if (r["session"], r["session_subject"]) in EXCLUDED_FROM_RELEASE
                 or not r["participant_id"]]
    takes = list(csv.DictReader(tables["takes.csv"].decode("utf-8").splitlines())) if "takes.csv" in tables else []
    bad_scope += [r["path"] for r in takes if (r["session"], r["session_subject"]) in EXCLUDED_FROM_RELEASE
                  or not r["participant_id"]]
    released = {m["release_path"].rsplit("/", 1)[0] for m in manifest}
    unlisted = sorted(released - {r["path"] for r in perf} - {r["path"] for r in takes})
    if bad_scope or missing or unlisted:
        sys.exit(f"out of scope: {bad_scope[:5]}; missing sources: {missing[:5]}; not in performances: {unlisted[:5]}")
    non_ascii = [n for n, b in tables.items() if not b.isascii()]
    if non_ascii:
        sys.exit(f"metadata is not plain ASCII: {non_ascii}")
    license_text = LICENSE.read_text(encoding="utf-8") if LICENSE.exists() else None
    if args.public and license_text is None:
        sys.exit(f"no license text at {LICENSE}; the repository stays private")

    files = gz_ensure(manifest)                          # release path -> file to publish
    n_bytes = sum(f.stat().st_size for f in files.values())
    local: dict[str, tuple[Path | bytes, str]] = {}     # published path -> (source, sha256 or git sha1)
    cache = json.loads(SHA_CACHE.read_text(encoding="utf-8")) if SHA_CACHE.exists() else {}
    for i, m in enumerate(manifest, 1):
        src = files[m["release_path"]]
        local[published_path(m["release_path"])] = (src, sha256(src, cache))
        if i % 500 == 0:
            print(f"  hashed {i}/{len(manifest)}")
            SHA_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    SHA_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    texts = {**{f"metadata/{n}": b for n, b in tables.items()},
             "metadata/summary.json": json.dumps(summary, indent=2).encode(),
             "README.md": card(args.version, summary, perf, license_text, n_bytes).encode()}
    if license_text is not None:
        texts["LICENSE.txt"] = LICENSE.read_bytes()
    for p, data in texts.items():
        local[p] = (data, git_blob_sha1(data))

    load_env()
    api = HfApi()
    if args.dry_run:
        print(f"dry run: {len(local)} files ({len(manifest)} data, {len(texts)} metadata) -> {args.repo}")
        return
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.update_repo_settings(args.repo, repo_type="dataset", gated=args.gated)

    remote = {f.path: f for f in api.list_repo_tree(args.repo, repo_type="dataset", recursive=True)
              if hasattr(f, "size")}

    def same(path: str, src: Path | bytes, digest: str) -> bool:
        f = remote.get(path)
        if f is None:
            return False
        lfs = getattr(f, "lfs", None)
        if lfs is not None:
            return lfs.sha256 == digest
        if f.blob_id == digest:                      # metadata texts carry their git blob id
            return True
        # a small file kept in git (not LFS): compare its git blob id when the sizes agree
        return isinstance(src, Path) and f.size == src.stat().st_size and f.blob_id == git_blob_sha1(src.read_bytes())

    adds = [CommitOperationAdd(p, src if isinstance(src, bytes) else str(src))
            for p, (src, digest) in local.items() if not same(p, src, digest)]
    deletes = [CommitOperationDelete(p) for p in remote if p not in local and p not in KEEP_REMOTE]
    print(f"{args.repo}: {len(adds)} to upload, {len(deletes)} to delete, {len(local) - len(adds)} unchanged")
    ops = deletes + adds
    for k in range(0, len(ops), BATCH):
        part = ops[k:k + BATCH]
        api.create_commit(args.repo, part, repo_type="dataset",
                          commit_message=f"v{args.version}: files {k + 1}-{k + len(part)} of {len(ops)}")
        print(f"  committed {k + len(part)}/{len(ops)}")
    api.create_tag(args.repo, tag=f"v{args.version}", repo_type="dataset", exist_ok=True)
    if args.public:
        api.update_repo_settings(args.repo, repo_type="dataset", private=False)
    print(f"tagged v{args.version}; {'public' if args.public else 'private'}, gated ({args.gated})")


if __name__ == "__main__":
    main()
