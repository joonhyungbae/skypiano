"""Sanitize Audiveris-exported MusicXML so partitura/OSMD can consume it.

Audiveris (5.10.2 tested) emits two patterns that downstream tools choke on:

1. **Mid-piece <divisions> changes** — Audiveris splits a piece into logical
   sections and resets divisions per section. partitura raises
   "Note array from parts with multiple divisions is not supported" and OSMD's
   Fraction layout misbehaves. Fix: rescale every <duration> to a single LCM
   divisions value used throughout.

2. **<octave-shift>** without consistent anchor → OSMD throws
   "Cannot read properties of undefined (reading 'realValue')" in
   `calculateSingleOctaveShift`. Strip the element entirely (visually loses
   the 8va marking, but preserves displayed pitches).

Also injects a default `<key><fifths>0</fifths></key>` when missing, which
some renderers require.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from functools import reduce
from math import gcd
from pathlib import Path


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _normalize_divisions(xml: str) -> str:
    div_values = [int(v) for v in re.findall(r"<divisions>(\d+)</divisions>", xml)]
    if not div_values:
        return xml
    target = reduce(_lcm, set(div_values))

    measure_re = re.compile(r"<measure\b[^>]*>.*?</measure>", re.DOTALL)
    cur_div = target
    out: list[str] = []
    last = 0
    for m in measure_re.finditer(xml):
        out.append(xml[last : m.start()])
        block = m.group(0)
        md = re.search(r"<divisions>(\d+)</divisions>", block)
        if md:
            cur_div = int(md.group(1))
        factor = target // cur_div
        if factor != 1:
            block = re.sub(
                r"<duration>(\d+)</duration>",
                lambda m: f"<duration>{int(m.group(1)) * factor}</duration>",
                block,
            )
        block = re.sub(r"<divisions>\d+</divisions>", f"<divisions>{target}</divisions>", block)
        out.append(block)
        last = m.end()
    out.append(xml[last:])
    return "".join(out)


def _strip_octave_shifts(xml: str) -> str:
    xml = re.sub(r"<octave-shift[^/]*/>", "", xml)
    xml = re.sub(r"<octave-shift\b[^>]*>.*?</octave-shift>", "", xml, flags=re.DOTALL)
    xml = re.sub(r"<direction-type>\s*</direction-type>", "", xml)
    xml = re.sub(r"<direction\b[^>]*>\s*<staff>\d+</staff>\s*</direction>", "", xml)
    xml = re.sub(r"<direction\b[^>]*>\s*</direction>", "", xml)
    return xml


def _inject_default_key(xml: str) -> str:
    if "<key>" in xml or "<key " in xml:
        return xml
    return re.sub(
        r"(<attributes>\s*<divisions>\d+</divisions>\s*)(<time)",
        r"\1<key><fifths>0</fifths></key>\n        \2",
        xml,
        count=1,
    )


def clean_xml(xml: str) -> str:
    xml = _normalize_divisions(xml)
    xml = _strip_octave_shifts(xml)
    xml = _inject_default_key(xml)
    return xml


def clean_mxl(src: Path, dst: Path) -> None:
    """Read .mxl, clean inner XML, repack as .mxl. dst may equal src."""
    src = Path(src)
    dst = Path(dst)
    tmp = src.with_suffix(".cleaning.tmp")
    if tmp.exists():
        shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()
    tmp.mkdir()
    try:
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(tmp)
        # Find the inner musicxml/xml file (not container.xml)
        candidates = [
            p for p in tmp.rglob("*")
            if p.is_file()
            and p.suffix.lower() in (".xml", ".musicxml")
            and "container" not in p.name.lower()
        ]
        if not candidates:
            raise RuntimeError(f"no inner musicxml in {src}")
        inner = candidates[0]
        rel_path = inner.relative_to(tmp).as_posix()
        cleaned = clean_xml(inner.read_text(encoding="utf-8"))
        inner.write_text(cleaned, encoding="utf-8")

        container_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<container>\n"
            "  <rootfiles>\n"
            f'    <rootfile full-path="{rel_path}" media-type="application/vnd.recordare.musicxml+xml"/>\n'
            "  </rootfiles>\n"
            "</container>\n"
        )

        # Repack
        out_tmp = dst.with_suffix(".tmp.mxl")
        with zipfile.ZipFile(out_tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(inner, rel_path)
            z.writestr("META-INF/container.xml", container_xml)
        os.replace(out_tmp, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?", help="default: overwrite src")
    args = ap.parse_args()
    dst = args.dst if args.dst else args.src
    clean_mxl(args.src, dst)
    print(f"cleaned: {args.src} -> {dst}")
