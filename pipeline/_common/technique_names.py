"""
Technique drill names in the words of the score.

The SET 2 drills of the technique book (Data/Score/technique2_explaination.pdf,
last page) are one run of scale, arpeggio, octave repeat (one bar), octave
scale and broken octave arpeggio, played "at a moderate tempo (120 BPM) and
at a fast tempo (160 BPM+)", then "the octave-repeat bar five times at the
fastest tempo". The session logs wrote these their own ways ("Scale (slow)",
"Scale_Slow", "Broken Arpeggio", "Repetition_Fast", "Octave Repeat 1"); score_name gives one
name for each:

    <drill> (moderate | fast)          slow -> moderate, as the score says
    Broken Octave Arpeggio (...)       the score's drill, logged "Broken Arpeggio"
    Octave Repeat (fastest, x5)        the numbered extra repeats

The octave-repeat bar has 8 chords; played five times over (at least
X5_MIN_CHORDS chords) its name gets ", x5" (with_repeats).
"""
from __future__ import annotations

import re

DRILLS = ("Scale", "Arpeggio", "Octave Repeat", "Octave Scale", "Broken Octave Arpeggio")
TEMPO_WORDS = {"slow": "moderate", "moderate": "moderate", "fast": "fast"}
X5_MIN_CHORDS = 20
EXTRA_REPEAT = "Octave Repeat (fastest, x5)"

_LOGGED = re.compile(r"^\s*(broken\s*(?:octave\s*)?arpeggio|octave\s*repeat|repetition|octave\s*scale|arpeggio|scale)"
                     r"\s*(?:\(\s*(slow|moderate|fast)\s*\)|_\s*(slow|fast))\s*$", re.I)
_EXTRA = re.compile(r"^\s*octave\s*repeat\s*\d+\s*$", re.I)


def score_name(name: str) -> str:
    """A logged SET 2 drill name in the words of the score; any other name as it is."""
    if _EXTRA.match(name):
        return EXTRA_REPEAT
    m = _LOGGED.match(name)
    if not m:
        return name
    key = re.sub(r"\s+", "", m.group(1).lower())
    drill = {"brokenarpeggio": "Broken Octave Arpeggio", "brokenoctavearpeggio": "Broken Octave Arpeggio",
             "octaverepeat": "Octave Repeat", "repetition": "Octave Repeat", "octavescale": "Octave Scale",
             "arpeggio": "Arpeggio",
             "scale": "Scale"}[key]
    return f"{drill} ({TEMPO_WORDS[(m.group(2) or m.group(3)).lower()]})"


def with_repeats(name: str, n_chords: int) -> str:
    """An octave repeat named by how many bars were played."""
    if name.startswith("Octave Repeat (") and "x5" not in name and n_chords >= X5_MIN_CHORDS:
        return name[:-1] + ", x5)"
    return name


def chord_count(onsets: list[float], within_s: float = 0.05) -> int:
    on = sorted(onsets)
    return sum(1 for k, o in enumerate(on) if k == 0 or o - on[k - 1] > within_s)
