"""
Mocap-adapted fingering extraction for the released corpus.

Adapts a geometry-based scoring algorithm from prior fingering work
(originally designed for MediaPipe-estimated hand landmarks) to precise
21-marker-per-hand mocap. Removes the MediaPipe model-skeleton calibration
and z-depth-based floating-hand detection (both unnecessary with direct
3D mocap). Preserves the per-frame scoring formula (exact match = 1.0,
half-key tolerance quadratic falloff) and the per-note decision rule
(50% normal / 80% strong candidate; auto-label only when exactly one
strong candidate; multi- and no-candidate notes retained with an explicit
ambiguity flag instead of forcing a best-guess label). The formula and the
rule (midicomparison.py and decider.py in the prior work's code release)
and their three thresholds are carried over unchanged for cross-paper
consistency. The other video-only steps are dropped as well: landmark
detection (the fingertip markers are read directly), keyboard
lens-distortion correction (a rigid frame from two stationary markers
suffices) and hand-attribution tie-breaking (marker names carry L/R, so a
finger's hand is known).

Inputs. A hand-motion CSV at 120 Hz with flattened columns
{MARKER}_x, {MARKER}_y, {MARKER}_z, read by piano_utils.load_fingertip:
the ten fingertip markers LTF LIF LMF LRF LLF / RTF RIF RMF RRF RLF
(thumb, index, middle, ring, little) and the keyboard end markers, the
auxiliary LEnd1 (A0 side) / REnd1 (C8 side) and the primary LEnd / REnd.
A fingertip sample with any NaN coordinate contributes nothing. A MIDI
file already aligned to the motion timeline (time 0 = motion frame 0);
every note of every instrument is scored, in instrument order and then
the file's note order (not re-sorted).

Keyboard frame and keys: see KeyboardGeometry. A point is projected to
(u, w, z_rel): u along the keyboard from A0 to C8, w horizontal toward
the player, z_rel height above the white-key surface.

Pass 1 (geometry score, see _score_finger): for each of the ten fingers
the per-frame contributions over the note's frames
[round(onset*fps), round(offset*fps)) are summed and divided by the
note's frame count (at least 1). A finger whose ratio exceeds
NORMAL_CANDIDATE_THRESHOLD is a normal candidate, above
STRONG_CANDIDATE_THRESHOLD a strong one. No normal candidate ->
no_candidate; exactly one -> labeled with it; several with exactly one
strong -> labeled with the strong one; otherwise multi_candidate.

Z-onset enhancement: in mocap-based fingering, the fingertip
z-trajectory at MIDI onset is a direct press signal that the video-only
prior work cannot exploit. Two extra passes use this:

  Pass 2 (Z-discrimination on multi-candidate): when the geometry-based
  scoring returns multiple candidates, the candidate whose z is lowest in
  a small window around the onset is the actual presser if (a) its z is
  at or below the press threshold, and (b) its z is clearly lower than
  the runner-up (gap >= Z_DISCRIM_GAP_MM). This rescues notes the original
  algorithm flagged ambiguous.

  Pass 3 (Z-rescue on no-candidate): when no finger accumulated enough
  (u, w) score over the note duration (typical for short / fast notes),
  we relax the (u, w) constraint to a wider tolerance and require the
  z-press signal at onset. This rescues short notes whose press dip is
  unambiguous in z but doesn't survive the duration-averaged score.

Output of FingeringMocap.run(): fps, n_frames, n_notes, n_labeled,
n_multi_candidate, n_no_candidate (the three statuses partition the
notes), decision_paths (note count per decision path), keyboard_reference
("aux", "primary" or "primary_session", see FingeringMocap.load), params
(the parameter values used) and fingering, one entry per MIDI note in
the order above (NoteFingering):
  note_idx, onset, offset (s), pitch;
  status       labeled | multi_candidate | no_candidate;
  hand         "Left" | "Right", None when not labeled;
  finger       1 (thumb) .. 5 (little) within the hand, None when not labeled;
  score        duration ratio of the chosen finger (0.0 for a Pass 3
               rescue), None when not labeled;
  candidates   list of {hand, finger, finger_name, score, z_onset_mm}:
               the Pass 1 normal candidates by descending ratio, or, when
               Pass 3 found any, the rescued fingers by ascending
               z_onset_mm with score 0.0 and uw_dist_mm added; kept for
               every note, labeled or not, so a consumer can re-decide
               (with stricter thresholds) or add human disambiguation;
  decision_path  score_none | score_single | score_strong | score_multi
               (Pass 1), z_discrim_multi (Pass 2), z_rescue_single |
               z_rescue_with_gap | z_rescue_multi (Pass 3).
steps/s01_pseudo_fingering.py writes this, with stem and dataset, as one
JSON per piece; steps/s04_release_fingering.py uses it as Tiers A and B.

Known limitations. Short notes: a finger pressing 3 of 8 frames scores
0.375, below the normal threshold, so fast passages (scales, arpeggios,
trills) leave correct presses at no_candidate unless Pass 3 finds a clear
z dip at onset. Adjacent fingers that both stay within the half-key
tolerance of the key end up multi_candidate unless Pass 2 sees a z gap
of at least Z_DISCRIM_GAP_MM. There is no corpus-specific precision
figure; the prior work reports 95.6% precision on a hand-annotated
subset (150 notes x 10 pieces) of its own dataset. The keyboard frame is
one rigid frame per piece from marker medians, which assumes the piano
does not move during the piece (true for these recordings).

References: prior fingering work (reference omitted for review).
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from piano_utils import load_fingertip


# ---------------------------------------------------------------------------
# Parameters — inherited from the reference fingering scorer for cross-paper consistency
# ---------------------------------------------------------------------------
WHITE_KEY_WIDTH_MM = 23.5
WHITE_KEY_DEPTH_MM = 150.0
BLACK_KEY_WIDTH_MM = 13.7
BLACK_KEY_DEPTH_MM = 90.0
BLACK_KEY_HEIGHT_MM = 15.0        # black key top above white key surface

N_WHITE_KEYS = 52                 # standard 88-key piano

HALF_KEY_TOLERANCE_MM = WHITE_KEY_WIDTH_MM / 2        # 11.75 mm
NORMAL_CANDIDATE_THRESHOLD = 0.5
STRONG_CANDIDATE_THRESHOLD = 0.8

# Physical span of the 88-key layout: from A0 center to C8 center.
PHYSICAL_KEY_SPAN_MM = (N_WHITE_KEYS - 1) * WHITE_KEY_WIDTH_MM  # 1198.5

# LEnd1 and REnd1 markers sit at the physical ends of the keyboard case,
# slightly outside the A0 and C8 key centers. The surplus distance between
# the markers and the outermost key centers is split evenly (assumed
# symmetric); the exact per-session margin is inferred from
# (measured_marker_length - PHYSICAL_KEY_SPAN_MM) / 2.
# A fingertip more than this far above the top of the scored key is not
# pressing it and contributes nothing in that frame. Per fingertip and
# relative to the key top, so raised black keys are accounted for; chosen
# empirically from fingertip heights in the 240828 trials. Replaces the
# prior work's hand-level, z-depth-ratio floating-hand detection.
FLOATING_Z_THRESHOLD_MM = 30.0          # fingertip above key top → not pressing

# ---------------------------------------------------------------------------
# Z-onset enhancement parameters
# ---------------------------------------------------------------------------
# Window around MIDI onset (in 120 Hz frames) over which the fingertip
# z-minimum is taken; ±3 frames ≈ ±25 ms covers normal alignment residual
# and a typical key-press dip duration.
Z_ONSET_WINDOW_FRAMES = 3
# Fingertip z (mm above key surface) at which the marker is considered to
# be unambiguously pressing. Marker is on the dorsal side of the distal
# phalanx, so during a real press the marker sits at +5 to +15 mm; we use
# the lower bound to avoid false positives on hovering fingers.
Z_PRESS_MM = 5.0
# Required z gap between the lowest (presumed presser) and runner-up for
# z-discrimination to commit. With multiple fingers all near the surface,
# a small gap means the geometry alone can't disambiguate and we keep the
# multi-candidate flag.
Z_DISCRIM_GAP_MM = 5.0
# (u, w) tolerance multiplier used in the no-candidate rescue path. The
# duration-averaged geometry score may fail on very short notes even when
# the (u, w) trajectory passes through the key region briefly; we rescue
# such notes when the z-press is unambiguous and the closest (u, w)
# distance in the onset window is within this multiple of the half-key
# tolerance.
Z_RESCUE_DIST_MULT = 1.5

# The key rectangles (w in [0, depth]) were set up on the auxiliary end
# markers LEnd1/REnd1. Sessions without them (220819) build the frame on the
# primary keyboard-edge markers LEnd/REnd instead, which sit this far toward
# the player (+w): measured where both exist, 78.0 mm (sd 0.09) over every
# 240828 piece. The key rectangles of a primary-marker frame are shifted back
# by this much so they cover the same physical keys.
PRIMARY_END_W_OFFSET_MM = 78.0

MIDI_A0 = 21
MIDI_C8 = 108

# Finger index layout: 0..4 = Left thumb..little, 5..9 = Right thumb..little
# Finger *number* follows the standard convention (1 = thumb, 5 = little) per hand.
FINGER_NAMES = [
    "LTF", "LIF", "LMF", "LRF", "LLF",
    "RTF", "RIF", "RMF", "RRF", "RLF",
]
FINGER_NUMBERS = {
    "LTF": 1, "LIF": 2, "LMF": 3, "LRF": 4, "LLF": 5,
    "RTF": 1, "RIF": 2, "RMF": 3, "RRF": 4, "RLF": 5,
}
FINGER_HAND = {fn: ("Left" if fn.startswith("L") else "Right") for fn in FINGER_NAMES}

BLACK_KEY_PC = {1, 3, 6, 8, 10}  # pitch classes of black keys (C=0)


# ---------------------------------------------------------------------------
# Keyboard geometry
# ---------------------------------------------------------------------------
def _build_key_layout(u_A0: float):
    """
    Return per-MIDI u-position (in mm along keyboard axis, measured from
    LEnd1) and a black/white flag array.

    Args:
        u_A0: u-coordinate (mm) of the A0 key center; C8 center sits
              `PHYSICAL_KEY_SPAN_MM` further along the axis.
    """
    midis = list(range(MIDI_A0, MIDI_C8 + 1))
    is_black = [(m % 12) in BLACK_KEY_PC for m in midis]

    u_positions = np.zeros(len(midis))
    white_idx = 0
    for i, black in enumerate(is_black):
        if not black:
            u_positions[i] = u_A0 + white_idx * WHITE_KEY_WIDTH_MM
            white_idx += 1

    # Black keys are centered between the two neighboring white keys.
    for i, black in enumerate(is_black):
        if not black:
            continue
        prev_w = i - 1
        while prev_w >= 0 and is_black[prev_w]:
            prev_w -= 1
        next_w = i + 1
        while next_w < len(midis) and is_black[next_w]:
            next_w += 1
        if prev_w >= 0 and next_w < len(midis):
            u_positions[i] = (u_positions[prev_w] + u_positions[next_w]) / 2
        elif prev_w >= 0:
            u_positions[i] = u_positions[prev_w]
        else:
            u_positions[i] = u_positions[next_w]

    return u_positions, is_black


class KeyboardGeometry:
    """
    Piano keyboard coordinate system built from left/right end markers.

    Defines a 2D planar frame (u along A0→C8, w perpendicular in the
    horizontal plane) plus a world-z surface height, and stores the
    per-key rectangle used for in-region testing.

    With L and R the A0-side and C8-side end-marker positions (per-piece
    medians, see FingeringMocap.load):
      u_axis = (R - L) / |R - L|
      w_axis = normalize(u_axis x (0, 0, 1)), horizontal, from the end
               markers toward the player; world z stays the vertical.
      u = (p - L) . u_axis,  w = (p - L) . w_axis,
      z_rel = p_z - surface_z, surface_z = mean of the two markers' z.
    The markers sit slightly outside the A0 and C8 key centers; the excess
    of |R - L| over PHYSICAL_KEY_SPAN_MM (51 x 23.5 = 1198.5 mm) is split
    evenly, so the A0 key center is at u = margin_mm.
    """

    def __init__(self, l_end: np.ndarray, r_end: np.ndarray, key_w_offset_mm: float = 0.0):
        """
        Args:
            l_end: 3D median position of the LEnd (A0-side) marker.
            r_end: 3D median position of the REnd (C8-side) marker.
            key_w_offset_mm: w of the key fronts in this frame (0 for the
                auxiliary end markers, -PRIMARY_END_W_OFFSET_MM for the
                primary ones).
        """
        self.key_w_offset_mm = float(key_w_offset_mm)
        self.l_end = np.asarray(l_end, dtype=float).copy()
        self.r_end = np.asarray(r_end, dtype=float).copy()

        # LEnd/REnd markers sit at the keyboard case ends (slightly outside
        # the A0 and C8 key centers); they already lie on the key surface
        # plane in this dataset, so no vertical adjustment is needed.

        self.u_axis = self.r_end - self.l_end
        self.keyboard_length_mm = float(np.linalg.norm(self.u_axis))
        self.u_axis /= self.keyboard_length_mm

        # Horizontal direction perpendicular to u, pointing toward the player
        # (front of the keys). `cross(u_axis, world_z)` gives this orientation
        # for this corpus's hand-mocap convention (Z up, LEnd behind keys).
        world_z = np.array([0.0, 0.0, 1.0])
        self.w_axis = np.cross(self.u_axis, world_z)
        w_norm = float(np.linalg.norm(self.w_axis))
        if w_norm < 1e-6:
            raise ValueError("Keyboard u-axis is (nearly) parallel to world Z; "
                             "LEnd/REnd markers look suspect.")
        self.w_axis /= w_norm

        # Keyboard surface z = marker z level (markers are at key-surface height)
        self.surface_z = (self.l_end[2] + self.r_end[2]) / 2.0

        # Margin between LEnd1 marker and A0 key center (and similarly
        # between REnd1 and C8). Assumed symmetric.
        self.margin_mm = max(0.0,
            (self.keyboard_length_mm - PHYSICAL_KEY_SPAN_MM) / 2.0)

        self._build_key_rectangles()

    def _build_key_rectangles(self):
        """One (u, w) rectangle per key, MIDI 21 (A0) to 108 (C8).

        White keys: centers on a WHITE_KEY_WIDTH_MM grid from u = margin_mm,
        half-width 11.75 mm, depth 150 mm, top at z_rel 0. Black keys: center
        midway between the neighboring white keys, half-width 6.85 mm, depth
        90 mm, top at z_rel 15 mm. The rectangle spans w in
        [key_w_offset_mm, key_w_offset_mm + depth].
        """
        u_positions, is_black = _build_key_layout(u_A0=self.margin_mm)

        self.key_info = {}
        for i, midi in enumerate(range(MIDI_A0, MIDI_C8 + 1)):
            if is_black[i]:
                half_width = BLACK_KEY_WIDTH_MM / 2
                depth = BLACK_KEY_DEPTH_MM
                top_z = BLACK_KEY_HEIGHT_MM
            else:
                half_width = WHITE_KEY_WIDTH_MM / 2
                depth = WHITE_KEY_DEPTH_MM
                top_z = 0.0
            self.key_info[midi] = {
                "u_center": float(u_positions[i]),
                "half_width": half_width,
                "w_front": self.key_w_offset_mm,
                "w_back": self.key_w_offset_mm + depth,
                "is_black": is_black[i],
                "top_z_rel": top_z,   # z of the key's top surface, relative to surface_z
            }

    def project(self, point_3d: np.ndarray) -> tuple:
        """Project a 3D world point to keyboard-frame (u, w, z_rel).

        z_rel is the world-z of the point minus the keyboard surface z
        (positive = above the white-key surface).
        """
        rel = point_3d - self.l_end
        u = float(np.dot(rel, self.u_axis))
        w = float(np.dot(rel, self.w_axis))
        z_rel = float(point_3d[2] - self.surface_z)
        return u, w, z_rel

    def distance_to_key(self, u: float, w: float, midi: int) -> float:
        """Planar (u, w) distance in mm from the point to the key rectangle.
        Returns 0 when the point is inside the rectangle.
        """
        info = self.key_info.get(midi)
        if info is None:
            return float("inf")
        u_dist = max(0.0, abs(u - info["u_center"]) - info["half_width"])
        if w < info["w_front"]:
            w_dist = info["w_front"] - w
        elif w > info["w_back"]:
            w_dist = w - info["w_back"]
        else:
            w_dist = 0.0
        return float(np.hypot(u_dist, w_dist))


# ---------------------------------------------------------------------------
# Fingering extractor
# ---------------------------------------------------------------------------
@dataclass
class NoteFingering:
    note_idx: int
    onset: float
    offset: float
    pitch: int
    status: str              # "labeled" | "multi_candidate" | "no_candidate"
    hand: Optional[str]
    finger: Optional[int]
    score: Optional[float]
    candidates: list
    decision_path: str = "score_based"  # which logic branch produced the final status


class FingeringMocap:
    """Generate pseudo-fingering labels from 3D hand mocap + MIDI.

    Usage:
        fm = FingeringMocap(motion_csv, midi_file)
        fm.load()
        result = fm.run()
    """

    def __init__(self, motion_path: str, midi_path: str, fps: int = 120,
                 fallback_ends: Optional[tuple] = None):
        """
        Args:
            fallback_ends: (LEnd, REnd) primary-marker positions to use when
                the file has no end-marker samples at all (the session's
                positions from its other pieces; the piano does not move).
        """
        self.motion_path = str(motion_path)
        self.midi_path = str(midi_path)
        self.fps = fps
        self.fallback_ends = fallback_ends
        self.keyboard_reference = None

        self.motion = None
        self.keyboard = None
        self.midi_notes = None
        self.n_frames = None

    # -- loading -----------------------------------------------------------
    def load(self):
        """Read fingertips, end markers and MIDI notes; build the keyboard.

        The frame uses the per-piece nanmedian of the auxiliary LEnd1/REnd1
        markers when both have samples (keyboard_reference "aux"), else of
        the primary LEnd/REnd with key rectangles shifted by
        -PRIMARY_END_W_OFFSET_MM ("primary"); if those have no samples
        either, fallback_ends, also with the shifted rectangles
        ("primary_session"). ValueError when there is no fallback.
        """
        (Lend1, _Lend2, Rend1, _Rend2,
         rtf, rif, rmf, rrf, rlf,
         ltf, lif, lmf, lrf, llf) = load_fingertip(self.motion_path)

        self.motion = {
            "LTF": ltf, "LIF": lif, "LMF": lmf, "LRF": lrf, "LLF": llf,
            "RTF": rtf, "RIF": rif, "RMF": rmf, "RRF": rrf, "RLF": rlf,
        }
        self.n_frames = len(ltf)

        # load_fingertip substitutes the primary marker for an auxiliary one
        # that has no samples; use the auxiliary pair only when both are real.
        aux = not (np.array_equal(Lend1, _Lend2, equal_nan=True)
                   or np.array_equal(Rend1, _Rend2, equal_nan=True))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ends = [np.nanmedian(m, axis=0) for m in ((Lend1, Rend1) if aux else (_Lend2, _Rend2))]
        self.keyboard_reference = "aux" if aux else "primary"
        if any(np.isnan(e).any() for e in ends):
            if self.fallback_ends is None:
                raise ValueError(f"LEnd/REnd markers have no valid samples in {self.motion_path}")
            ends = [np.asarray(e, dtype=float) for e in self.fallback_ends]
            self.keyboard_reference = "primary_session"
        self.keyboard = KeyboardGeometry(
            ends[0], ends[1], key_w_offset_mm=0.0 if aux else -PRIMARY_END_W_OFFSET_MM)
        # Keyboard-frame (u, w, z_rel) of every fingertip sample, computed once;
        # a sample with any NaN coordinate is NaN in all three.
        self._uwz = {}
        for name, traj in self.motion.items():
            traj = np.asarray(traj, dtype=float)
            bad = np.isnan(traj).any(axis=1)
            rel = traj - self.keyboard.l_end
            u = rel @ self.keyboard.u_axis
            w = rel @ self.keyboard.w_axis
            z = traj[:, 2] - self.keyboard.surface_z
            u[bad] = w[bad] = z[bad] = np.nan
            self._uwz[name] = (u, w, z)

        midi = pretty_midi.PrettyMIDI(self.midi_path)
        self.midi_notes = [
            {"onset": float(n.start), "offset": float(n.end), "pitch": int(n.pitch)}
            for inst in midi.instruments for n in inst.notes
        ]

    # -- scoring -----------------------------------------------------------
    def _time_to_frame(self, t: float) -> int:
        return int(round(t * self.fps))

    def _key_distance(self, u: np.ndarray, w: np.ndarray, pitch: int) -> np.ndarray:
        """Vectorized KeyboardGeometry.distance_to_key over many samples."""
        info = self.keyboard.key_info.get(pitch)
        if info is None:
            return np.full_like(u, np.inf)
        u_dist = np.maximum(0.0, np.abs(u - info["u_center"]) - info["half_width"])
        w_dist = np.where(w < info["w_front"], info["w_front"] - w,
                          np.where(w > info["w_back"], w - info["w_back"], 0.0))
        return np.hypot(u_dist, w_dist)

    def _score_finger(self, finger_name: str, pitch: int,
                      start_frame: int, end_frame: int) -> float:
        """Accumulate the finger's contribution across the note's frames.

        With d the planar (u, w) distance from the fingertip to the key
        rectangle (0 inside) and tau = HALF_KEY_TOLERANCE_MM, a frame
        contributes 1 when d == 0, (1 - d / tau)^2 when 0 < d < tau, and 0
        otherwise. Frames with a NaN sample or with the fingertip more than
        FLOATING_Z_THRESHOLD_MM above the key top contribute 0.
        """
        key_top = self.keyboard.key_info[pitch]["top_z_rel"]
        floating_limit = key_top + FLOATING_Z_THRESHOLD_MM

        s0 = max(0, start_frame)
        s1 = min(self.n_frames, end_frame)
        if s1 <= s0:
            return 0.0

        u, w, z = (a[s0:s1] for a in self._uwz[finger_name])
        keep = ~np.isnan(z) & (z <= floating_limit)
        d = self._key_distance(u[keep], w[keep], pitch)
        contrib = np.where(d == 0.0, 1.0,
                           np.where(d < HALF_KEY_TOLERANCE_MM,
                                    (1.0 - d / HALF_KEY_TOLERANCE_MM) ** 2, 0.0))
        return float(sum(contrib.tolist()))   # frame order, as the per-frame loop summed

    def _z_min_at_onset(self, finger_name: str, onset_frame: int) -> Optional[float]:
        """Return the minimum fingertip z (mm above key surface) within the
        onset window, or None if no valid samples are present.

        The minimum captures the deepest point of any press dip occurring
        within ±Z_ONSET_WINDOW_FRAMES of the MIDI onset.
        """
        s0 = max(0, onset_frame - Z_ONSET_WINDOW_FRAMES)
        s1 = min(self.n_frames, onset_frame + Z_ONSET_WINDOW_FRAMES + 1)
        z = self._uwz[finger_name][2][s0:s1] if s1 > s0 else np.empty(0)
        z = z[~np.isnan(z)]
        return float(z.min()) if z.size else None

    def _uw_min_distance_at_onset(self, finger_name: str, pitch: int,
                                   onset_frame: int) -> Optional[float]:
        """Return the minimum (u, w) distance from the fingertip to the
        target key within the onset window, or None if no valid samples.
        Used by the no-candidate rescue path to relax the duration-averaged
        score when the press itself is geometrically plausible at onset.
        """
        s0 = max(0, onset_frame - Z_ONSET_WINDOW_FRAMES)
        s1 = min(self.n_frames, onset_frame + Z_ONSET_WINDOW_FRAMES + 1)
        if s1 <= s0:
            return None
        u, w, z = (a[s0:s1] for a in self._uwz[finger_name])
        keep = ~np.isnan(z)
        if not keep.any():
            return None
        return float(self._key_distance(u[keep], w[keep], pitch).min())

    def _decide_for_note(self, note: dict, note_idx: int) -> NoteFingering:
        start_frame = self._time_to_frame(note["onset"])
        end_frame = self._time_to_frame(note["offset"])
        total_frames = max(1, end_frame - start_frame)
        pitch = int(note["pitch"])

        # ---- Pass 1: geometry-based (u, w) scoring over note duration ----
        candidates = []
        for fname in FINGER_NAMES:
            raw = self._score_finger(fname, pitch, start_frame, end_frame)
            ratio = raw / total_frames
            z_min = self._z_min_at_onset(fname, start_frame)
            if ratio > NORMAL_CANDIDATE_THRESHOLD:
                candidates.append({
                    "hand": FINGER_HAND[fname],
                    "finger": FINGER_NUMBERS[fname],
                    "finger_name": fname,
                    "score": round(float(ratio), 4),
                    "z_onset_mm": round(float(z_min), 2) if z_min is not None else None,
                })

        candidates.sort(key=lambda c: -c["score"])
        strong = [c for c in candidates if c["score"] > STRONG_CANDIDATE_THRESHOLD]

        if len(candidates) == 0:
            status, chosen, decision_path = "no_candidate", None, "score_none"
        elif len(candidates) == 1:
            status, chosen, decision_path = "labeled", candidates[0], "score_single"
        elif len(strong) == 1:
            status, chosen, decision_path = "labeled", strong[0], "score_strong"
        else:
            status, chosen, decision_path = "multi_candidate", None, "score_multi"

        # ---- Pass 2: Z-discrimination on multi-candidate ----
        # If geometry-based scoring left the note ambiguous, the fingertip whose
        # z is lowest at onset is the actual presser provided (a) it is
        # below the press threshold and (b) its z gap to the runner-up is
        # large enough to be informative.
        if status == "multi_candidate":
            cands_with_z = [c for c in candidates if c["z_onset_mm"] is not None]
            if cands_with_z:
                cands_with_z.sort(key=lambda c: c["z_onset_mm"])
                winner = cands_with_z[0]
                runner = cands_with_z[1] if len(cands_with_z) > 1 else None
                if winner["z_onset_mm"] <= Z_PRESS_MM and (
                    runner is None
                    or runner["z_onset_mm"] - winner["z_onset_mm"] >= Z_DISCRIM_GAP_MM
                ):
                    status = "labeled"
                    chosen = winner
                    decision_path = "z_discrim_multi"

        # ---- Pass 3: Z-rescue on no-candidate ----
        # Short / fast notes can leave every finger below the 0.5 ratio
        # threshold while the actual presser still produces an unambiguous
        # press dip at onset. Relax the (u, w) distance constraint and
        # require the z-press signal to be present.
        if status == "no_candidate":
            rescued = []
            for fname in FINGER_NAMES:
                z_min = self._z_min_at_onset(fname, start_frame)
                if z_min is None or z_min > Z_PRESS_MM:
                    continue
                d_uw = self._uw_min_distance_at_onset(fname, pitch, start_frame)
                if d_uw is None or d_uw > Z_RESCUE_DIST_MULT * HALF_KEY_TOLERANCE_MM:
                    continue
                rescued.append({
                    "hand": FINGER_HAND[fname],
                    "finger": FINGER_NUMBERS[fname],
                    "finger_name": fname,
                    "score": 0.0,
                    "z_onset_mm": round(float(z_min), 2),
                    "uw_dist_mm": round(float(d_uw), 2),
                })

            if rescued:
                rescued.sort(key=lambda c: c["z_onset_mm"])
                if len(rescued) == 1:
                    status = "labeled"
                    chosen = rescued[0]
                    decision_path = "z_rescue_single"
                elif rescued[1]["z_onset_mm"] - rescued[0]["z_onset_mm"] >= Z_DISCRIM_GAP_MM:
                    status = "labeled"
                    chosen = rescued[0]
                    decision_path = "z_rescue_with_gap"
                else:
                    status = "multi_candidate"
                    decision_path = "z_rescue_multi"
                candidates = rescued

        return NoteFingering(
            note_idx=note_idx,
            onset=float(note["onset"]),
            offset=float(note["offset"]),
            pitch=pitch,
            status=status,
            hand=chosen["hand"] if chosen else None,
            finger=chosen["finger"] if chosen else None,
            score=chosen["score"] if chosen else None,
            candidates=candidates,
            decision_path=decision_path,
        )

    # -- orchestration -----------------------------------------------------
    def run(self) -> dict:
        if self.motion is None:
            self.load()

        per_note = [self._decide_for_note(n, i) for i, n in enumerate(self.midi_notes)]

        n_labeled = sum(1 for p in per_note if p.status == "labeled")
        n_multi = sum(1 for p in per_note if p.status == "multi_candidate")
        n_none = sum(1 for p in per_note if p.status == "no_candidate")

        path_counts: dict[str, int] = {}
        for p in per_note:
            path_counts[p.decision_path] = path_counts.get(p.decision_path, 0) + 1

        return {
            "fps": self.fps,
            "n_frames": self.n_frames,
            "n_notes": len(per_note),
            "n_labeled": n_labeled,
            "n_multi_candidate": n_multi,
            "n_no_candidate": n_none,
            "decision_paths": path_counts,
            "keyboard_reference": self.keyboard_reference,
            "params": {
                "half_key_tolerance_mm": HALF_KEY_TOLERANCE_MM,
                "normal_threshold": NORMAL_CANDIDATE_THRESHOLD,
                "strong_threshold": STRONG_CANDIDATE_THRESHOLD,
                "floating_z_threshold_mm": FLOATING_Z_THRESHOLD_MM,
                "z_onset_window_frames": Z_ONSET_WINDOW_FRAMES,
                "z_press_mm": Z_PRESS_MM,
                "z_discrim_gap_mm": Z_DISCRIM_GAP_MM,
                "z_rescue_dist_mult": Z_RESCUE_DIST_MULT,
                "key_w_offset_mm": self.keyboard.key_w_offset_mm,
                "white_key_width_mm": WHITE_KEY_WIDTH_MM,
                "black_key_width_mm": BLACK_KEY_WIDTH_MM,
                "reference": "Adapted from prior fingering work (reference omitted for review); z-onset enhancement adds Pass 2 (z-discrim multi) and Pass 3 (z-rescue none).",
            },
            "fingering": [asdict(p) for p in per_note],
        }


def process_trial(motion_path: str, midi_path: str, fps: int = 120) -> dict:
    """Convenience: run the full pipeline on a single trial."""
    fm = FingeringMocap(motion_path, midi_path, fps=fps)
    fm.load()
    return fm.run()
