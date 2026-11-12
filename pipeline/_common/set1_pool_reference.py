"""
Reference profile for the 17-piece Set1 pool.

Each entry captures expected characteristics the automatic cluster profile can
match against:
  - key, mode:         key signature (C, G, F#, ...; major/minor)
  - tempo_range_bpm:   typical performed tempo band
  - duration_range:    typical full-piece duration (sec); subjects may play
                       excerpts so cluster duration can be SHORTER.
  - note_density_range: notes per second (approximate)
  - level:             Beginner/Intermediate/Advanced/Professional from CSV
  - title:             CSV title for curator readability

Source: CSV `Piece Selection - Set1.csv` + classical music knowledge
(standard piano repertoire). Values are conservative ranges for cross-
version robustness; not tight targets.
"""
SET1_POOL = [
    {
        "pool_id": "bach_minuet_g",
        "title": "Bach Minuet in G Major (BWV Anh.114)",
        "level": "Beginner",
        "key": "G", "mode": "major",
        "tempo_range_bpm": (100, 140),
        "duration_range_sec": (90, 180),    # full 2 sections + repeats
        "note_density_range": (6, 12),
    },
    {
        "pool_id": "clementi_op36_no1",
        "title": "Clementi Piano Sonata Op.36 No.1",
        "level": "Beginner",
        "key": "C", "mode": "major",
        "tempo_range_bpm": (100, 140),
        "duration_range_sec": (120, 240),   # 1st mvt ~3 min
        "note_density_range": (6, 10),
    },
    {
        "pool_id": "chopin_prelude_op28_no4",
        "title": "Chopin Prelude in E Minor, Op.28 No.4",
        "level": "Intermediate",
        "key": "E", "mode": "minor",
        "tempo_range_bpm": (40, 70),
        "duration_range_sec": (100, 180),
        "note_density_range": (3, 7),
    },
    {
        "pool_id": "beethoven_moonlight_1",
        "title": "Moonlight Sonata, 1st movement (Beethoven)",
        "level": "Intermediate",
        "key": "C#", "mode": "minor",
        "tempo_range_bpm": (40, 65),
        "duration_range_sec": (240, 400),
        "note_density_range": (5, 9),
    },
    {
        "pool_id": "mozart_k545_1",
        "title": "Mozart Sonata No.16 in C Major, K.545 (1st mvt)",
        "level": "Intermediate",
        "key": "C", "mode": "major",
        "tempo_range_bpm": (100, 140),
        "duration_range_sec": (180, 300),
        "note_density_range": (5, 10),
    },
    {
        "pool_id": "schumann_traumerei",
        "title": "Schumann: Traumerei (Kinderszenen Op.15 No.7)",
        "level": "Intermediate",
        "key": "F", "mode": "major",
        "tempo_range_bpm": (40, 65),
        "duration_range_sec": (120, 240),
        "note_density_range": (2, 5),
    },
    {
        "pool_id": "bach_prelude_c_wtc1",
        "title": "Bach Prelude No.1 in C Major (WTC I)",
        "level": "Intermediate",
        "key": "C", "mode": "major",
        "tempo_range_bpm": (60, 100),
        "duration_range_sec": (100, 200),
        "note_density_range": (5, 10),
    },
    {
        "pool_id": "mozart_twinkle_var",
        "title": "Mozart 12 Variations on \"Twinkle Twinkle\" (K.265)",
        "level": "Intermediate",
        "key": "C", "mode": "major",
        "tempo_range_bpm": (80, 160),
        "duration_range_sec": (120, 600),   # whole set or excerpts
        "note_density_range": (4, 12),
    },
    {
        "pool_id": "chopin_waltz_op64_no2",
        "title": "Chopin Waltz in C# Minor, Op.64 No.2",
        "level": "Advanced",
        "key": "C#", "mode": "minor",
        "tempo_range_bpm": (60, 160),       # agogic, varies widely
        "duration_range_sec": (180, 300),
        "note_density_range": (5, 12),
    },
    {
        "pool_id": "chopin_waltz_op18_eb",
        "title": "Chopin Waltz in E-Flat Major, Op.18",
        "level": "Advanced",
        "key": "D#", "mode": "major",        # Eb = D# enharmonic
        "tempo_range_bpm": (120, 200),
        "duration_range_sec": (240, 360),
        "note_density_range": (8, 16),
    },
    {
        "pool_id": "chopin_minute_waltz",
        "title": "Chopin Minute Waltz (Op.64 No.1)",
        "level": "Advanced",
        "key": "C#", "mode": "major",        # Db = C# enharmonic
        "tempo_range_bpm": (130, 180),
        "duration_range_sec": (90, 150),
        "note_density_range": (10, 16),
    },
    {
        "pool_id": "silvery_waves",
        "title": "Silvery Waves",
        "level": "Advanced",
        "key": "F", "mode": "major",
        "tempo_range_bpm": (80, 140),
        "duration_range_sec": (180, 300),
        "note_density_range": (4, 10),
    },
    {
        "pool_id": "schubert_impromptu_op90_no4",
        "title": "Schubert Impromptu, Op.90 No.4",
        "level": "Advanced",
        "key": "G#", "mode": "major",       # Ab = G# enharmonic
        "tempo_range_bpm": (90, 140),
        "duration_range_sec": (300, 500),
        "note_density_range": (4, 10),
    },
    {
        "pool_id": "chopin_raindrop",
        "title": "Chopin Prelude in D-Flat Major, Op.28 No.15 (Raindrop)",
        "level": "Advanced",
        "key": "C#", "mode": "major",        # Db = C# enharmonic
        "tempo_range_bpm": (60, 100),
        "duration_range_sec": (240, 360),
        "note_density_range": (4, 8),
    },
    {
        "pool_id": "chopin_nocturne_op9_no2",
        "title": "Chopin Nocturne in E-Flat Major, Op.9 No.2",
        "level": "Advanced",
        "key": "D#", "mode": "major",        # Eb = D#
        "tempo_range_bpm": (40, 80),
        "duration_range_sec": (240, 360),
        "note_density_range": (3, 8),
    },
    # Position 5 was historically labeled "Maiden's Wish (Chopin/Liszt)" but
    # the actual curriculum score file (05_Badarzewska_The_Maidens_Prayer.mxl)
    # and the recorded performance (240828 sub07 P3) are Bądarzewska's
    # "The Maiden's Prayer" Op.4 — a similarly-named but distinct
    # Beginner-tier salon piece. Corrected 2026-05-01 after sub07 confirmation.
    {
        "pool_id": "badarzewska_maidens_prayer",
        "title": "Bądarzewska — The Maiden's Prayer, Op.4",
        "level": "Beginner",
        "key": "Eb", "mode": "major",
        "tempo_range_bpm": (60, 100),
        "duration_range_sec": (240, 360),
        "note_density_range": (3, 8),
    },
]


# Key equivalence for enharmonic handling (set-based lookup)
ENHARMONIC_EQUIVALENTS = {
    "C": ["C"],         "C#": ["C#", "Db", "Bx"],
    "D": ["D"],         "D#": ["D#", "Eb"],
    "E": ["E", "Fb"],   "F": ["F", "E#"],
    "F#": ["F#", "Gb"], "G": ["G"],
    "G#": ["G#", "Ab"], "A": ["A"],
    "A#": ["A#", "Bb"], "B": ["B", "Cb"],
}
KEY_CANON = {}
for canon, group in ENHARMONIC_EQUIVALENTS.items():
    for k in group:
        KEY_CANON[k] = canon


def key_match(key_a: str, key_b: str) -> bool:
    return KEY_CANON.get(key_a, key_a) == KEY_CANON.get(key_b, key_b)
