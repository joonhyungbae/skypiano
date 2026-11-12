import { useState } from "react";
import {
  ArrowRight,
  Camera,
  Check,
  ClipboardCopy,
  Clock,
  Database,
  Download,
  FileText,
  Github,
  PlayCircle,
  Hand,
  Image as ImageIcon,
  Mic,
  Minus,
  Users,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";

/** Everything on this page is sourced from the ISMIR 2026 camera-ready paper.
    When a number changes there, change it here — no second source of truth. */

const EXPLORER_URL = `${import.meta.env.BASE_URL}explorer/`;
const CODE_URL = "https://github.com/joonhyungbae/skypiano";
const DATASET_URL = "https://huggingface.co/datasets/SKY-Piano/SKY-Piano";
const LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/";
const PAPER_URL = `${import.meta.env.BASE_URL}paper.pdf`;
const ISMIR_URL = "https://ismir2026program.ismir.net/poster_298.html";
const VIDEO_ID = "SCbQMOKsNRk";
const VIDEO_URL = `https://youtu.be/${VIDEO_ID}`;
const POSTER_URL = `${import.meta.env.BASE_URL}poster.png`;

const AUTHORS = [
  { name: "Joonhyung Bae", aff: 1 },
  { name: "Dawon Park", aff: 2 },
  { name: "Taegyun Kwon", aff: 1 },
  { name: "Yoon-Seok Choi", aff: 2 },
  { name: "Hyeon Hur", aff: 2 },
  { name: "Satoshi Obata", aff: 3 },
  { name: "Shigeru Kai", aff: 3 },
  { name: "Yohei Wada", aff: 3 },
  { name: "Yu Takahashi", aff: 3 },
  { name: "Akira Maezawa", aff: 3 },
  { name: "Jaebum Park", aff: 2 },
  { name: "Jonghwa Park", aff: 2 },
  { name: "Juhan Nam", aff: 1 },
];

const AFFILIATIONS = [
  "KAIST",
  "Seoul National University",
  "Yamaha Corporation",
];

const HERO_STATS = [
  { value: "11 h", label: "Synchronized performance" },
  { value: "19", label: "Pianists", sub: "7 professional · 12 amateur" },
  { value: "35", label: "Pieces", sub: "Technique, graded, free" },
  { value: "120 Hz", label: "Hand & body mocap", sub: "Two optical systems" },
];

/** Table 1 of the paper. `hand`/`body` are real motion capture, not estimated. */
const COMPARISON = [
  { name: "DAEMG", hand: false, body: true, video: "1-view", audio: true, midi: true, score: false, fingering: false, pianists: "2 / –", pieces: "1", hours: "7" },
  { name: "Solos", hand: false, body: false, video: "1-view", audio: true, midi: false, score: false, fingering: false, pianists: "unlabeled", pieces: "–", hours: "12" },
  { name: "MOSA", hand: false, body: true, video: "–", audio: true, midi: false, score: false, fingering: false, pianists: "8 / –", pieces: "10", hours: "15" },
  { name: "Für Elise", hand: false, body: false, video: "–", audio: true, midi: true, score: false, fingering: false, pianists: "15 / –", pieces: "153", hours: "10" },
  { name: "PianoMotion10M", hand: false, body: false, video: "1-view", audio: true, midi: false, score: false, fingering: false, pianists: "unlabeled", pieces: "–", hours: "116" },
  { name: "PianoKPM", hand: false, body: false, video: "–", audio: false, midi: false, score: false, fingering: false, pianists: "20 / –", pieces: "7", hours: "12.6" },
  { name: "PianoVAM", hand: false, body: false, video: "1-view", audio: true, midi: true, score: false, fingering: true, pianists: "– / 10", pieces: "106", hours: "21.0" },
];

const OURS_ROW = {
  name: "SKY-Piano (ours)",
  hand: true, body: true, video: "up to 4-view", audio: true, midi: true,
  score: true, fingering: true, pianists: "7 / 12", pieces: "35", hours: "11.0",
};

/** Table 3 of the paper. */
const CAPTURE = [
  {
    icon: Hand,
    title: "Hand motion",
    specs: ["OptiTrack, 8 IR cameras", "120 Hz, 23 markers per side (4 mm)", "Includes elbow and shoulder anchors"],
  },
  {
    icon: Users,
    title: "Body motion",
    specs: ["Qualisys, 8 IR cameras", "120 Hz, full-body marker suit", "17-joint Visual3D model"],
  },
  {
    icon: Camera,
    title: "Video",
    specs: ["4 synchronized cameras", "4K / 60 fps, HD downsampled", "Top, left, right, low-right"],
  },
  {
    icon: Mic,
    title: "Audio & MIDI",
    specs: ["Focusrite Scarlett, 48 kHz / 24-bit", "Yamaha Disklavier DC7X ENPRO", "Velocity, timing, pedals"],
  },
];

/** Table 2 of the paper. */
const TECHNIQUE_1 = [
  "Adduction/abduction",
  "Making short and long sounds",
  "Playing with strength in the upper arm",
  "Finger extension/flexion",
  "Press forward with force–rebound",
  "Bounce back hand",
];

const TECHNIQUE_2 = [
  "Scale",
  "Arpeggio",
  "Octave repetition",
  "Octave scale",
  "Broken octave arpeggio",
];

const GRADED = [
  {
    level: "Beginner",
    pieces: ["Bach Minuet in G Major (BWV Anh. 114)", "Clementi Piano Sonata Op. 36 No. 1"],
  },
  {
    level: "Intermediate",
    pieces: [
      "Chopin Prelude in E Minor Op. 28 No. 4",
      "Beethoven Moonlight Sonata (1st mvt)",
      "Mozart Sonata No. 16 K. 545",
      "Schumann Träumerei Op. 15 No. 7",
      "Bach Prelude No. 1 in C Major (WTC I)",
      "Mozart 12 Variations on “Twinkle Twinkle Little Star” K. 265",
    ],
  },
  {
    level: "Advanced",
    pieces: [
      "Chopin Waltz in C♯ Minor Op. 64 No. 2",
      "Chopin Waltz in E-flat Major Op. 18",
      "Chopin Minute Waltz Op. 64 No. 1",
      "Silvery Waves",
      "Schubert Impromptu Op. 90 No. 4",
      "Chopin Prelude in D-flat Major Op. 28 No. 15 (“Raindrop”)",
      "Chopin Nocturne in E-flat Major Op. 9 No. 2",
    ],
  },
];

const FREE_PIECES = [
  "Chopin Etude Op. 10 No. 4",
  "Chopin Etude Op. 10 No. 10",
  "Chopin Mazurka Op. 17 No. 4",
  "Chopin Sonata No. 2 (1st mvt)",
  "Bach Prelude and Fugue in C♯ Major (BWV 848)",
  "Ravel La Valse",
  "Ravel Jeux d’eau",
  "Ravel Gaspard de la Nuit (Ondine)",
  "Scriabin Sonata No. 2 (both mvts)",
];

/** Section 5 of the paper. */
const FINGERING_TIERS = [
  {
    tier: "Tier A",
    title: "Geometric scoring",
    desc: "Each fingertip accumulates a duration-averaged score inside the pressed key's 3D region, scored only on frames below the press depth.",
  },
  {
    tier: "Tier B",
    title: "z-onset refinement",
    desc: "Ambiguous notes are resolved from measured press depth in a ±25 ms onset window — evidence a video-only pipeline cannot access.",
  },
  {
    tier: "Tier C",
    title: "Learned imputation",
    desc: "A BiLSTM fills the residual notes, with the softmax masked to the geometrically plausible candidate set, closing coverage to 100%.",
  },
];

const BIBTEX = `@inproceedings{bae2026skypiano,
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
}`;

const NAV_SECTIONS = ["Abstract", "Presentation", "Explorer", "Dataset", "Repertoire", "Fingering", "Cite"];

const Yes = () => <Check className="mx-auto h-4 w-4 text-emerald-600" strokeWidth={2.5} />;
const No = () => <X className="mx-auto h-4 w-4 text-subtle/60" strokeWidth={2.5} />;
const Na = () => <Minus className="mx-auto h-4 w-4 text-subtle/50" strokeWidth={2.5} />;

const SectionHeading = ({
  eyebrow,
  title,
  lead,
}: {
  eyebrow: string;
  title: string;
  lead?: string;
}) => (
  <div className="mb-10 text-center">
    <p className="eyebrow mb-3">{eyebrow}</p>
    <h2 className="text-3xl font-bold tracking-tight text-foreground md:text-4xl">
      {title}
    </h2>
    {lead && (
      <p className="mx-auto mt-3 max-w-2xl text-base text-muted-foreground">
        {lead}
      </p>
    )}
  </div>
);

const Index = () => {
  const [copied, setCopied] = useState(false);

  const scrollToSection = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  const copyBibtex = async () => {
    await navigator.clipboard.writeText(BIBTEX);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="custom-scrollbar min-h-screen overflow-x-hidden bg-background">
      {/* Navigation */}
      <nav className="fixed inset-x-0 top-0 z-50 border-b border-border bg-card/85 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
          <button
            onClick={() => scrollToSection("hero")}
            className="text-base font-bold tracking-tight brand-gradient-text"
          >
            SKY-Piano
          </button>
          <div className="hidden items-center gap-1 md:flex">
            {NAV_SECTIONS.map((item) => (
              <button
                key={item}
                onClick={() => scrollToSection(item.toLowerCase())}
                className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-primary/5 hover:text-primary"
              >
                {item}
              </button>
            ))}
          </div>
          <a
            href={EXPLORER_URL}
            className="inline-flex items-center gap-1.5 rounded-md border border-border-strong px-3 py-1.5 text-sm font-semibold text-primary transition-colors hover:bg-primary/5"
          >
            Open explorer
            <ArrowRight className="h-3.5 w-3.5" />
          </a>
        </div>
      </nav>

      {/* Hero */}
      <section id="hero" className="px-4 pb-20 pt-32 md:pt-40">
        <div className="mx-auto max-w-4xl text-center">
          <a
            href={ISMIR_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="notice-pill mb-7 font-semibold hover:underline"
          >
            <span className="font-bold uppercase tracking-[0.06em] text-primary">
              ISMIR 2026
            </span>
            <span className="h-3 w-px bg-primary/25" />
            <span className="text-muted-foreground">Poster P13-8</span>
            <span className="h-3 w-px bg-primary/25" />
            <span className="text-muted-foreground">Abu Dhabi, UAE</span>
          </a>

          <h1 className="mb-4 text-5xl font-bold tracking-tight md:text-6xl">
            <span className="brand-gradient-text">SKY-Piano</span>
          </h1>
          <p className="mx-auto mb-8 max-w-2xl text-xl text-muted-foreground md:text-2xl">
            A Multimodal Piano Performance Dataset
          </p>

          {/* Authors */}
          <div className="mx-auto mb-2 flex max-w-3xl flex-wrap justify-center gap-x-3 gap-y-1 text-[15px] text-foreground">
            {AUTHORS.map((a) => (
              <span key={a.name}>
                {a.name}
                <sup className="ml-0.5 text-primary">{a.aff}</sup>
              </span>
            ))}
          </div>
          <p className="mb-9 text-sm text-muted-foreground">
            {AFFILIATIONS.map((aff, i) => (
              <span key={aff} className="mx-2 inline-block">
                <sup className="mr-0.5 text-primary">{i + 1}</sup>
                {aff}
              </span>
            ))}
          </p>

          {/* Actions */}
          <div className="mb-14 flex flex-wrap justify-center gap-2">
            <Button asChild className="h-10 gap-1.5 whitespace-nowrap px-3 text-sm">
              <a href={PAPER_URL} target="_blank" rel="noopener noreferrer">
                <FileText className="h-4 w-4" />
                Paper
              </a>
            </Button>
            <Button asChild className="h-10 gap-1.5 whitespace-nowrap px-3 text-sm">
              <a href={VIDEO_URL} target="_blank" rel="noopener noreferrer">
                <PlayCircle className="h-4 w-4" />
                Video
              </a>
            </Button>
            <Button asChild className="h-10 gap-1.5 whitespace-nowrap px-3 text-sm">
              <a href={POSTER_URL} target="_blank" rel="noopener noreferrer">
                <ImageIcon className="h-4 w-4" />
                Poster
              </a>
            </Button>
            <Button asChild className="h-10 gap-1.5 whitespace-nowrap px-3 text-sm">
              <a href={DATASET_URL} target="_blank" rel="noopener noreferrer">
                <Download className="h-4 w-4" />
                Hugging Face
              </a>
            </Button>
            <span title="The dataset will also be released on Zenodo with version 1.0.0.">
              <Button
                variant="outline"
                disabled
                className="h-10 gap-1.5 whitespace-nowrap border-border-strong px-3 text-sm"
              >
                <Download className="h-4 w-4" />
                Zenodo (with 1.0.0)
              </Button>
            </span>
            <Button asChild variant="outline" className="h-10 gap-1.5 whitespace-nowrap border-border-strong px-3 text-sm">
              <a href={CODE_URL} target="_blank" rel="noopener noreferrer">
                <Github className="h-4 w-4" />
                Code
              </a>
            </Button>
            <Button asChild variant="outline" className="h-10 gap-1.5 whitespace-nowrap border-border-strong px-3 text-sm">
              <a href={EXPLORER_URL}>
                <Database className="h-4 w-4" />
                Explore the data
              </a>
            </Button>
          </div>

          {/* Key stats */}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {HERO_STATS.map((s) => (
              <div key={s.label} className="surface-card p-5 text-center">
                <div className="mb-1 text-3xl font-bold text-primary">{s.value}</div>
                <div className="text-sm font-medium text-foreground">{s.label}</div>
                {s.sub && <div className="mt-0.5 text-xs text-subtle">{s.sub}</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Abstract */}
      <section id="abstract" className="px-4 py-20">
        <div className="mx-auto max-w-3xl">
          <SectionHeading eyebrow="Overview" title="Abstract" />
          <div className="surface-card space-y-5 p-8 text-[15px] leading-relaxed text-muted-foreground md:p-10">
            <p>
              Music information retrieval research on piano performance increasingly
              involves diverse modalities of data and annotations beyond audio and MIDI.
              We present <strong className="font-semibold text-foreground">SKY-Piano</strong>,
              a multimodal piano performance dataset of{" "}
              <strong className="font-semibold text-foreground">11 hours</strong> of
              performance recordings — motion, multi-view video, audio, and MIDI from{" "}
              <strong className="font-semibold text-foreground">7 professional and
              12 amateur pianists</strong> — along with MusicXML scores. The pieces were
              selected considering playing technique, difficulty, and performer expertise
              on a shared core repertoire.
            </p>
            <p>
              The motion data include both hand and body motion, released in both{" "}
              <em className="text-foreground">flagged</em> form, where samples lost to
              marker occlusion are marked as unreliable, and{" "}
              <em className="text-foreground">imputed</em> form, where those gaps are
              reconstructed, together with Visual3D body-segment kinematics and other
              time-synchronized modalities.
            </p>
            <p>
              To browse the modalities at a glance we provide an interactive web
              explorer. We additionally developed a fingering annotation model and tool
              that derives pseudo fingering annotations from the MIDI and motion data,
              and we present MIDI-to-motion generation through a fine-tuning experiment
              as a use case of the dataset.
            </p>
          </div>
        </div>
      </section>

      {/* Presentation: video and poster */}
      <section id="presentation" className="px-4 py-20">
        <div className="mx-auto max-w-4xl">
          <SectionHeading eyebrow="ISMIR 2026 · Poster P13-8" title="Video and poster" />
          <div className="surface-card mb-6 overflow-hidden">
            <div className="relative aspect-video w-full">
              <iframe
                className="absolute inset-0 h-full w-full"
                src={`https://www.youtube-nocookie.com/embed/${VIDEO_ID}`}
                title="SKY-Piano: A Multimodal Piano Performance Dataset"
                loading="lazy"
                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                referrerPolicy="strict-origin-when-cross-origin"
                allowFullScreen
              />
            </div>
          </div>
          <a
            href={POSTER_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="surface-card block overflow-hidden"
          >
            <img
              src={POSTER_URL}
              alt="SKY-Piano poster for ISMIR 2026"
              loading="lazy"
              className="h-auto w-full"
            />
          </a>
          <p className="mt-4 text-center text-sm text-muted-foreground">
            <a
              href={ISMIR_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-primary underline underline-offset-4"
            >
              ISMIR 2026 paper page
            </a>
            {" · "}
            <a
              href={VIDEO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-primary underline underline-offset-4"
            >
              Video on YouTube
            </a>
            {" · "}
            <a
              href={POSTER_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-primary underline underline-offset-4"
            >
              Poster (full resolution)
            </a>
          </p>
        </div>
      </section>

      {/* Explorer */}
      <section id="explorer" className="px-4 py-20">
        <div className="mx-auto max-w-5xl">
          <SectionHeading
            eyebrow="Interactive"
            title="Explore the data in your browser"
            lead="Every modality of a trial on one timeline — score, hand and body motion, multi-view video, audio, MIDI, fingering, and Visual3D kinematics."
          />
          <a
            href={EXPLORER_URL}
            className="surface-card surface-card-hover group block overflow-hidden"
          >
            <img
              src={`${import.meta.env.BASE_URL}teaser.png`}
              alt="The SKY-Piano explorer showing synchronized score, motion, video, audio, MIDI, and kinematics for one trial"
              className="w-full border-b border-border"
            />
            <div className="flex flex-wrap items-center justify-between gap-4 p-6">
              <div>
                <h3 className="text-lg font-semibold text-foreground">
                  Open the explorer
                </h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Sample trials spanning both technique categories, three difficulty
                  levels, and a professional/amateur comparison on the same exercise.
                </p>
              </div>
              <span className="inline-flex items-center gap-2 rounded-md brand-gradient-fill px-5 py-2.5 text-sm font-semibold text-primary-foreground transition-transform group-hover:translate-x-0.5">
                Launch
                <ArrowRight className="h-4 w-4" />
              </span>
            </div>
          </a>
        </div>
      </section>

      {/* Dataset: comparison + capture */}
      <section id="dataset" className="px-4 py-20">
        <div className="mx-auto max-w-6xl">
          <SectionHeading
            eyebrow="Positioning"
            title="How SKY-Piano compares"
            lead="No prior corpus combines directly measured hand and body motion with a repertoire that varies technique, difficulty, and expertise at once."
          />

          <div className="surface-card mb-16 overflow-x-auto">
            <table className="w-full min-w-[860px] text-sm">
              <thead>
                <tr className="surface-tint border-b border-border-strong text-[11px] uppercase tracking-[0.04em] text-muted-foreground">
                  <th className="px-4 py-3 text-left font-bold">Dataset</th>
                  <th className="px-3 py-3 font-bold">Hand<span className="ml-1 font-medium normal-case">(real)</span></th>
                  <th className="px-3 py-3 font-bold">Body<span className="ml-1 font-medium normal-case">(real)</span></th>
                  <th className="px-3 py-3 font-bold">Video</th>
                  <th className="px-3 py-3 font-bold">Audio</th>
                  <th className="px-3 py-3 font-bold">MIDI</th>
                  <th className="px-3 py-3 font-bold">Score</th>
                  <th className="px-3 py-3 font-bold">Fingering</th>
                  <th className="px-3 py-3 font-bold">Pro / Am</th>
                  <th className="px-3 py-3 font-bold">Pieces</th>
                  <th className="px-3 py-3 font-bold">Hours</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground">
                {COMPARISON.map((row) => (
                  <tr key={row.name} className="border-b border-border last:border-0">
                    <td className="px-4 py-2.5 font-medium text-foreground">{row.name}</td>
                    <td className="px-3 py-2.5 text-center">{row.hand ? <Yes /> : <No />}</td>
                    <td className="px-3 py-2.5 text-center">{row.body ? <Yes /> : <No />}</td>
                    <td className="px-3 py-2.5 text-center">{row.video}</td>
                    <td className="px-3 py-2.5 text-center">{row.audio ? <Yes /> : <No />}</td>
                    <td className="px-3 py-2.5 text-center">{row.midi ? <Yes /> : <No />}</td>
                    <td className="px-3 py-2.5 text-center">{row.score ? <Yes /> : <No />}</td>
                    <td className="px-3 py-2.5 text-center">{row.fingering ? <Yes /> : <No />}</td>
                    <td className="px-3 py-2.5 text-center">{row.pianists}</td>
                    <td className="px-3 py-2.5 text-center">{row.pieces}</td>
                    <td className="px-3 py-2.5 text-center">{row.hours}</td>
                  </tr>
                ))}
                <tr className="bg-primary/[0.06]">
                  <td className="px-4 py-3 font-bold text-primary">{OURS_ROW.name}</td>
                  <td className="px-3 py-3 text-center"><Yes /></td>
                  <td className="px-3 py-3 text-center"><Yes /></td>
                  <td className="px-3 py-3 text-center font-semibold text-primary">{OURS_ROW.video}</td>
                  <td className="px-3 py-3 text-center"><Yes /></td>
                  <td className="px-3 py-3 text-center"><Yes /></td>
                  <td className="px-3 py-3 text-center"><Yes /></td>
                  <td className="px-3 py-3 text-center"><Yes /></td>
                  <td className="px-3 py-3 text-center font-semibold text-primary">{OURS_ROW.pianists}</td>
                  <td className="px-3 py-3 text-center font-semibold text-primary">{OURS_ROW.pieces}</td>
                  <td className="px-3 py-3 text-center font-semibold text-primary">{OURS_ROW.hours}</td>
                </tr>
              </tbody>
            </table>
            <p className="border-t border-border px-4 py-3 text-xs text-subtle">
              <Yes /> available · <No /> not available · <Na /> not applicable.
              “Score” denotes composer-intended symbolic notation aligned to the
              performance, distinct from performance MIDI; “unlabeled” marks
              web-collected data with no per-pianist skill classification.
            </p>
          </div>

          <SectionHeading
            eyebrow="Acquisition"
            title="Capture system"
            lead="A Rosendahl nanoSync generator driven by a Black Magic master clock stamps every stream with a common 30 fps SMPTE timecode, so all modalities share one frame index."
          />
          <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {CAPTURE.map((c) => (
              <div key={c.title} className="surface-card p-6">
                <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-lg bg-primary/[0.07]">
                  <c.icon className="h-5 w-5 text-primary" />
                </div>
                <h3 className="mb-3 font-semibold text-foreground">{c.title}</h3>
                <ul className="space-y-1.5">
                  {c.specs.map((s) => (
                    <li key={s} className="flex gap-2 text-sm text-muted-foreground">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-primary/40" />
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <div className="surface-card flex flex-wrap items-center gap-x-8 gap-y-4 p-6">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary/[0.07]">
                <Clock className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold text-foreground">Release statistics</h3>
                <p className="text-sm text-muted-foreground">
                  Averages per pianist, across four recording sessions
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm text-muted-foreground">
              <span>
                <strong className="text-foreground">39.4 min</strong> per professional
                <span className="text-subtle"> (30–53)</span>
              </span>
              <span>
                <strong className="text-foreground">32.1 min</strong> per amateur
                <span className="text-subtle"> (20–56)</span>
              </span>
              <span>
                <strong className="text-foreground">14%</strong> median marker dropout
                <span className="text-subtle"> (6–44% by marker)</span>
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Repertoire */}
      <section id="repertoire" className="px-4 py-20">
        <div className="mx-auto max-w-5xl">
          <SectionHeading
            eyebrow="Design"
            title="Selected repertoire"
            lead="Eleven technique exercises from Fink's Mastering Piano Technique, and graded pieces anchored to the Royal Conservatory of Music Piano Syllabus. Most graded pieces are recorded as excerpts, and the released MusicXML is trimmed to the recorded passage."
          />

          <div className="mb-4 grid gap-4 md:grid-cols-2">
            <div className="surface-card p-6">
              <h3 className="mb-4 font-semibold text-foreground">
                Technique 1
                <span className="ml-2 text-sm font-normal text-subtle">
                  movement-oriented exercises
                </span>
              </h3>
              <div className="flex flex-wrap gap-2">
                {TECHNIQUE_1.map((t) => (
                  <span
                    key={t}
                    className="rounded-md bg-primary/[0.07] px-2.5 py-1.5 text-sm text-primary"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <div className="surface-card p-6">
              <h3 className="mb-4 font-semibold text-foreground">
                Technique 2
                <span className="ml-2 text-sm font-normal text-subtle">
                  standard pianistic patterns
                </span>
              </h3>
              <div className="flex flex-wrap gap-2">
                {TECHNIQUE_2.map((t) => (
                  <span
                    key={t}
                    className="rounded-md bg-accent/[0.07] px-2.5 py-1.5 text-sm text-accent"
                  >
                    {t}
                  </span>
                ))}
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                Each sitting performs all five drills at two target tempi, so the
                C-major slow scale alone is played by all 19 participants.
              </p>
            </div>
          </div>

          <div className="surface-card mb-4 p-6 md:p-8">
            <h3 className="mb-6 font-semibold text-foreground">Graded pieces</h3>
            <div className="space-y-5">
              {GRADED.map((g) => (
                <div key={g.level} className="flex flex-col gap-2 sm:flex-row sm:gap-6">
                  <div className="w-32 shrink-0">
                    <span className="eyebrow">{g.level}</span>
                  </div>
                  <ul className="flex-1 space-y-1 text-sm text-muted-foreground">
                    {g.pieces.map((p) => (
                      <li key={p}>{p}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          <div className="surface-card p-6 md:p-8">
            <h3 className="mb-2 font-semibold text-foreground">
              Free repertoire
              <span className="ml-2 text-sm font-normal text-subtle">
                professional, performer-selected
              </span>
            </h3>
            <p className="text-sm leading-relaxed text-muted-foreground">
              {FREE_PIECES.join(" · ")}
            </p>
          </div>
        </div>
      </section>

      {/* Fingering */}
      <section id="fingering" className="px-4 py-20">
        <div className="mx-auto max-w-5xl">
          <SectionHeading
            eyebrow="Annotation"
            title="Mocap-grounded pseudo fingering"
            lead="At the moment a MIDI note sounds, the finger that pressed it is the one whose measured fingertip sits inside that key and lowest in depth. Three tiers work outward from that evidence."
          />

          <div className="mb-6 grid gap-4 md:grid-cols-3">
            {FINGERING_TIERS.map((t) => (
              <div key={t.tier} className="surface-card p-6">
                <span className="eyebrow text-primary">{t.tier}</span>
                <h3 className="mb-2 mt-2 font-semibold text-foreground">{t.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{t.desc}</p>
              </div>
            ))}
          </div>

          <div className="surface-card grid gap-6 p-6 sm:grid-cols-3 md:p-8">
            {[
              { value: "94.5%", label: "Strict precision", sub: "over the 94.0%-coverage committed subset" },
              { value: "2,269", label: "Hand-audited notes", sub: "three graded pieces, three pianists" },
              { value: "91.4%", label: "Keystroke detection", sub: "of 113,023 professional MIDI notes" },
            ].map((m) => (
              <div key={m.label}>
                <div className="text-3xl font-bold text-primary">{m.value}</div>
                <div className="mt-1 text-sm font-medium text-foreground">{m.label}</div>
                <div className="mt-0.5 text-xs text-subtle">{m.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Access & citation */}
      <section id="cite" className="px-4 py-20">
        <div className="mx-auto max-w-3xl">
          <SectionHeading eyebrow="Access" title="Data and code" />

          <div className="surface-card mb-4 p-6 md:p-8">
            <p className="text-[15px] leading-relaxed text-muted-foreground">
              The pipeline and explorer source are released on{" "}
              <a
                href={CODE_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-primary underline underline-offset-4"
              >
                GitHub
              </a>
              . The dataset is distributed on{" "}
              <a
                href={DATASET_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-primary underline underline-offset-4"
              >
                Hugging Face
              </a>{" "}
              with gated access: request it there and agree to the license terms;
              requests are reviewed every Monday. It is released in versions, adding
              modalities until the complete 1.0.0, which will also be released on
              Zenodo. Sample trials covering every modality
              are browsable in the{" "}
              <a href={EXPLORER_URL} className="font-medium text-primary underline underline-offset-4">
                explorer
              </a>
              .
            </p>
            <p className="mt-4 text-[15px] leading-relaxed text-muted-foreground">
              License: The SKY-Piano dataset is released under{" "}
              <a
                href={LICENSE_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-primary underline underline-offset-4"
              >
                CC BY-NC-ND 4.0
              </a>
              , with an Additional Permission for Research Use that allows researchers
              from any organization, including commercial ones, to use the dataset for
              bona fide scientific research. See the full license terms on Hugging Face.
            </p>
            <p className="mt-4 text-sm leading-relaxed text-subtle">
              The study was approved by the Institutional Review Board of KAIST. All
              participants provided written informed consent covering audio, MIDI,
              motion capture, and face-anonymized video, and the consent specifies the
              public-release scope per modality.
            </p>
          </div>

          <div className="surface-card overflow-hidden">
            <div className="flex items-center justify-between border-b border-border px-5 py-3">
              <span className="eyebrow">BibTeX</span>
              <button
                onClick={copyBibtex}
                className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-semibold text-muted-foreground transition-colors hover:bg-primary/5 hover:text-primary"
              >
                {copied ? <Check className="h-3.5 w-3.5" /> : <ClipboardCopy className="h-3.5 w-3.5" />}
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
            <pre className="custom-scrollbar overflow-x-auto px-5 py-4 text-xs leading-relaxed text-muted-foreground">
              {BIBTEX}
            </pre>
          </div>
        </div>
      </section>

      <footer className="border-t border-border px-4 py-12">
        <div className="mx-auto max-w-4xl text-center">
          <p className="text-lg font-bold brand-gradient-text">SKY-Piano</p>
          <p className="mt-2 text-sm text-muted-foreground">
            A Multimodal Piano Performance Dataset
          </p>
          <p className="mt-4 text-xs text-subtle">
            KAIST · Seoul National University · Yamaha Corporation — ISMIR 2026
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
