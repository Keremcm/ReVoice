# ReVoice — Fully Local Video Dubbing

ReVoice re-voices videos entirely on your machine. It transcribes the original
speech, translates it into a target language, clones the original speaker's
voice (or each speaker separately), and re-syncs the dubbed audio back to the
video. **No cloud, no uploads, no API keys** — every step runs locally.

```
input video ──► transcribe ──► translate ──► voice clone ──► sync ──► dubbed video
               (Whisper)      (Ollama LLM)  (XTTS v2)       (ffmpeg)
```

## Features

- **100% local & private** — audio never leaves your computer.
- **Voice cloning** with Coqui XTTS v2 (multilingual, 17 target languages).
- **Multi-speaker support** (`--diarize`) — each detected speaker is assigned
  their own voice and their *entire* recorded speech is used as the cloning
  reference.
- **Automatic sync** — every dubbed segment is time-stretched/padded to match
  the original timing, so the video stays in lip-sync.
- **Clean terminal UX** — noisy library output is silenced; a concise summary,
  step-by-step progress and total time are printed.
- **GPU or CPU** — NVIDIA GPU (auto-detected) with CPU fallback.

## Quick start

Requirements: Python 3.11, ffmpeg/ffprobe, a running **Ollama** server
(`ollama serve`), and optionally an NVIDIA GPU. See
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for the full setup guide.

```bash
git clone https://github.com/Keremcm/ReVoice && cd revoice
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt
ffmpeg -version   # must be on PATH
ollama pull translategemma:4b-it-q8_0   # or your preferred translation model

# Dub a video (Turkish → English):
./run.sh --input video.mp4 --target-lang en

# Multi-speaker video (each speaker keeps their own voice):
./run.sh --input interview.mp4 --target-lang tr --diarize

# Force a specific number of speakers if auto-detection is off:
./run.sh --input interview.mp4 --target-lang tr --diarize --speakers 2
```

The first run downloads the models (Whisper, XTTS v2, Silero VAD, ECAPA) into
your local cache — a one-time, on-disk download. The result is written as
`<input>_dubbed.mp4`.

## CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--input` | *(required)* | Source video file |
| `--output` | `<input>_dubbed.mp4` | Output video path |
| `--target-lang` | `en` | Target language code (e.g. `en, tr, de, es`) |
| `--source-lang` | auto | Source language code (auto-detected if empty) |
| `--whisper-model` | `medium` | Whisper model (max `medium`; `tiny/base/small/medium`) |
| `--ollama-model` | `translategemma:4b-it-q8_0` | Ollama translation model |
| `--ollama-url` | `http://localhost:11434` | Ollama API address |
| `--device` | `auto` | `cuda`, `cpu` or `auto` |
| `--keep-tmp` | off | Keep intermediate files (default: removed) |
| `--ref-text` | — | Optional reference text for XTTS |
| `--diarize` | off | Split speakers; dub each with its own voice |
| `--speakers` | auto | Force speaker count when `--diarize` is on |

## How it works

| Step | Task | Tool |
| --- | --- | --- |
| 1 | Extract mono audio (22.05 kHz) from video | ffmpeg |
| 2 | Transcribe into timestamped segments | faster-whisper |
| 3 | Translate each segment | Ollama (local LLM) |
| 4 *(optional)* | Speaker diarization | Silero VAD → SpeechBrain ECAPA-TDNN → agglomerative clustering |
| 5 | Load voice-clone model | Coqui XTTS v2 |
| 6 | Synthesize each segment with the speaker's reference voice | XTTS v2 |
| 7 | Fit segments to original timing, mix and mux | ffmpeg |

## Project layout

```
revoice/
├── main.py                 CLI entry point & pipeline orchestration
├── run.sh                  Launcher (sets CUDA library paths)
├── requirements.txt
└── voice/
    ├── audio.py            ffmpeg helpers (extract, stretch, mux)
    ├── transcribe.py       Whisper transcription + Segment dataclass
    ├── translate.py        Ollama translation + language table
    ├── diarize.py          Speaker diarization & segment assignment
    ├── tts_clone.py        XTTS v2 voice cloning
    ├── sync.py             Timing fit (time-stretch + pad + assemble)
    └── utils.py            subprocess, device & ffprobe helpers
```

## Limitations

- **Whisper model is capped at `medium`** — `large` does not fit in the tested
  6 GB GPU along with XTTS; use `--device cpu` if you must.
- **GPU memory**: XTTS v2 needs ~2.2 GB free VRAM. If Ollama keeps a model in
  VRAM (`ollama ps`), stop it first (`ollama stop <model>`) or run on CPU.
- **Auto speaker count is heuristic** — if diarization under/over-splits, force
  the count with `--speakers N`.
- **Reference quality matters** — the longer and cleaner a speaker's recorded
  speech, the better the cloned voice. ReVoice uses *all* of each speaker's
  speech as the reference.

## License

MIT
