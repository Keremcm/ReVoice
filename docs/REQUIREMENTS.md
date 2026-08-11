# Requirements & Installation

## System requirements

### Hardware

| Component | Minimum | Recommended (tested) |
| --- | --- | --- |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU works, slow) | NVIDIA, ~6 GB VRAM (tested: RTX 4050 6 GB) |
| Disk | 15 GB free | 20+ GB (model caches + outputs) |

GPU memory notes:

- XTTS v2 needs **~2.2 GB free VRAM**.
- Whisper `medium` (float16) needs roughly **2.5–3 GB**.
- Ollama translation runs on **CPU only** (`num_gpu: 0`), so it never competes
  for VRAM.
- If VRAM is tight, run everything on CPU with `--device cpu`.

### Software

- **Linux** (tested) — macOS/Windows should work with minor changes
- **Python 3.11** with `venv`
- **ffmpeg + ffprobe** on `PATH` (used for extraction, stretching and muxing)
- **Ollama** running locally at `http://localhost:11434`
  (`ollama serve`) for translation

## Installation

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd transvoice

# 2. Create a virtual environment and install dependencies
python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# 3. Verify system tools
ffmpeg -version
ollama --version && ollama serve   # keep it running

# 4. Pre-pull a translation model (optional, done automatically otherwise)
ollama pull translategemma:4b-it-q8_0
```

`run.sh` sets up the CUDA library path (system `nvidia` packages or Ollama's
vendored CUDA) before launching `main.py`, so on NVIDIA GPUs you can simply run:

```bash
./run.sh --input video.mp4 --target-lang en
```

## One-time model downloads

The first run downloads the following models to the local cache (nothing is
sent to anyone; these are on-disk downloads from public model hubs):

| Model | Purpose | Approx. size | Cache location |
| --- | --- | --- | --- |
| faster-whisper `medium` | Transcription | ~1.5 GB | `~/.cache/huggingface` |
| XTTS v2 | Voice cloning | ~1.8 GB | `~/.local/share/tts` / HF cache |
| Silero VAD | Voice-activity detection | ~2 MB | `~/.cache/torch/hub` |
| ECAPA-TDNN (`spkrec-ecapa-voxceleb`) | Speaker embeddings | ~40 MB | `~/.cache/transvoice` |
| `translategemma:4b-it-q8_0` | Translation (via Ollama) | ~2.5 GB | Ollama model dir |

You can substitute any Ollama model with `--ollama-model`.

## Troubleshooting

**`GPU belleği yetersiz` (insufficient GPU memory)**
: Another process (usually Ollama) is holding VRAM. Check `ollama ps`, run
  `ollama stop <model>`, or use `--device cpu`.

**`Komut başarısız: ffmpeg ...`**
: ffmpeg/ffprobe not found or an audio-codec issue. Ensure both are on `PATH`.

**No speech detected / empty segments**
: The video has no clear speech, or `--source-lang` is wrong.

**Diarization reports the wrong speaker count**
: Auto-detection is heuristic. Force it: `--diarize --speakers N`.

**Slow on CPU**
: Use `--device cuda` with a NVIDIA GPU, or accept CPU speed (still fully
  functional).
