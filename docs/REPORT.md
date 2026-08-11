# Technical Report

## 1. Overview

ReVoice is a fully local video-dubbing pipeline. Given a video, it replaces the
spoken language while preserving timing and the original speaker's voice. All
heavy lifting (ASR, translation, TTS, diarization) runs on the machine.

```
video ─► extract_audio ─► transcribe ─► translate ─► [diarize] ─► tts ─► fit/assemble ─► mux ─► video
        (audio.py)       (faster-whisper) (Ollama)  (diarize.py)  (XTTS)  (sync.py)      (audio.py)
```

Pipeline steps: 6 by default, 7 with `--diarize`.

## 2. Modules

### `transvoice/audio.py`
Thin ffmpeg wrappers, all silenced with `-hide_banner -loglevel error`:

- `extract_audio()` — mono 22.05 kHz PCM extraction (`pcm_s16le`).
- `time_stretch()` — atempo chain (supports tempo 0.25–4.0) for sync fitting.
- `mux()` — remuxes original video stream with the dubbed audio (AAC, `-c:v copy`).

### `transvoice/transcribe.py`
faster-whisper transcription into `Segment(start, end, text, translated, speaker)`
dataclasses. VAD filtering is enabled. Whisper is deliberately capped at
`medium` (see §5). GPU is released (`torch.cuda.empty_cache`) right after
transcription.

### `transvoice/translate.py`
Per-segment translation through Ollama's local API (`/api/generate`, `stream:false`,
`temperature 0.1`, `num_gpu: 0` so the GPU stays free for TTS). Falls back to the
original text if the model returns nothing. `ensure_model()` auto-pulls missing
Ollama models. Default model: `translategemma:4b-it-q8_0`.

### `transvoice/diarize.py`
Multi-speaker detection (used only with `--diarize`):

1. **VAD** — Silero VAD at 16 kHz produces speech chunks; chunks closer than
   `MERGE_GAP = 0.3 s` are merged, chunks shorter than `MIN_CHUNK = 1.0 s` are
   dropped (too noisy for reliable embeddings).
2. **Embeddings** — SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`) produces a
   192-d speaker embedding per chunk.
3. **Clustering** — agglomerative clustering on cosine distance
   (`linkage="average"`) with `CLUSTER_THRESHOLD = 0.5`; `--speakers N` overrides
   with a fixed cluster count.
4. **References** — for each speaker, *all* of their chunks are concatenated in
   time order (0.15 s silence padding between) into `ref_speaker_N.wav`. More
   reference audio ⇒ better XTTS clone.
5. **Assignment** — each Whisper segment is assigned to the speaker with the
   greatest time overlap; unassigned segments go to the most-talkative speaker.

The VAD and ECAPA models are loaded lazily and cached per process.

### `transvoice/tts_clone.py`
Coqui XTTS v2 wrapper. `XTTS_LANGS` maps language codes to XTTS's multilingual
set (17 languages). Output is quieted by redirecting stdout to `/dev/null`
during model load and synthesis, and a `weights_only=False` patch is applied for
torch 2.6+ state-loading compatibility. Each synthesized segment uses the
reference WAV of its assigned speaker.

### `transvoice/sync.py`
Fits every synthesized segment to its original slot:

- `fit_segments()` — computes `tempo = duration/T` (clamped to 0.5–2.0), applies
  `time_stretch()`, then pads or truncates to the exact target duration.
- `assemble()` — paints each fitted segment onto a 22.05 kHz silence canvas at
  the original segment offset, yielding the final mono `dubbed.wav`.

### `transvoice/utils.py`
`run()` (subprocess with error capture — last 3 stderr lines on failure),
`get_device()` (torch CUDA probe), `ffprobe_duration()`.

### `main.py`
CLI orchestration: arg parsing, environment silencing (`silence_noisy_libs`),
VRAM guard (`require_vram(2.2 GB)` before XTTS), GPU cache management
(`free_gpu` between stages), step-by-step progress logging, and cleanup of the
`<output>.work/` directory unless `--keep-tmp` is passed.

## 3. Key parameters

| Parameter | Value | Reason |
| --- | --- | --- |
| Audio sample rate | 22.05 kHz (mono) | Transcription + sync |
| Diarization rate | 16 kHz | Silero/ECAPA expectation |
| VAD min chunk | 1.0 s | Drop noisy snippets |
| VAD merge gap | 0.3 s | Avoid merging across speaker turns |
| Cluster threshold | 0.5 (cosine, avg-linkage) | Balances over/under-merging |
| TTS VRAM guard | 2.2 GB free | XTTS footprint |
| Sync tempo clamp | 0.25–4.0 (fit: 0.5–2.0) | Avoid audible artifacts |
| Ollama `temperature` | 0.1 | Deterministic translations |

## 4. Test results

All tests ran on the dev machine (RTX 4050 6 GB, Ubuntu).

| Test | Input | Result |
| --- | --- | --- |
| End-to-end dubbing (TR→EN/DE/RU/KO) | `deneme_*.mp4` (~12–46 s) | Outputs produced, timing preserved |
| Diarization, single speaker | 46 s monologue | Correctly detected **1 speaker** |
| Diarization, two speakers | synthetic 2-voice clip | Detected **2 clusters** |
| Diarization runtime | 46 s audio, CPU | ~4.5 s (VAD+ECAPA+cluster) |
| Two-speaker end-to-end | `iki.mp4` | `iki_dubbed.mp4` produced with `--diarize` |

Note: synthetic pitch-shifted "second voices" are not reliably separated by
ECAPA (they are the same acoustic identity); real distinct speakers separate
cleanly. Use `--speakers N` if auto-detection misjudges.

## 5. Known limitations & trade-offs

- **Whisper capped at `medium`** — `large`/`distil` exceed the 6 GB VRAM budget
  alongside XTTS. Larger models could be enabled on bigger GPUs by removing the
  guard in `main.py:84`.
- **Speaker-count detection is heuristic** — a fixed cosine threshold cannot be
  perfect across all acoustic conditions; the `--speakers` override is the
  escape hatch.
- **Noise/music degrade embeddings** — background music lowers cross-chunk
  cosine similarity and can fragment clusters; clean dialogue diarizes best.
- **XTTS is not trained for long-form consistency** — cloned timbre is stable,
  but very long videos may show slight prosody drift.
- **UI is terminal-only** by design; no web interface is shipped.

## 6. Roadmap ideas

- Configurable Whisper model size with automatic VRAM budgeting.
- Temperature/syllable-level alignment for tighter lip-sync.
- Diarization quality metrics + calibration of `CLUSTER_THRESHOLD`.
- Optional denoising (e.g. `deepfilternet`) before diarization.
- CUDA graphs / batching to speed up multi-segment synthesis.
