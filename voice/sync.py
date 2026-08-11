import wave

from .audio import time_stretch
from .transcribe import Segment
from .utils import ffprobe_duration

SAMPLE_RATE = 22050
SAMPLE_WIDTH = 2


def _read_frames(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != SAMPLE_WIDTH or w.getnchannels() != 1:
            raise ValueError(f"{path}: mono 16-bit PCM bekleniyor")
        return w.getframerate(), w.readframes(w.getnframes())


def _write_frames(path, frames):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(frames)


def _pad_to_duration(path, out_path, target_dur):
    target_frames = int(target_dur * SAMPLE_RATE)
    sr, frames = _read_frames(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: örnekleme hızı {sr} beklenen {SAMPLE_RATE}")
    cur_frames = len(frames) // SAMPLE_WIDTH
    if cur_frames > target_frames:
        frames = frames[: target_frames * SAMPLE_WIDTH]
    elif cur_frames < target_frames:
        frames += b"\x00" * ((target_frames - cur_frames) * SAMPLE_WIDTH)
    _write_frames(out_path, frames)


def fit_segments(segments, seg_wav_paths, workdir, max_tempo=2.0):
    """Her segmenti orijinal süresine (end-start) uydurur ve yazdırır."""
    out_paths = []
    for i, (seg, wav) in enumerate(zip(segments, seg_wav_paths)):
        target_dur = max(0.1, seg.end - seg.start)
        cur_dur = ffprobe_duration(wav)
        tempo = min(max_tempo, max(1.0 / max_tempo, cur_dur / target_dur)) if cur_dur > 0 else 1.0
        stretched = f"{workdir}/stretched_{i:04d}.wav"
        time_stretch(wav, stretched, tempo)
        fitted = f"{workdir}/fitted_{i:04d}.wav"
        _pad_to_duration(stretched, fitted, target_dur)
        out_paths.append(fitted)
    return out_paths


def assemble(segments, fitted_wavs, total_duration, out_path):
    total_frames = int(total_duration * SAMPLE_RATE)
    canvas = bytearray(total_frames * SAMPLE_WIDTH)
    for seg, wav in zip(segments, fitted_wavs):
        sr, frames = _read_frames(wav)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{wav}: örnekleme hızı {sr}")
        offset = int(seg.start * SAMPLE_RATE) * SAMPLE_WIDTH
        end = min(len(canvas), offset + len(frames))
        canvas[offset:end] = frames[: end - offset]
    _write_frames(out_path, bytes(canvas))
