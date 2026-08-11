import json
import subprocess
import tempfile

from .utils import ffprobe_duration, run

_QUIET = ["-hide_banner", "-loglevel", "error"]


def extract_audio(video_path, out_wav, sample_rate=22050):
    run(
        [
            "ffmpeg",
            "-y",
            *_QUIET,
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            out_wav,
        ]
    )


def time_stretch(in_path, out_path, tempo, sample_rate=22050, channels=1):
    tempo = max(0.25, min(4.0, tempo))
    filters = []
    factor = tempo
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    filters.append(f"atempo={factor:.6f}")
    run(
        [
            "ffmpeg",
            "-y",
            *_QUIET,
            "-i",
            in_path,
            "-filter:a",
            ",".join(filters),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            out_path,
        ]
    )


def mux(video_path, audio_path, out_path, bitrate="192k"):
    run(
        [
            "ffmpeg",
            "-y",
            *_QUIET,
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            "-shortest",
            out_path,
        ]
    )
