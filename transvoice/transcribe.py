from dataclasses import dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str
    translated: str = ""
    speaker: int = 0


def transcribe(audio_path, model_name="large-v3", device="cpu", compute_type="int8", language=None):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        word_timestamps=False,
    )
    segs = [
        Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in segments
        if s.text.strip()
    ]
    _unload_gpu(model, device)
    return info.language, segs


def _unload_gpu(model, device):
    del model
    if device == "cuda":
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
