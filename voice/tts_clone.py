import contextlib
import logging
import os
import sys

for _name in ("TTS", "transformers", "ctranslate2", "datasets", "huggingface_hub", "urllib3"):
    logging.getLogger(_name).setLevel(logging.ERROR)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

XTTS_LANGS = {
    "ar": "ar", "zh-cn": "zh-cn", "zh": "zh-cn", "cs": "cs", "nl": "nl",
    "en": "en", "fr": "fr", "de": "de", "hi": "hi", "hu": "hu",
    "it": "it", "ja": "ja", "ko": "ko", "pl": "pl", "pt": "pt",
    "ru": "ru", "es": "es", "tr": "tr",
}


def to_xtts_lang(code):
    if code in XTTS_LANGS:
        return XTTS_LANGS[code]
    raise ValueError(
        f"XTTS '{code}' dilini desteklemiyor. Desteklenen: {sorted(set(XTTS_LANGS.values()))}"
    )


def _patch_torch_load():
    import torch

    import TTS.utils.io as tts_io

    orig = tts_io.load_fsspec

    def patched(path, map_location=None, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(path, map_location=map_location, **kwargs)

    tts_io.load_fsspec = patched


_patch_torch_load()


@contextlib.contextmanager
def _quiet_stdout():
    old = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


class VoiceCloner:
    def __init__(self, device="cpu"):
        from TTS.api import TTS

        with _quiet_stdout():
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def synthesize(self, text, out_path, reference_wav, target_lang="en", ref_text=None):
        with _quiet_stdout():
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_wav,
                language=to_xtts_lang(target_lang),
                file_path=out_path,
                ref_text=ref_text,
            )
