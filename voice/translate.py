import subprocess

import requests

from .transcribe import Segment

LANG_NAMES = {
    "en": "English",
    "tr": "Turkish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "nl": "Dutch",
    "cs": "Czech",
    "pl": "Polish",
    "hi": "Hindi",
    "hu": "Hungarian",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "uk": "Ukrainian",
}


def _lang_name(code):
    return LANG_NAMES.get(code, code)


def _translate_one(text, src_lang, dst_lang, model, base, timeout):
    prompt = (
        f"Translate the following text from {_lang_name(src_lang)} to {_lang_name(dst_lang)}.\n"
        "Output only the translation, nothing else.\n\n"
        f"Text: {text}\n\nTranslation:"
    )
    resp = requests.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512, "num_gpu": 0},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def ensure_model(model, base="http://localhost:11434"):
    resp = requests.get(f"{base}/api/tags", timeout=30)
    resp.raise_for_status()
    names = {m["name"] for m in resp.json().get("models", [])}
    if model not in names:
        print(f"Ollama modeli '{model}' indiriliyor...")
        proc = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Ollama indirme başarısız: {proc.stderr.strip()}")


def unload_model(model, base="http://localhost:11434"):
    try:
        requests.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=15,
        )
    except requests.RequestException:
        pass


def translate_segments(
    segments,
    dst_lang,
    src_lang=None,
    model="translategemma:4b-it-q8_0",
    base="http://localhost:11434",
    timeout=180,
):
    for i, seg in enumerate(segments):
        try:
            out = _translate_one(seg.text, src_lang, dst_lang, model, base, timeout)
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama çeviri hatası (segment {i}): {e}") from e
        seg.translated = out or seg.text
    return segments
