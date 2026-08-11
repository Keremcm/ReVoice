import shutil
import subprocess


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        detail = err[-3:] if err else []
        raise RuntimeError(f"Komut başarısız: {' '.join(cmd)}\n" + "\n".join(detail))
    return proc


def which(bin):
    return shutil.which(bin)


def get_device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ffprobe_duration(path):
    import json

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])
