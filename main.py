import argparse
import gc
import logging
import os
import shutil
import sys
import time
import warnings

from transvoice import audio, sync, transcribe, translate
from transvoice.tts_clone import VoiceCloner
from transvoice.utils import ffprobe_duration, get_device

log = logging.getLogger("transvoice")


def silence_noisy_libs():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    warnings.filterwarnings("ignore")
    for name in (
        "TTS",
        "transformers",
        "faster_whisper",
        "ctranslate2",
        "datasets",
        "huggingface_hub",
        "torch",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def free_gpu():
    if get_device() == "cuda":
        import torch

        gc.collect()
        torch.cuda.empty_cache()


def require_vram(min_bytes):
    import torch

    free, _ = torch.cuda.mem_get_info()
    if free < min_bytes:
        sys.exit(
            f"Hata: GPU belleği yetersiz ({free // 2**20}MB boş, {min_bytes // 2**20}MB gerekli). "
            "Başka bir model GPU'yu tutuyor olabilir (ör: ollama). Çözüm:\n"
            "  ollama stop <yüklü-model>   # hangi model GPU'daysa (ollama ps ile görün)\n"
            "  veya --device cpu ile CPU üzerinde çalıştırın"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Video ses dilini değiştir: transkripsiyon + çeviri + ses klonlama")
    p.add_argument("--input", required=True, help="Kaynak video dosyası")
    p.add_argument("--output", default=None, help="Çıktı video (varsayılan: <input>_dubbed.mp4)")
    p.add_argument("--target-lang", default="en", help="Hedef dil kodu (ör: en, tr, de, es)")
    p.add_argument("--source-lang", default=None, help="Kaynak dil kodu (boşsa otomatik algılanır)")
    p.add_argument("--whisper-model", default="medium", help="Whisper model adı (maks: medium; ör: tiny, base, small, medium)")
    p.add_argument("--ollama-model", default="translategemma:4b-it-q8_0", help="Ollama çeviri modeli")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API adresi")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--keep-tmp", action="store_true", help="Ara dosyaları silme")
    p.add_argument("--ref-text", default=None, help="XTTS referans metni (opsiyonel)")
    p.add_argument("--diarize", action="store_true", help="Konuşmacı ayrımı yap, her kişiye kendi sesiyle seslendir")
    p.add_argument("--speakers", type=int, default=None, help="Diarizasyonda kişi sayısını zorla (opsiyonel, otomatik ise boş bırak)")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("transvoice").setLevel(logging.INFO)
    silence_noisy_libs()
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Hata: {args.input} bulunamadı")
    if args.whisper_model.startswith(("large", "distil")):
        sys.exit(f"Hata: whisper modeli 'large'/'distil' olamaz (GPU sığmıyor). Maksimum: medium")
    for bin_name in ("ffmpeg", "ffprobe"):
        if not shutil.which(bin_name):
            sys.exit(f"Hata: {bin_name} sistemde yok")

    device = get_device() if args.device == "auto" else args.device
    compute_type = "float16" if device == "cuda" else "int8"

    output = args.output or os.path.splitext(args.input)[0] + "_dubbed.mp4"
    workdir = output + ".work"
    os.makedirs(workdir, exist_ok=True)

    original_wav = os.path.join(workdir, "original.wav")
    t0 = time.time()
    total_steps = 7 if args.diarize else 6

    try:
        log.info("Video: %s (%.1fs)", os.path.basename(args.input), ffprobe_duration(args.input))
        log.info("Hedef: %s · Whisper: %s · Ollama: %s · Cihaz: %s", args.target_lang, args.whisper_model, args.ollama_model, device)

        log.info("Adım 1/%d: video sesi çıkarılıyor", total_steps)
        audio.extract_audio(args.input, original_wav)
        total_dur = ffprobe_duration(original_wav)

        log.info("Adım 2/%d: transkripsiyon", total_steps)
        src_lang, segments = transcribe.transcribe(
            original_wav,
            model_name=args.whisper_model,
            device=device,
            compute_type=compute_type,
            language=args.source_lang,
        )
        if not segments:
            sys.exit("Hata: konuşma tespit edilemedi")
        log.info("Kaynak dil: %s (%s -> %s) · %d segment", src_lang, src_lang, args.target_lang, len(segments))
        free_gpu()

        log.info("Adım 3/%d: çeviri", total_steps)
        translate.ensure_model(args.ollama_model, base=args.ollama_url)
        segments = translate.translate_segments(
            segments,
            dst_lang=args.target_lang,
            src_lang=src_lang,
            model=args.ollama_model,
            base=args.ollama_url,
        )
        translate.unload_model(args.ollama_model, base=args.ollama_url)

        speaker_refs = None
        if args.diarize:
            log.info("Adım 4/%d: konuşmacı ayrımı (diarizasyon)", total_steps)
            from transvoice import diarize as diarize_mod

            turns, speaker_refs = diarize_mod.diarize(
                original_wav,
                workdir,
                device=device,
                num_speakers=args.speakers,
            )
            assignments = diarize_mod.assign_segments(segments, turns)
            for seg, spk in zip(segments, assignments):
                seg.speaker = spk
            log.info(
                "%d konuşmacı bulundu (%s)",
                len(speaker_refs),
                " → ".join(os.path.basename(p) for p in speaker_refs.values()),
            )
            free_gpu()

        step = 5 if args.diarize else 4
        log.info("Adım %d/%d: ses klonlama modeli yükleniyor (XTTS v2)", step, total_steps)
        if device == "cuda":
            require_vram(2_200_000_000)
        silence_noisy_libs()
        cloner = VoiceCloner(device=device)

        step = 6 if args.diarize else 5
        log.info("Adım %d/%d: segment seslendirme", step, total_steps)
        seg_wavs = []
        for i, seg in enumerate(segments):
            out = os.path.join(workdir, f"seg_{i:04d}.wav")
            cloner.synthesize(
                seg.translated,
                out,
                reference_wav=speaker_refs[seg.speaker] if speaker_refs else original_wav,
                target_lang=args.target_lang,
                ref_text=args.ref_text,
            )
            seg_wavs.append(out)
            who = f" · ses {seg.speaker + 1}" if speaker_refs else ""
            log.info("  [%d/%d]%.1fs-%.1fs%s -> %s", i + 1, len(segments), seg.start, seg.end, who, seg.translated)

        del cloner
        free_gpu()

        step = 7 if args.diarize else 6
        log.info("Adım %d/%d: senkronizasyon ve paketleme", step, total_steps)
        fitted = sync.fit_segments(segments, seg_wavs, workdir)
        dubbed_wav = os.path.join(workdir, "dubbed.wav")
        sync.assemble(segments, fitted, total_dur, dubbed_wav)
        audio.mux(args.input, dubbed_wav, output)

        log.info("Tamamlandı: %s (%.0fdk %.0fsn)", output, (time.time() - t0) // 60, (time.time() - t0) % 60)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
