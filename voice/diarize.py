import os

import numpy as np

SAMPLE_RATE = 16000
MODEL_CACHE = os.path.expanduser("~/.cache/transvoice")
MIN_CHUNK = 1.0
MERGE_GAP = 0.3
TURN_MERGE_GAP = 0.8
CLUSTER_THRESHOLD = 0.5

_vad = None
_embedder = None


def _load_vad():
    global _vad
    if _vad is None:
        import torch

        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        _vad = (model, utils)
    return _vad


def _load_embedder(device):
    global _embedder
    if _embedder is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _embedder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(MODEL_CACHE, "ecapa"),
            run_opts={"device": device},
        )
        _embedder.eval()
    return _embedder


def _read_mono_16k(path):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        gcd = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // gcd, sr // gcd)
    return audio


def _vad_chunks(audio):
    import torch

    model, utils = _load_vad()
    get_speech_timestamps = utils[0]
    with torch.no_grad():
        tstamps = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=SAMPLE_RATE)
    chunks = []
    for ts in tstamps:
        s, e = int(ts["start"]), int(ts["end"])
        if chunks and (s - chunks[-1][1]) / SAMPLE_RATE < MERGE_GAP:
            chunks[-1] = (chunks[-1][0], e)
        elif (e - s) / SAMPLE_RATE >= MIN_CHUNK:
            chunks.append((s, e))
    return chunks


def _embeddings(audio, chunks, device):
    import torch

    embedder = _load_embedder(device)
    embs = []
    with torch.no_grad():
        for s, e in chunks:
            wav = torch.from_numpy(audio[s:e]).unsqueeze(0)
            emb = embedder.encode_batch(wav).squeeze().cpu().numpy()
            embs.append(emb)
    return embs


def _cluster(embs, num_speakers):
    from sklearn.cluster import AgglomerativeClustering

    n = len(embs)
    if n == 1:
        return np.zeros(1, dtype=int)
    X = np.stack(embs)
    if num_speakers is not None:
        k = max(1, min(num_speakers, n))
        if k == 1:
            return np.zeros(n, dtype=int)
        return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
    return AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=CLUSTER_THRESHOLD,
    ).fit_predict(X)


def _build_turns(chunks, labels):
    turns = []
    for spk in np.unique(labels):
        idxs = sorted((i for i, l in enumerate(labels) if l == spk), key=lambda i: chunks[i][0])
        s, e = chunks[idxs[0]]
        for i in idxs[1:]:
            cs, ce = chunks[i]
            if (cs - e) / SAMPLE_RATE < TURN_MERGE_GAP:
                e = max(e, ce)
            else:
                turns.append((int(spk), s / SAMPLE_RATE, e / SAMPLE_RATE))
                s, e = cs, ce
        turns.append((int(spk), s / SAMPLE_RATE, e / SAMPLE_RATE))
    turns.sort(key=lambda t: t[1])
    return turns


def _extract_refs(audio, chunks, labels, workdir):
    import soundfile as sf

    refs = {}
    for spk in np.unique(labels):
        idxs = sorted((i for i, l in enumerate(labels) if l == spk), key=lambda i: chunks[i][0])
        parts = []
        for i in idxs:
            s, e = chunks[i]
            if parts:
                parts.append(np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32))
            parts.append(audio[s:e])
        ref = np.concatenate(parts) if parts else audio[chunks[idxs[0]][0]:chunks[idxs[0]][1]]
        path = os.path.join(workdir, f"ref_speaker_{spk}.wav")
        sf.write(path, ref, SAMPLE_RATE)
        refs[int(spk)] = path
    return refs


def diarize(audio_path, workdir, device="cpu", num_speakers=None):
    os.makedirs(workdir, exist_ok=True)
    audio = _read_mono_16k(audio_path)
    chunks = _vad_chunks(audio)
    if not chunks:
        raise ValueError("Diarizasyon için konuşma tespit edilemedi")
    embs = _embeddings(audio, chunks, device)
    labels = _cluster(embs, num_speakers)
    turns = _build_turns(chunks, labels)
    refs = _extract_refs(audio, chunks, labels, workdir)
    return turns, refs


def assign_segments(segments, turns):
    turn_times = {spk: [(s, e) for spk2, s, e in turns if spk2 == spk] for spk in {t[0] for t in turns}}
    total = {spk: sum(e - s for s, e in tts) for spk, tts in turn_times.items()}
    if not total:
        return [0] * len(segments)

    def overlap(spk, s, e):
        return sum(max(0.0, min(e, ce) - max(s, cs)) for cs, ce in turn_times[spk])

    assignments = []
    for seg in segments:
        scores = {spk: overlap(spk, seg.start, seg.end) for spk in turn_times}
        if max(scores.values(), default=0) > 0:
            assignments.append(max(scores, key=scores.get))
        else:
            assignments.append(max(total, key=total.get))
    return assignments
