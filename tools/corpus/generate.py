#!/usr/bin/env python3
"""Generate the synthetic validation corpus with Piper TTS.

Everything is deterministic for a given seed. Output goes to corpus/<subset>/*.wav (16 kHz,
mono, 16-bit) plus corpus/manifest.json. The voices are downloaded once into corpus/voices/.

Subsets:
  positive   the wake word alone (per voice, speech rate, noise scale), augmented
  embedded   the wake word inside a longer utterance, with the expected time span
  near_miss  phonetic neighbours of a wake word, alone, augmented
  negative   long files of sentences without any wake word
  noise      noise-only and music-like synthetic signals

Usage: .venv/bin/python generate.py [--out corpus] [--seed 1] [--jobs 8]
"""
import argparse
import json
import pathlib
import wave
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from scipy.signal import resample_poly

import sentences as S

SR = 16000
VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICES = {
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_US-amy-medium": "en/en_US/amy/medium",
    "en_US-ryan-medium": "en/en_US/ryan/medium",
    "en_US-joe-medium": "en/en_US/joe/medium",
    "en_GB-alan-medium": "en/en_GB/alan/medium",
    "en_GB-jenny_dioco-medium": "en/en_GB/jenny_dioco/medium",
    "en_US-libritts_r-medium": "en/en_US/libritts_r/medium",
}
# Speaker ids used for the multi-speaker voice.
LIBRITTS_SPEAKERS = [0, 17, 93, 250]

LENGTH_SCALES = [0.8, 1.0, 1.2]
NOISE_SCALES = [0.5, 0.8]


def download_voices(vdir: pathlib.Path):
    import urllib.request
    vdir.mkdir(parents=True, exist_ok=True)
    for name, path in VOICES.items():
        for ext in ("onnx", "onnx.json"):
            dst = vdir / f"{name}.{ext}"
            if dst.exists():
                continue
            url = f"{VOICE_BASE}/{path}/{name}.{ext}"
            print("download", url)
            urllib.request.urlretrieve(url, dst)


@lru_cache(maxsize=None)
def load_voice(path: str):
    from piper import PiperVoice
    return PiperVoice.load(path)


def synth(voice_path: str, text: str, speaker, length_scale, noise_scale) -> np.ndarray:
    """Synthesize text and return float32 samples at 16 kHz in [-1, 1]."""
    from piper.config import SynthesisConfig
    v = load_voice(voice_path)
    cfg = SynthesisConfig(speaker_id=speaker, length_scale=length_scale,
                          noise_scale=noise_scale, normalize_audio=True)
    parts = [c.audio_float_array.astype(np.float32) for c in v.synthesize(text, cfg)]
    rate = v.config.sample_rate
    audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    if rate != SR:
        from math import gcd
        g = gcd(SR, rate)
        audio = resample_poly(audio, SR // g, rate // g).astype(np.float32)
    return trim_silence(audio)


def trim_silence(a: np.ndarray, thresh=1e-3, pad=0.05) -> np.ndarray:
    idx = np.nonzero(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    p = int(pad * SR)
    return a[max(0, idx[0] - p): min(len(a), idx[-1] + p)]


def db(x):
    return 10 ** (x / 20)


def make_noise(rng, kind: str, n: int) -> np.ndarray:
    if kind == "white":
        return rng.standard_normal(n).astype(np.float32)
    if kind == "brown":
        w = rng.standard_normal(n)
        b = np.cumsum(w)
        b -= np.linspace(b[0], b[-1], n)  # remove drift
        return (b / (np.abs(b).max() + 1e-9)).astype(np.float32)
    if kind == "pink":
        w = rng.standard_normal(n)
        f = np.fft.rfft(w)
        freqs = np.arange(len(f)) + 1.0
        p = np.fft.irfft(f / np.sqrt(freqs), n)
        return (p / (np.abs(p).max() + 1e-9)).astype(np.float32)
    raise ValueError(kind)


def mix_at_snr(rng, speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sp = np.mean(speech ** 2) + 1e-12
    n = noise[:len(speech)] if len(noise) >= len(speech) else np.resize(noise, len(speech))
    npow = np.mean(n ** 2) + 1e-12
    n = n * np.sqrt(sp / (npow * db(snr_db) ** 2))
    return speech + n


def augment(rng, speech: np.ndarray, babble_pool, with_noise: bool):
    """Gain, leading and trailing silence, optional background at 10 to 20 dB SNR.

    Returns (audio, info, speech_offset_seconds)."""
    gain_db = float(rng.uniform(-12, 0))
    lead = float(rng.uniform(0.5, 1.5))
    trail = float(rng.uniform(0.5, 1.5))
    x = speech * db(gain_db)
    info = {"gain_db": round(gain_db, 2), "lead_s": round(lead, 3), "trail_s": round(trail, 3),
            "noise": None, "snr_db": None}
    y = np.concatenate([np.zeros(int(lead * SR), np.float32), x, np.zeros(int(trail * SR), np.float32)])
    if with_noise:
        kind = rng.choice(["white", "brown", "pink", "babble"])
        snr = float(rng.uniform(10, 20))
        if kind == "babble":
            bab = babble_pool[rng.integers(len(babble_pool))]
            noise = np.resize(bab, len(y))
        else:
            noise = make_noise(rng, str(kind), len(y))
        y = mix_at_snr(rng, y, noise, snr)
        info.update(noise=str(kind), snr_db=round(snr, 1))
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32), info, lead


def write_wav(path: pathlib.Path, audio: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(audio * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def voice_variants():
    """(label, voice file name, speaker id) for every voice used."""
    out = []
    for name in VOICES:
        if name.startswith("en_US-libritts_r"):
            for spk in LIBRITTS_SPEAKERS:
                out.append((f"{name}#{spk}", name, spk))
        else:
            out.append((name, name, None))
    return out


# ---------------------------------------------------------------------------------------
# Workers (run in subprocesses; each loads its voices lazily and caches them)

def job_positive(args):
    out, vdir, word, text, label, vname, spk, ls, ns, seed, with_noise, babble = args
    rng = np.random.default_rng(seed)
    sp = synth(str(vdir / f"{vname}.onnx"), text, spk, ls, ns)
    y, info, lead = augment(rng, sp, babble, with_noise)
    rel = f"positive/{word}/{label.replace('#', '_')}_ls{ls}_ns{ns}_{seed}.wav"
    write_wav(out / rel, y)
    return {"file": rel, "subset": "positive", "word": word, "text": text, "voice": label,
            "length_scale": ls, "noise_scale": ns, "duration_s": round(len(y) / SR, 3),
            "expected_start_s": round(lead, 3), "expected_end_s": round(lead + len(sp) / SR, 3), **info}


def job_embedded(args):
    out, vdir, word, text, prefix, suffix, label, vname, spk, ls, ns, seed, with_noise, babble = args
    rng = np.random.default_rng(seed)
    vp = str(vdir / f"{vname}.onnx")
    gap = np.zeros(int(0.25 * SR), np.float32)
    parts, start, end = [], 0.0, 0.0
    if prefix:
        p = synth(vp, prefix, spk, ls, ns)
        parts += [p, gap]
        start = (len(p) + len(gap)) / SR
    w = synth(vp, text, spk, ls, ns)
    parts.append(w)
    end = start + len(w) / SR
    if suffix:
        s_ = synth(vp, suffix, spk, ls, ns)
        parts += [gap, s_]
    sp = np.concatenate(parts)
    y, info, lead = augment(rng, sp, babble, with_noise)
    rel = f"embedded/{word}/{label.replace('#', '_')}_{seed}.wav"
    write_wav(out / rel, y)
    full = " ".join(x for x in (prefix, text + ("," if suffix else ""), suffix) if x)
    return {"file": rel, "subset": "embedded", "word": word, "text": full, "voice": label,
            "length_scale": ls, "noise_scale": ns, "duration_s": round(len(y) / SR, 3),
            "expected_start_s": round(lead + start, 3), "expected_end_s": round(lead + end, 3), **info}


def job_near_miss(args):
    out, vdir, text, target, label, vname, spk, ls, ns, seed, with_noise, babble = args
    rng = np.random.default_rng(seed)
    sp = synth(str(vdir / f"{vname}.onnx"), text, spk, ls, ns)
    y, info, lead = augment(rng, sp, babble, with_noise)
    slug = text.lower().replace(" ", "_")
    rel = f"near_miss/{target}/{slug}_{label.replace('#', '_')}_{seed}.wav"
    write_wav(out / rel, y)
    return {"file": rel, "subset": "near_miss", "word": target, "text": text, "voice": label,
            "length_scale": ls, "noise_scale": ns, "duration_s": round(len(y) / SR, 3),
            "expected_start_s": round(lead, 3), "expected_end_s": round(lead + len(sp) / SR, 3), **info}


def job_negative(args):
    out, vdir, idx, texts, label, vname, spk, seed, with_noise, babble = args
    rng = np.random.default_rng(seed)
    vp = str(vdir / f"{vname}.onnx")
    parts, segments, pos = [], [], 0
    for t in texts:
        ls = float(rng.choice(LENGTH_SCALES))
        ns = float(rng.choice(NOISE_SCALES))
        sp = synth(vp, t, spk, ls, ns)
        gap = np.zeros(int(rng.uniform(0.3, 1.0) * SR), np.float32)
        segments.append({"text": t, "start_s": pos / SR, "end_s": (pos + len(sp)) / SR})
        parts += [sp, gap]
        pos += len(sp) + len(gap)
    sp = np.concatenate(parts)
    y, info, lead = augment(rng, sp, babble, with_noise)
    for seg in segments:
        seg["start_s"] = round(seg["start_s"] + lead, 3)
        seg["end_s"] = round(seg["end_s"] + lead, 3)
    rel = f"negative/{label.replace('#', '_')}_{idx:03d}.wav"
    write_wav(out / rel, y)
    return {"file": rel, "subset": "negative", "word": None, "text": None, "voice": label,
            "sentences": len(texts), "segments": segments, "duration_s": round(len(y) / SR, 3), **info}


def music_like(rng, seconds: float) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = np.zeros(n, np.float32)
    # Chords: 2 second blocks of 3 to 4 notes with harmonics and an envelope.
    roots = [110, 130.8, 146.8, 164.8, 196, 220]
    for b in range(int(seconds // 2)):
        s, e = b * 2 * SR, min(n, (b + 1) * 2 * SR)
        tt = t[s:e] - t[s]
        env = np.minimum(1, tt / 0.05) * np.exp(-tt / 1.2)
        root = roots[rng.integers(len(roots))]
        for ratio in (1, 1.25, 1.5, 2):
            for h, a in ((1, 1), (2, 0.4), (3, 0.2)):
                y[s:e] += a * 0.08 * env * np.sin(2 * np.pi * root * ratio * h * tt)
    # Drums: kick every 0.5 s, snare offbeat, hats every 0.125 s.
    for k in np.arange(0, seconds, 0.5):
        s = int(k * SR); e = min(n, s + int(0.25 * SR)); tt = t[s:e] - t[s]
        y[s:e] += 0.5 * np.exp(-tt * 18) * np.sin(2 * np.pi * (60 + 80 * np.exp(-tt * 30)) * tt)
    for k in np.arange(0.5, seconds, 1.0):
        s = int(k * SR); e = min(n, s + int(0.15 * SR)); tt = t[s:e] - t[s]
        y[s:e] += 0.25 * np.exp(-tt * 25) * rng.standard_normal(len(tt))
    for k in np.arange(0, seconds, 0.125):
        s = int(k * SR); e = min(n, s + int(0.04 * SR)); tt = t[s:e] - t[s]
        hp = np.diff(rng.standard_normal(len(tt) + 1))
        y[s:e] += 0.08 * np.exp(-tt * 80) * hp
    return (y / (np.abs(y).max() + 1e-9) * 0.7).astype(np.float32)


def job_noise(args):
    out, idx, kind, seconds, seed = args
    rng = np.random.default_rng(seed)
    if kind == "music":
        y = music_like(rng, seconds)
    else:
        y = make_noise(rng, kind, int(seconds * SR)) * db(float(rng.uniform(-30, -10)))
    rel = f"noise/{kind}_{idx:02d}.wav"
    write_wav(out / rel, y)
    return {"file": rel, "subset": "noise", "word": None, "text": None, "voice": None,
            "noise": kind, "duration_s": round(len(y) / SR, 3)}


# ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--negative-minutes", type=float, default=65.0,
                    help="target duration of TTS negatives (across all voices)")
    ap.add_argument("--smoke", action="store_true", help="one job per subset, for a quick check")
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    vdir = out / "voices"
    download_voices(vdir)
    assert not S.check_negatives(), S.check_negatives()

    rng = np.random.default_rng(a.seed)
    variants = voice_variants()
    seed_iter = iter(range(a.seed * 1_000_000, a.seed * 1_000_000 + 10_000_000))

    # Babble pool: a few sentences at 16 kHz from a couple of voices, used as background.
    print("babble pool")
    babble = [synth(str(vdir / "en_US-lessac-medium.onnx"), t, None, 1.0, 0.667) for t in S.BABBLE[:2]]
    babble += [synth(str(vdir / "en_GB-alan-medium.onnx"), t, None, 1.0, 0.667) for t in S.BABBLE[2:]]
    babble = [b / (np.abs(b).max() + 1e-9) for b in babble]

    jobs = []
    # Positives: every variant x length scales x noise scales.
    for word, text in S.WAKE_WORDS.items():
        for label, vname, spk in variants:
            for ls in LENGTH_SCALES:
                for ns in NOISE_SCALES:
                    sd = next(seed_iter)
                    jobs.append(("positive", (out, vdir, word, text, label, vname, spk, ls, ns, sd,
                                              bool(sd % 2), babble)))
    # Embedded: 3 prefix and 3 suffix cases per word per single-speaker voice.
    for word, text in S.WAKE_WORDS.items():
        for vi, (label, vname, spk) in enumerate(variants):
            if spk is not None:
                continue
            for k in range(3):
                sd = next(seed_iter)
                prefix = S.EMBEDDING_PREFIXES[(k + vi) % len(S.EMBEDDING_PREFIXES)]
                jobs.append(("embedded", (out, vdir, word, text, prefix, "", label, vname, spk,
                                          1.0, 0.667, sd, bool(sd % 2), babble)))
                sd = next(seed_iter)
                suffix = S.EMBEDDING_SUFFIXES[(k + vi) % len(S.EMBEDDING_SUFFIXES)]
                jobs.append(("embedded", (out, vdir, word, text, "", suffix, label, vname, spk,
                                          1.0, 0.667, sd, bool(sd % 2), babble)))
    # Near misses: every text x every single-speaker voice x 2 length scales.
    for text, target in S.NEAR_MISSES:
        for label, vname, spk in variants:
            if spk is not None:
                continue
            for ls in (0.9, 1.1):
                sd = next(seed_iter)
                jobs.append(("near_miss", (out, vdir, text, target, label, vname, spk, ls, 0.667, sd,
                                           bool(sd % 2), babble)))
    # Negatives: every variant reads shuffled sentences in chunks of 12 until the target minutes.
    per_variant = a.negative_minutes * 60 / len(variants)
    est_s = 3.6  # rough seconds per sentence including the pause
    for label, vname, spk in variants:
        order = list(rng.permutation(len(S.NEGATIVES)))
        chunks = [order[i:i + 12] for i in range(0, len(order), 12)]
        needed = int(per_variant / (12 * est_s)) + 1
        for idx in range(needed):
            texts = [S.NEGATIVES[j] for j in chunks[idx % len(chunks)]]
            sd = next(seed_iter)
            jobs.append(("negative", (out, vdir, idx, texts, label, vname, spk, sd, bool(sd % 2), babble)))
    # Noise: 10 minutes.
    kinds = ["white", "brown", "pink", "music", "music", "music", "white", "pink", "music", "brown"]
    for idx, kind in enumerate(kinds):
        jobs.append(("noise", (out, idx, kind, 60.0, next(seed_iter))))

    if a.smoke:
        seen, keep = set(), []
        for kind, arg in jobs:
            if kind not in seen:
                seen.add(kind)
                keep.append((kind, arg))
        jobs = keep
    print(f"{len(jobs)} jobs")
    fns = {"positive": job_positive, "embedded": job_embedded, "near_miss": job_near_miss,
           "negative": job_negative, "noise": job_noise}
    manifest = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(fns[kind], arg) for kind, arg in jobs]
        for i, f in enumerate(futs):
            manifest.append(f.result())
            if i % 50 == 0:
                print(f"{i}/{len(jobs)}", manifest[-1]["file"])
    manifest.sort(key=lambda m: m["file"])
    doc = {"generator": "tools/corpus/generate.py", "seed": a.seed, "sample_rate": SR,
           "voices": sorted(VOICES), "entries": manifest}
    (out / "manifest.json").write_text(json.dumps(doc, indent=1))
    by = {}
    for m in manifest:
        s = by.setdefault(m["subset"], [0, 0.0])
        s[0] += 1
        s[1] += m["duration_s"]
    for k, (n, d) in sorted(by.items()):
        print(f"{k:10s} {n:5d} files {d / 60:7.1f} min")


if __name__ == "__main__":
    main()
