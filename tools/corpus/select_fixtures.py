#!/usr/bin/env python3
"""Pick a small fixture set from the generated corpus into testdata/audio/.

The fixtures are functional test inputs for wakeword/audio_test.go: a few positives per
wake word (varied voices, one noisy), one near miss per wake word and a short negative
segment. Candidates are filtered with the Go detector (cmd/wakeword) at the manifest
settings so that the committed set is one the current code detects, or rejects, as
expected. The total stays under 2 MB.

Usage: .venv/bin/python select_fixtures.py [--corpus corpus] [--out ../../testdata/audio]
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "testdata" / "models" / "v2"
WORDS = ["alexa", "hey_jarvis", "hey_mycroft", "okay_nabu"]
PREROLL_S = 3.5


def build_cli(tmp: pathlib.Path) -> pathlib.Path:
    exe = tmp / "wakeword"
    subprocess.run(["go", "build", "-o", str(exe), "./cmd/wakeword"], cwd=ROOT, check=True,
                   env={**__import__("os").environ, "CGO_ENABLED": "0"})
    return exe


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def write_wav(path, pcm):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def detections(exe, word, pcm):
    """Run the CLI on preroll + pcm over stdin; return detection times relative to pcm."""
    rng = np.random.default_rng(3)
    pre = rng.integers(-2, 3, int(PREROLL_S * 16000)).astype("<i2")
    raw = np.concatenate([pre, pcm]).tobytes()
    r = subprocess.run([str(exe), "detect", "--raw", "--model", str(MODELS / f"{word}.tflite")],
                       input=raw, capture_output=True, check=True)
    out = []
    for line in r.stdout.decode().splitlines():
        if " detected " in line:
            out.append(float(line.split()[0][2:-1]) - PREROLL_S)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default=str(ROOT / "testdata" / "audio"))
    a = ap.parse_args()
    corpus = pathlib.Path(a.corpus)
    out = pathlib.Path(a.out)
    manifest = json.load(open(corpus / "manifest.json"))["entries"]
    tmp = pathlib.Path("/tmp") / "wakeword-fixtures"
    tmp.mkdir(exist_ok=True)
    exe = build_cli(tmp)

    if out.exists():
        shutil.rmtree(out)
    chosen = []

    def keep(e, name, expect):
        pcm = read_wav(corpus / e["file"])
        for word in WORDS:
            got = detections(exe, word, pcm)
            if word == e["word"] or (e["subset"] == "negative"):
                if expect and word == e["word"]:
                    ok = len(got) == 1 and e["expected_start_s"] <= got[0] <= e["expected_end_s"] + 1.5
                else:
                    ok = len(got) == 0
                if not ok:
                    return False
        rel = f"{e['subset']}/{name}.wav"
        write_wav(out / rel, pcm)
        chosen.append({"file": rel, "subset": e["subset"], "word": e.get("word") or "",
                       "voice": e.get("voice") or "", "text": e.get("text") or "",
                       "expected_start_s": e.get("expected_start_s", 0.0),
                       "expected_end_s": e.get("expected_end_s", 0.0),
                       "noise": e.get("noise"), "source": e["file"]})
        return True

    for word in WORDS:
        pos = [e for e in manifest if e["subset"] == "positive" and e["word"] == word]
        emb = [e for e in manifest if e["subset"] == "embedded" and e["word"] == word]
        # one clean clip, one noisy clip from another voice, one embedded, one near miss.
        wanted = [("clean", lambda e: e["noise"] is None and "jenny" in e["voice"]),
                  ("noisy", lambda e: e["noise"] is not None and "ryan" in e["voice"])]
        for name, pred in wanted:
            for e in pos:
                if pred(e) and keep(e, f"{word}_{name}", True):
                    break
        for e in emb:
            if e["noise"] is None and keep(e, f"{word}_embedded", True):
                break
        nm = [e for e in manifest if e["subset"] == "near_miss" and e["word"] == word and e["noise"] is None]
        for e in nm:
            if keep(e, f"{word}_near_miss", False):
                break

    # A short negative: 10 s cut from a negative file, silent for all four models.
    for e in manifest:
        if e["subset"] != "negative" or e["noise"] is not None:
            continue
        pcm = read_wav(corpus / e["file"])[: 10 * 16000]
        segs = [s["text"] for s in e["segments"] if s["end_s"] <= 10.0]
        if not segs:
            continue
        if all(len(detections(exe, w, pcm)) == 0 for w in WORDS):
            rel = "negative/speech_10s.wav"
            write_wav(out / rel, pcm)
            chosen.append({"file": rel, "subset": "negative", "word": "", "voice": e["voice"],
                           "text": " ".join(segs), "expected_start_s": 0.0, "expected_end_s": 0.0,
                           "noise": None, "source": e["file"]})
            break

    (out / "manifest.json").write_text(json.dumps(chosen, indent=1))
    total = sum(p.stat().st_size for p in out.rglob("*.wav"))
    print(f"{len(chosen)} fixtures, {total / 1024:.0f} KB")
    for c in chosen:
        print(f"  {c['file']:40s} {c['voice']:28s} {c['text']}")


if __name__ == "__main__":
    main()
