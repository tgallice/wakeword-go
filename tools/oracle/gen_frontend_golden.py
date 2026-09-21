#!/usr/bin/env python3
"""Generates the golden vectors of the microfrontend (audio features) via pymicro-features.

pymicro-features embeds the C code of tensorflow/lite/experimental/microfrontend/lib
with exactly the configuration of microWakeWord and ESPHome (see docs/spec.md).
It returns the features as float (uint16 * 0.0390625); they are converted back to exact
uint16 values (the factor is 5/128, the product is exact in double precision).

Output: testdata/frontend/<signal>.json with:
  - pcm: little-endian int16 in base64 (16 kHz mono);
  - chunks: for each submitted block of 160 samples, samples_read and, if a frame
    is produced, the 40 uint16 features and their int8 conversion (ESPHome formula).

Usage: python gen_frontend_golden.py [--seconds 2.0]
"""
import argparse

import numpy as np
from pymicro_features import MicroFrontend

from common import FRONTEND_DIR, b64, dump_json

SAMPLE_RATE = 16000
CHUNK = 160            # 10 ms, frame step
FLOAT_SCALE = 0.0390625  # 1/25.6, applied by pymicro-features


def to_int8_esphome(values_u16):
    """ESPHome conversion (components/micro_wake_word/micro_wake_word.cpp, generate_features_):
    input = (feature * 256 + 333) / 666 - 128, clamped to [-128, 127], in integer arithmetic."""
    v = (values_u16.astype(np.int64) * 256 + 333) // 666 - 128
    return np.clip(v, -128, 127).astype(np.int8)


def signals(seconds, rng):
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    out = {}
    out["silence"] = np.zeros(n, dtype=np.int16)
    out["sine_440"] = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    out["chirp_100_7000"] = (6000 * np.sin(2 * np.pi * (100 * t + (6900 / (2 * seconds)) * t * t))).astype(np.int16)
    out["white_noise"] = rng.normal(0, 2500, n).clip(-32768, 32767).astype(np.int16)
    imp = np.zeros(n, dtype=np.int16); imp[::1600] = 20000
    out["impulses"] = imp
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    out["am_noise_bursts"] = (env * rng.normal(0, 6000, n)).clip(-32768, 32767).astype(np.int16)
    out["full_scale_square"] = (np.where(np.sin(2 * np.pi * 200 * t) >= 0, 32767, -32768)).astype(np.int16)
    out["dc_offset"] = np.full(n, 1000, dtype=np.int16)
    return out


def run(name, pcm):
    fe = MicroFrontend()
    chunks = []
    for start in range(0, len(pcm) - CHUNK + 1, CHUNK):
        block = pcm[start:start + CHUNK]
        res = fe.process_samples(block.tobytes())
        rec = {"samples_read": int(res.samples_read)}
        if res.features:
            f = np.asarray(res.features, dtype=np.float64) / FLOAT_SCALE
            u16 = np.rint(f).astype(np.uint16)
            assert np.all(np.abs(f - u16) < 1e-6), "inexact uint16 reconversion"
            assert len(u16) == 40
            rec["features_u16"] = u16.tolist()
            rec["features_int8"] = to_int8_esphome(u16).tolist()
        chunks.append(rec)
    doc = {
        "signal": name,
        "generator": "tools/oracle/gen_frontend_golden.py",
        "sample_rate": SAMPLE_RATE,
        "chunk_samples": CHUNK,
        "pcm": b64(pcm),
        "chunks": chunks,
    }
    dump_json(FRONTEND_DIR / f"{name}.json", doc)
    frames = sum(1 for c in chunks if "features_u16" in c)
    mx = max((max(c["features_u16"]) for c in chunks if "features_u16" in c), default=0)
    print(f"   {name}: {len(chunks)} blocks, {frames} frames, max feature {mx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260903)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for name, pcm in signals(a.seconds, rng).items():
        run(name, pcm)


if __name__ == "__main__":
    main()
