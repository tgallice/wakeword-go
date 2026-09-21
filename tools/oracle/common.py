"""Utilities shared by the oracles."""
import base64
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "testdata" / "models" / "v2"
PARITY_DIR = ROOT / "testdata" / "parity"
FRONTEND_DIR = ROOT / "testdata" / "frontend"

MODEL_NAMES = ["alexa", "hey_jarvis", "hey_mycroft", "okay_nabu", "vad"]

DTYPE_NAMES = {
    np.dtype("int8"): "int8",
    np.dtype("uint8"): "uint8",
    np.dtype("int32"): "int32",
    np.dtype("int16"): "int16",
    np.dtype("float32"): "float32",
}


def b64(arr) -> str:
    """Encode a numpy array to base64 (raw bytes, little-endian, C order)."""
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == ">":
        a = a.byteswap().view(a.dtype.newbyteorder("<"))
    return base64.b64encode(a.tobytes()).decode("ascii")


def dump_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
