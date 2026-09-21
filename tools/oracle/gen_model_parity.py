#!/usr/bin/env python3
"""Generates the golden parity vectors for the microWakeWord v2 models.

For each model in testdata/models/v2, produces testdata/parity/<name>.json containing:
  - the input/output contract (index, shape, dtype, quantization);
  - sequences of int8 [1,3,40] frames with the expected uint8 output at each step,
    the streaming state being kept between steps and reset between sequences;
  - full traces (value of all non-constant tensors after each step)
    to test each operator in isolation.

The BUILTIN_REF resolver is mandatory: these are the reference kernels, the ones that
tflite-micro uses. The optimized kernels sometimes give a result that differs by
one unit, which would make strict parity impossible.

Usage: python gen_model_parity.py [--sequences 50] [--steps 40] [--trace-steps 8] [models...]
"""
import argparse

import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from common import DTYPE_NAMES, MODELS_DIR, MODEL_NAMES, PARITY_DIR, b64, dump_json


def constant_tensor_indices(path):
    """Indices of tensors in subgraph 0 that have a constant buffer (weights, parameters)."""
    model = tflite.Model.GetRootAsModel(open(path, "rb").read(), 0)
    g = model.Subgraphs(0)
    out = set()
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        if model.Buffers(t.Buffer()).DataLength() > 0:
            out.add(i)
    return out


def quant_desc(detail):
    qp = detail["quantization_parameters"]
    return {
        "scales": [float(x) for x in qp["scales"]],
        "zero_points": [int(x) for x in qp["zero_points"]],
        "quantized_dimension": int(qp["quantized_dimension"]),
    }


def io_desc(detail):
    return {
        "index": int(detail["index"]),
        "name": detail["name"],
        "shape": [int(x) for x in detail["shape"]],
        "dtype": DTYPE_NAMES[np.dtype(detail["dtype"])],
        "quantization": quant_desc(detail),
    }


def make_sequences(rng, n_random, steps):
    """Input sequences: a few degenerate cases, then random sequences."""
    shape = (steps, 1, 3, 40)
    seqs = [
        ("zeros_real", np.full(shape, -128, dtype=np.int8)),   # real value 0 (zero point)
        ("min", np.full(shape, -128, dtype=np.int8)),
        ("max", np.full(shape, 127, dtype=np.int8)),
        ("ramp", np.broadcast_to(np.arange(-128, 127, 255 / steps, dtype=np.float32)[:steps, None, None, None],
                                 shape).astype(np.int8)),
    ]
    for k in range(n_random):
        seqs.append((f"random_{k}", rng.integers(-128, 128, shape, dtype=np.int8)))
    # Realistic sequences: speech features mostly occupy the lower half of the range.
    for k in range(max(1, n_random // 4)):
        low = rng.integers(-128, 0, shape, dtype=np.int8)
        seqs.append((f"lowrange_{k}", low))
    return seqs


def run_model(name, n_random, steps, trace_steps, seed):
    path = MODELS_DIR / f"{name}.tflite"
    consts = constant_tensor_indices(path)
    it = Interpreter(model_path=str(path),
                     experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                     experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    assert inp["dtype"] == np.int8 and out["dtype"] == np.uint8

    dynamic = [d for d in it.get_tensor_details()
               if d["index"] not in consts and d["dtype"] not in (np.dtype("O"),)
               and DTYPE_NAMES.get(np.dtype(d["dtype"])) is not None]

    rng = np.random.default_rng(seed)
    sequences = []
    traces = []
    for si, (label, frames) in enumerate(make_sequences(rng, n_random, steps)):
        it.reset_all_variables()
        rec = {"name": label, "steps": []}
        for k in range(steps):
            it.set_tensor(inp["index"], frames[k])
            it.invoke()
            y = int(it.get_tensor(out["index"])[0, 0])
            rec["steps"].append({"input": b64(frames[k]), "output": y})
            if si == 0 or (label.startswith("random") and k < trace_steps and si < 6):
                tensors = {}
                for d in dynamic:
                    try:
                        v = it.get_tensor(d["index"])
                    except ValueError:
                        continue
                    tensors[str(d["index"])] = {
                        "name": d["name"],
                        "shape": [int(x) for x in v.shape],
                        "dtype": DTYPE_NAMES[v.dtype],
                        "data": b64(v),
                    }
                traces.append({"sequence": si, "step": k, "tensors": tensors})
        sequences.append(rec)

    doc = {
        "model": f"{name}.tflite",
        "generator": "tools/oracle/gen_model_parity.py",
        "resolver": "BUILTIN_REF",
        "seed": seed,
        "input": io_desc(inp),
        "output": io_desc(out),
        "tensors": {str(d["index"]): {"name": d["name"], "dtype": DTYPE_NAMES[np.dtype(d["dtype"])],
                                      "quantization": quant_desc(d)} for d in dynamic},
        "sequences": sequences,
        "traces": traces,
    }
    dump_json(PARITY_DIR / f"{name}.json", doc)
    nz = sum(1 for s in sequences for st in s["steps"] if st["output"] != 0)
    print(f"   {name}: {len(sequences)} sequences x {steps} steps, {len(traces)} traces, "
          f"{nz} non-zero outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", type=int, default=50, help="number of random sequences")
    ap.add_argument("--steps", type=int, default=40, help="steps (invocations) per sequence")
    ap.add_argument("--trace-steps", type=int, default=8, help="steps traced in detail per random sequence")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("models", nargs="*", default=MODEL_NAMES)
    a = ap.parse_args()
    for name in a.models:
        run_model(name, a.sequences, a.steps, a.trace_steps, a.seed)


if __name__ == "__main__":
    main()
