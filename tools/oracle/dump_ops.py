#!/usr/bin/env python3
"""Lists the operators, tensors, options, and quantization of one or more .tflite files.

Usage: python dump_ops.py testdata/models/v2/*.tflite [-v]
Without -v: summary per model (operator count, dtypes).
With -v: detail of each operator with its inputs/outputs and options.
"""
import sys
from collections import Counter

import numpy as np
import tflite

TT = {v: k for k, v in tflite.TensorType.__dict__.items() if not k.startswith("_")}
ACT = {v: k for k, v in tflite.ActivationFunctionType.__dict__.items() if not k.startswith("_")}
PAD = {v: k for k, v in tflite.Padding.__dict__.items() if not k.startswith("_")}


def op_name(model, op):
    oc = model.OperatorCodes(op.OpcodeIndex())
    code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
    name = tflite.opcode2name(code)
    if code == tflite.BuiltinOperator.CUSTOM:
        name = f"CUSTOM:{oc.CustomCode().decode()}"
    return name, oc.Version()


def options(name, op):
    bo = op.BuiltinOptions()
    if bo is None:
        return ""
    if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        o = tflite.Conv2DOptions() if name == "CONV_2D" else tflite.DepthwiseConv2DOptions()
        o.Init(bo.Bytes, bo.Pos)
        s = (f"pad={PAD[o.Padding()]} stride=({o.StrideH()},{o.StrideW()}) "
             f"dil=({o.DilationHFactor()},{o.DilationWFactor()}) act={ACT[o.FusedActivationFunction()]}")
        if name == "DEPTHWISE_CONV_2D":
            s += f" mult={o.DepthMultiplier()}"
        return s
    if name == "FULLY_CONNECTED":
        o = tflite.FullyConnectedOptions(); o.Init(bo.Bytes, bo.Pos)
        return f"act={ACT[o.FusedActivationFunction()]} keep_num_dims={o.KeepNumDims()}"
    if name == "STRIDED_SLICE":
        o = tflite.StridedSliceOptions(); o.Init(bo.Bytes, bo.Pos)
        return (f"begin_mask={o.BeginMask()} end_mask={o.EndMask()} ellipsis={o.EllipsisMask()} "
                f"new_axis={o.NewAxisMask()} shrink={o.ShrinkAxisMask()}")
    if name == "CONCATENATION":
        o = tflite.ConcatenationOptions(); o.Init(bo.Bytes, bo.Pos)
        return f"axis={o.Axis()} act={ACT[o.FusedActivationFunction()]}"
    if name == "SPLIT_V":
        o = tflite.SplitVOptions(); o.Init(bo.Bytes, bo.Pos)
        return f"num_splits={o.NumSplits()}"
    if name == "CALL_ONCE":
        o = tflite.CallOnceOptions(); o.Init(bo.Bytes, bo.Pos)
        return f"init_subgraph={o.InitSubgraphIndex()}"
    if name == "VAR_HANDLE":
        o = tflite.VarHandleOptions(); o.Init(bo.Bytes, bo.Pos)
        return f"container={o.Container()!r} shared_name={o.SharedName()!r}"
    return ""


def tensor_desc(model, g, i):
    if i < 0:
        return "-"
    t = g.Tensors(i)
    shape = list(map(int, t.ShapeAsNumpy())) if t.ShapeLength() else []
    s = f"#{i} {t.Name().decode()[:48]}:{TT[t.Type()]}{shape}"
    q = t.Quantization()
    if q is not None and q.ScaleLength():
        s += f" q(scales={q.ScaleLength()} zp={q.ZeroPoint(0)} axis={q.QuantizedDimension()})"
    b = model.Buffers(t.Buffer())
    if b.DataLength():
        s += " CONST"
        if t.Type() == tflite.TensorType.INT32 and b.DataLength() <= 16:
            s += str(np.frombuffer(b.DataAsNumpy().tobytes(), dtype=np.int32).tolist())
    return s


def main():
    verbose = "-v" in sys.argv
    paths = [p for p in sys.argv[1:] if p != "-v"]
    for path in paths:
        model = tflite.Model.GetRootAsModel(open(path, "rb").read(), 0)
        print(f"\n=== {path} version={model.Version()} subgraphs={model.SubgraphsLength()}")
        for s in range(model.SubgraphsLength()):
            g = model.Subgraphs(s)
            cnt = Counter()
            print(f"-- subgraph {s} name={g.Name()} ops={g.OperatorsLength()} tensors={g.TensorsLength()}")
            print("   inputs :", [tensor_desc(model, g, g.Inputs(k)) for k in range(g.InputsLength())])
            print("   outputs:", [tensor_desc(model, g, g.Outputs(k)) for k in range(g.OutputsLength())])
            for j in range(g.OperatorsLength()):
                op = g.Operators(j)
                name, ver = op_name(model, op)
                cnt[f"{name}(v{ver})"] += 1
                if verbose:
                    ins = [tensor_desc(model, g, op.Inputs(k)) for k in range(op.InputsLength())]
                    outs = [tensor_desc(model, g, op.Outputs(k)) for k in range(op.OutputsLength())]
                    print(f"   {j:3d} {name} {options(name, op)}")
                    for x in ins:
                        print(f"        in  {x}")
                    for x in outs:
                        print(f"        out {x}")
            for k, v in sorted(cnt.items()):
                print(f"   {v:3d} x {k}")


if __name__ == "__main__":
    main()
