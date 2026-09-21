package runtime

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/tgallice/wakeword-go/tflite"
)

// validate enforces the closed operator set and the structural invariants the kernels rely on.
// Unsupported operators are all collected before failing, so the error names every one of them.
func (m *Model) validate() error {
	var unsupported []string
	for si := range m.Subgraphs {
		sg := &m.Subgraphs[si]
		for oi := range sg.Operators {
			op := &sg.Operators[oi]
			spec, ok := supportedOps[op.Code]
			if !ok {
				unsupported = append(unsupported, fmt.Sprintf("subgraph %d operator %d: %s (v%d)", si, oi, op.Name, op.Version))
				continue
			}
			if !containsInt32(spec.versions, int32(op.Version)) {
				unsupported = append(unsupported, fmt.Sprintf("subgraph %d operator %d: %s version %d (supported: %v)",
					si, oi, op.Name, op.Version, spec.versions))
			}
		}
	}
	if len(unsupported) > 0 {
		return fmt.Errorf("unsupported operators: %s", strings.Join(unsupported, "; "))
	}
	for si := range m.Subgraphs {
		sg := &m.Subgraphs[si]
		for ti := range sg.Tensors {
			t := &sg.Tensors[ti]
			if (t.DType == Int8 || t.DType == UInt8) && t.Quant == nil {
				return fmt.Errorf("subgraph %d: tensor %d (%s) is %s without quantization parameters", si, ti, t.Name, t.DType)
			}
			if t.Quant != nil && t.Quant.PerChannel() {
				qd := t.Quant.QuantizedDimension
				if qd < 0 || qd >= len(t.Shape) || len(t.Quant.Scales) != t.Shape[qd] {
					return fmt.Errorf("subgraph %d: tensor %d (%s): %d per-channel scales do not match dimension %d of shape %v",
						si, ti, t.Name, len(t.Quant.Scales), qd, t.Shape)
				}
			}
			if t.DType == Resource && t.IsConst() {
				return fmt.Errorf("subgraph %d: tensor %d (%s): resource tensor with constant data", si, ti, t.Name)
			}
		}
		for oi := range sg.Operators {
			if err := sg.validateOperator(m, oi); err != nil {
				return fmt.Errorf("subgraph %d: operator %d (%s): %w", si, oi, sg.Operators[oi].Name, err)
			}
		}
	}
	main := m.Main()
	if len(main.Inputs) != 1 || len(main.Outputs) != 1 {
		return fmt.Errorf("main subgraph has %d inputs and %d outputs, expected 1 and 1", len(main.Inputs), len(main.Outputs))
	}
	for vi := range m.Variables {
		v := &m.Variables[vi]
		if v.DType == 0 {
			return fmt.Errorf("variable %q is never read or assigned", v.SharedName)
		}
	}
	if m.InitSubgraph >= 0 {
		for oi := range m.Subgraphs[m.InitSubgraph].Operators {
			op := &m.Subgraphs[m.InitSubgraph].Operators[oi]
			if op.Code != tflite.BuiltinOperatorVAR_HANDLE && op.Code != tflite.BuiltinOperatorASSIGN_VARIABLE {
				return fmt.Errorf("init subgraph %d: operator %d (%s) is not VAR_HANDLE or ASSIGN_VARIABLE", m.InitSubgraph, oi, op.Name)
			}
		}
	}
	return nil
}

// validateOperator checks arity, dtypes and options of one operator against the usage the
// kernels support (docs/spec.md section 5).
func (sg *Subgraph) validateOperator(m *Model, oi int) error {
	op := &sg.Operators[oi]
	in := func(k int) *TensorInfo {
		if k >= len(op.Inputs) || op.Inputs[k] < 0 {
			return nil
		}
		return &sg.Tensors[op.Inputs[k]]
	}
	out := func(k int) *TensorInfo {
		if k >= len(op.Outputs) {
			return nil
		}
		return &sg.Tensors[op.Outputs[k]]
	}
	arity := func(nin, nout int) error {
		if len(op.Inputs) != nin || len(op.Outputs) != nout {
			return fmt.Errorf("expected %d inputs and %d outputs, got %d and %d", nin, nout, len(op.Inputs), len(op.Outputs))
		}
		for k, idx := range op.Inputs {
			if idx < 0 {
				return fmt.Errorf("input %d is absent", k)
			}
		}
		return nil
	}
	switch op.Code {
	case tflite.BuiltinOperatorCONV_2D, tflite.BuiltinOperatorDEPTHWISE_CONV_2D:
		if err := arity(3, 1); err != nil {
			return err
		}
		if err := checkDTypes(in(0), Int8, in(1), Int8, in(2), Int32, out(0), Int8); err != nil {
			return err
		}
		if len(in(0).Shape) != 4 || len(in(1).Shape) != 4 || len(out(0).Shape) != 4 {
			return errors.New("expected 4-D input, filter and output")
		}
		if !in(1).IsConst() || !in(2).IsConst() {
			return errors.New("filter and bias must be constant")
		}
		var dil [2]int
		var act Activation
		if op.Code == tflite.BuiltinOperatorCONV_2D {
			o := op.Options.(*Conv2DOptions)
			dil, act = [2]int{o.DilationH, o.DilationW}, o.Activation
			if o.StrideH < 1 || o.StrideW < 1 {
				return fmt.Errorf("invalid stride (%d,%d)", o.StrideH, o.StrideW)
			}
		} else {
			o := op.Options.(*DepthwiseConv2DOptions)
			dil, act = [2]int{o.DilationH, o.DilationW}, o.Activation
			if o.StrideH < 1 || o.StrideW < 1 {
				return fmt.Errorf("invalid stride (%d,%d)", o.StrideH, o.StrideW)
			}
			if o.DepthMultiplier != 1 {
				return fmt.Errorf("depth multiplier %d, only 1 is supported", o.DepthMultiplier)
			}
		}
		if dil != [2]int{1, 1} {
			return fmt.Errorf("dilation %v, only (1,1) is supported", dil)
		}
		if act != ActivationNone && act != ActivationRelu {
			return fmt.Errorf("fused activation %s, only NONE and RELU are supported", act)
		}
		if !in(1).Quant.PerChannel() {
			return errors.New("filter must use per-channel quantization")
		}
		if len(in(2).Quant.Scales) != len(in(1).Quant.Scales) {
			return fmt.Errorf("bias has %d scales, filter has %d", len(in(2).Quant.Scales), len(in(1).Quant.Scales))
		}
		return nil
	case tflite.BuiltinOperatorFULLY_CONNECTED:
		if err := arity(3, 1); err != nil {
			return err
		}
		if err := checkDTypes(in(0), Int8, in(1), Int8, in(2), Int32, out(0), Int8); err != nil {
			return err
		}
		if !in(1).IsConst() || !in(2).IsConst() {
			return errors.New("weights and bias must be constant")
		}
		if in(1).Quant.PerChannel() {
			return errors.New("per-channel weights are not supported")
		}
		if len(in(1).Shape) != 2 {
			return errors.New("weights must be 2-D")
		}
		o := op.Options.(*FullyConnectedOptions)
		if o.Activation != ActivationNone && o.Activation != ActivationRelu {
			return fmt.Errorf("fused activation %s, only NONE and RELU are supported", o.Activation)
		}
		return nil
	case tflite.BuiltinOperatorLOGISTIC:
		if err := arity(1, 1); err != nil {
			return err
		}
		if err := checkDTypes(in(0), Int8, out(0), Int8); err != nil {
			return err
		}
		// The int8 sigmoid of the reference kernels only produces the [0, 1] range mapped onto
		// [-128, 127] in 1/256 steps (tflite-micro logistic_common.cc, TFLite activations.cc).
		if out(0).Quant.PerChannel() {
			return errors.New("output must have per-tensor quantization")
		}
		if zp := out(0).Quant.ZeroPoints[0]; zp != math.MinInt8 {
			return fmt.Errorf("output zero point %d, the int8 sigmoid requires -128", zp)
		}
		if scale := out(0).Quant.Scales[0]; scale != 1.0/256 {
			return fmt.Errorf("output scale %v, the int8 sigmoid requires 1/256", scale)
		}
		if in(0).NumElements != out(0).NumElements {
			return errors.New("input and output element counts differ")
		}
		return nil
	case tflite.BuiltinOperatorQUANTIZE:
		if err := arity(1, 1); err != nil {
			return err
		}
		if err := checkDTypes(in(0), Int8, out(0), UInt8); err != nil {
			return err
		}
		if in(0).NumElements != out(0).NumElements {
			return errors.New("input and output element counts differ")
		}
		return nil
	case tflite.BuiltinOperatorCONCATENATION:
		if len(op.Inputs) < 1 || len(op.Outputs) != 1 {
			return fmt.Errorf("expected at least 1 input and 1 output, got %d and %d", len(op.Inputs), len(op.Outputs))
		}
		o := op.Options.(*ConcatenationOptions)
		if o.Activation != ActivationNone {
			return fmt.Errorf("fused activation %s is not supported", o.Activation)
		}
		nd := len(out(0).Shape)
		axis := o.Axis
		if axis < 0 {
			axis += nd
		}
		if axis < 0 || axis >= nd {
			return fmt.Errorf("axis %d out of range for %d-D output", o.Axis, nd)
		}
		total := 0
		for k := range op.Inputs {
			t := in(k)
			if t == nil {
				return fmt.Errorf("input %d is absent", k)
			}
			if err := checkDTypes(t, Int8); err != nil {
				return err
			}
			if len(t.Shape) != nd {
				return fmt.Errorf("input %d has %d dimensions, output has %d", k, len(t.Shape), nd)
			}
			for d := range nd {
				if d != axis && t.Shape[d] != out(0).Shape[d] {
					return fmt.Errorf("input %d shape %v does not match output shape %v outside axis %d", k, t.Shape, out(0).Shape, axis)
				}
			}
			total += t.Shape[axis]
			// The reference concatenation requantizes when parameters differ; that path is out
			// of scope, so equality is required (docs/spec.md section 7).
			if !t.Quant.Equal(out(0).Quant) {
				return fmt.Errorf("input %d quantization differs from the output's (requantizing concatenation is not supported)", k)
			}
		}
		if err := checkDTypes(out(0), Int8); err != nil {
			return err
		}
		if total != out(0).Shape[axis] {
			return fmt.Errorf("inputs sum to %d along axis %d, output has %d", total, axis, out(0).Shape[axis])
		}
		return nil
	case tflite.BuiltinOperatorSTRIDED_SLICE:
		if err := arity(4, 1); err != nil {
			return err
		}
		if err := checkDTypes(in(0), Int8, in(1), Int32, in(2), Int32, in(3), Int32, out(0), Int8); err != nil {
			return err
		}
		nd := len(in(0).Shape)
		for k := 1; k <= 3; k++ {
			if !in(k).IsConst() || in(k).NumElements != nd {
				return fmt.Errorf("input %d must be a constant vector of %d elements", k, nd)
			}
		}
		o := op.Options.(*StridedSliceOptions)
		if o.EllipsisMask != 0 || o.NewAxisMask != 0 || o.ShrinkAxisMask != 0 || o.Offset {
			return errors.New("ellipsis, new_axis, shrink_axis masks and offset mode are not supported")
		}
		if !in(0).Quant.Equal(out(0).Quant) {
			return errors.New("input and output quantization differ")
		}
		return nil
	case tflite.BuiltinOperatorSPLIT_V:
		if len(op.Inputs) != 3 || len(op.Outputs) < 1 {
			return fmt.Errorf("expected 3 inputs and at least 1 output, got %d and %d", len(op.Inputs), len(op.Outputs))
		}
		if err := checkDTypes(in(0), Int8, in(1), Int32, in(2), Int32); err != nil {
			return err
		}
		if !in(1).IsConst() || !in(2).IsConst() || in(2).NumElements != 1 {
			return errors.New("size_splits and axis must be constant, axis must be a scalar")
		}
		if in(1).NumElements != len(op.Outputs) {
			return fmt.Errorf("size_splits has %d entries, operator has %d outputs", in(1).NumElements, len(op.Outputs))
		}
		for k := range op.Outputs {
			if err := checkDTypes(out(k), Int8); err != nil {
				return err
			}
			if !out(k).Quant.Equal(in(0).Quant) {
				return fmt.Errorf("output %d quantization differs from the input's", k)
			}
		}
		return nil
	case tflite.BuiltinOperatorRESHAPE:
		if err := arity(2, 1); err != nil {
			return err
		}
		if err := checkDTypes(in(0), Int8, in(1), Int32, out(0), Int8); err != nil {
			return err
		}
		if !in(1).IsConst() {
			return errors.New("shape must be constant")
		}
		if in(0).NumElements != out(0).NumElements {
			return errors.New("input and output element counts differ")
		}
		if !in(0).Quant.Equal(out(0).Quant) {
			return errors.New("input and output quantization differ")
		}
		if err := checkReshapeTarget(in(1), out(0)); err != nil {
			return err
		}
		return nil
	case tflite.BuiltinOperatorVAR_HANDLE:
		if err := arity(0, 1); err != nil {
			return err
		}
		return checkDTypes(out(0), Resource)
	case tflite.BuiltinOperatorREAD_VARIABLE:
		if err := arity(1, 1); err != nil {
			return err
		}
		return checkDTypes(in(0), Resource, out(0), Int8)
	case tflite.BuiltinOperatorASSIGN_VARIABLE:
		if err := arity(2, 0); err != nil {
			return err
		}
		return checkDTypes(in(0), Resource, in(1), Int8)
	case tflite.BuiltinOperatorCALL_ONCE:
		if err := arity(0, 0); err != nil {
			return err
		}
		o := op.Options.(*CallOnceOptions)
		if o.InitSubgraphIndex <= 0 || o.InitSubgraphIndex >= len(m.Subgraphs) {
			return fmt.Errorf("init subgraph index %d out of range", o.InitSubgraphIndex)
		}
		if sg.Index != 0 {
			return errors.New("CALL_ONCE outside the main subgraph")
		}
		if m.InitSubgraph >= 0 {
			return errors.New("more than one CALL_ONCE in the main subgraph")
		}
		m.InitSubgraph = o.InitSubgraphIndex
		return nil
	default:
		return fmt.Errorf("no validation rule for %s", op.Name)
	}
}

// checkReshapeTarget checks that the constant shape tensor of a RESHAPE, with at most one -1
// wildcard resolved against the element count, equals the output shape.
func checkReshapeTarget(shape, out *TensorInfo) error {
	want := int32sOf(shape.Const)
	if len(want) != len(out.Shape) {
		return fmt.Errorf("shape parameter has %d dimensions, output has %d", len(want), len(out.Shape))
	}
	wild := -1
	known := 1
	for i, d := range want {
		switch {
		case d == -1 && wild < 0:
			wild = i
		case d < 0:
			return fmt.Errorf("shape parameter %v has more than one wildcard", want)
		default:
			known *= int(d)
		}
	}
	for i, d := range want {
		v := int(d)
		if i == wild {
			if known == 0 {
				return fmt.Errorf("shape parameter %v cannot resolve its wildcard", want)
			}
			v = out.NumElements / known
		}
		if v != out.Shape[i] {
			return fmt.Errorf("shape parameter %v does not match output shape %v", want, out.Shape)
		}
	}
	return nil
}

// int32sOf decodes a little-endian int32 constant buffer.
func int32sOf(b []byte) []int32 {
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24)
	}
	return out
}

// checkDTypes takes (tensor, dtype) pairs and checks each tensor is present with that dtype.
func checkDTypes(pairs ...any) error {
	for i := 0; i+1 < len(pairs); i += 2 {
		t, _ := pairs[i].(*TensorInfo)
		want := pairs[i+1].(DType)
		if t == nil {
			return fmt.Errorf("operand %d is absent", i/2)
		}
		if t.DType != want {
			return fmt.Errorf("tensor %s is %s, expected %s", t.Name, t.DType, want)
		}
	}
	return nil
}

func containsInt32(xs []int32, x int32) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
