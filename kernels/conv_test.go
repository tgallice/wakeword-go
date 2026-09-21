package kernels

import (
	"strings"
	"testing"

	"github.com/tgallice/wakeword-go/runtime"
	"github.com/tgallice/wakeword-go/tflite"
)

// Synthetic single-operator tests with hand-computed expectations. Scales are chosen so the
// requantization multiplier is exactly 1 (identity) or exactly 0.5, which keeps the arithmetic
// checkable by hand while still exercising offsets, padding, strides and activations.

func quant(zeroPoint int32, scales ...float32) *runtime.Quantization {
	zps := make([]int32, len(scales))
	for i := range zps {
		zps[i] = zeroPoint
	}
	return &runtime.Quantization{Scales: scales, ZeroPoints: zps}
}

func perChannel(dim int, scales ...float32) *runtime.Quantization {
	q := quant(0, scales...)
	q.QuantizedDimension = dim
	return q
}

// runConv builds a single CONV_2D (or DEPTHWISE_CONV_2D) model, feeds the input and returns
// the output bytes as int8.
func runArith(t *testing.T, code tflite.BuiltinOperator, opts any, tensors []tensorSpec, input []int8) []int8 {
	t.Helper()
	name := tflite.EnumNamesBuiltinOperator[code]
	it := newSingleOp(t, code, name, opts, tensors, []int{0, 1, 2}, []int{3})
	if err := it.Input(0).SetInt8(input); err != nil {
		t.Fatal(err)
	}
	if err := it.RunOperator(0, 0); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return toInt8(it.Output(0).Data)
}

func assertInt8(t *testing.T, what string, got, want []int8) {
	t.Helper()
	assertBytes(t, what, toBytes(got), toBytes(want))
}

func TestConv2DPointwiseWithOffsets(t *testing.T) {
	// Input zero point 10 (offset -10), output zero point 3, identity scales.
	// Real input (13-10, 7-10) = (3, -3).
	// Channel 0: 2*3 + 1*(-3) + bias 1 = 4, plus output zero point = 7.
	// Channel 1: -1*3 + 4*(-3) + bias -2 = -17, plus output zero point = -14; ReLU floor is
	// the output zero point 3.
	tensors := func(act runtime.Activation) []tensorSpec {
		_ = act
		return []tensorSpec{
			{shape: []int{1, 1, 1, 2}, dtype: runtime.Int8, quant: quant(10, 1)},
			{shape: []int{2, 1, 1, 2}, dtype: runtime.Int8, constInt8: []int8{2, 1, -1, 4}, quant: perChannel(0, 1, 1)},
			{shape: []int{2}, dtype: runtime.Int32, constInt32: []int32{1, -2}, quant: quant(0, 1, 1)},
			{shape: []int{1, 1, 1, 2}, dtype: runtime.Int8, quant: quant(3, 1)},
		}
	}
	for _, tc := range []struct {
		act  runtime.Activation
		want []int8
	}{
		{runtime.ActivationNone, []int8{7, -14}},
		{runtime.ActivationRelu, []int8{7, 3}},
	} {
		t.Run(tc.act.String(), func(t *testing.T) {
			opts := &runtime.Conv2DOptions{Padding: runtime.PaddingValid, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1, Activation: tc.act}
			got := runArith(t, tflite.BuiltinOperatorCONV_2D, opts, tensors(tc.act), []int8{13, 7})
			assertInt8(t, "output", got, tc.want)
		})
	}
}

func TestConv2DTemporalTaps(t *testing.T) {
	// A [1, 0, -1] filter along the time axis on the ramp 1..5 (zero points 0, identity scale).
	filter := tensorSpec{shape: []int{1, 3, 1, 1}, dtype: runtime.Int8, constInt8: []int8{1, 0, -1}, quant: perChannel(0, 1)}
	bias := tensorSpec{shape: []int{1}, dtype: runtime.Int32, constInt32: []int32{0}, quant: quant(0, 1)}
	cases := []struct {
		name   string
		pad    runtime.Padding
		stride int
		in     []int8
		filter []int8
		outLen int
		want   []int8
	}{
		// x[y] - x[y+2] at every valid position.
		{"valid", runtime.PaddingValid, 1, []int8{1, 2, 3, 4, 5}, []int8{1, 0, -1}, 3, []int8{-2, -2, -2}},
		// Leading pad 1: y=0 sees (pad, x0, x1) = -x1 = -2, y=4 sees (x3, x4, pad) = x3 = 4.
		{"same", runtime.PaddingSame, 1, []int8{1, 2, 3, 4, 5}, []int8{1, 0, -1}, 5, []int8{-2, -2, -2, -2, 4}},
		// Stride 3 with a box filter on 1..7: (1+2+3), (4+5+6).
		{"stride 3", runtime.PaddingValid, 3, []int8{1, 2, 3, 4, 5, 6, 7}, []int8{1, 1, 1}, 2, []int8{6, 15}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := filter
			f.constInt8 = tc.filter
			tensors := []tensorSpec{
				{shape: []int{1, len(tc.in), 1, 1}, dtype: runtime.Int8, quant: quant(0, 1)},
				f, bias,
				{shape: []int{1, tc.outLen, 1, 1}, dtype: runtime.Int8, quant: quant(0, 1)},
			}
			opts := &runtime.Conv2DOptions{Padding: tc.pad, StrideH: tc.stride, StrideW: 1, DilationH: 1, DilationW: 1}
			got := runArith(t, tflite.BuiltinOperatorCONV_2D, opts, tensors, tc.in)
			assertInt8(t, "output", got, tc.want)
		})
	}
}

func TestDepthwiseConv2DPerChannelScale(t *testing.T) {
	// Two channels over three time steps. Channel 0 has identity scale and a box filter;
	// channel 1 has filter scale 0.5 (so its accumulator is halved with the reference
	// rounding, ties toward positive) and taps [1, -1, 2] plus bias 1.
	filter := tensorSpec{shape: []int{1, 3, 1, 2}, dtype: runtime.Int8, constInt8: []int8{1, 1, 1, -1, 1, 2}, quant: perChannel(3, 1, 0.5)}
	bias := tensorSpec{shape: []int{2}, dtype: runtime.Int32, constInt32: []int32{0, 1}, quant: quant(0, 1, 0.5)}
	cases := []struct {
		name string
		in   []int8
		want []int8
	}{
		// ch0: 1+3+5 = 9. ch1: 2 - 4 + 12 + 1 = 11, halved: 5.5 rounds to 6.
		{"positive tie", []int8{1, 2, 3, 4, 5, 6}, []int8{9, 6}},
		// ch1: -2 - 4 - 12 + 1 = -17, halved: -8.5 rounds to -8.
		{"negative tie", []int8{1, -2, 3, 4, 5, -6}, []int8{9, -8}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tensors := []tensorSpec{
				{shape: []int{1, 3, 1, 2}, dtype: runtime.Int8, quant: quant(0, 1)},
				filter, bias,
				{shape: []int{1, 1, 1, 2}, dtype: runtime.Int8, quant: quant(0, 1)},
			}
			opts := &runtime.DepthwiseConv2DOptions{Padding: runtime.PaddingValid, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1, DepthMultiplier: 1}
			got := runArith(t, tflite.BuiltinOperatorDEPTHWISE_CONV_2D, opts, tensors, tc.in)
			assertInt8(t, "output", got, tc.want)
		})
	}
}

func TestFullyConnectedOffsetsAndActivation(t *testing.T) {
	// Input zero point 1 (offset -1), real input (0, 1, 2, 3). Output zero point -120.
	// Row 0 (all ones): 6, plus zero point = -114.
	// Row 1 (alternating signs): 0 - 1 + 2 - 3 = -2, bias -20: -22, plus zero point = -142,
	// clamped to -128 without activation and to the zero point -120 with ReLU.
	tensors := []tensorSpec{
		{shape: []int{1, 4}, dtype: runtime.Int8, quant: quant(1, 1)},
		{shape: []int{2, 4}, dtype: runtime.Int8, constInt8: []int8{1, 1, 1, 1, 1, -1, 1, -1}, quant: quant(0, 1)},
		{shape: []int{2}, dtype: runtime.Int32, constInt32: []int32{0, -20}, quant: quant(0, 1)},
		{shape: []int{1, 2}, dtype: runtime.Int8, quant: quant(-120, 1)},
	}
	for _, tc := range []struct {
		act  runtime.Activation
		want []int8
	}{
		{runtime.ActivationNone, []int8{-114, -128}},
		{runtime.ActivationRelu, []int8{-114, -120}},
	} {
		t.Run(tc.act.String(), func(t *testing.T) {
			opts := &runtime.FullyConnectedOptions{Activation: tc.act}
			got := runArith(t, tflite.BuiltinOperatorFULLY_CONNECTED, opts, tensors, []int8{1, 2, 3, 4})
			assertInt8(t, "output", got, tc.want)
		})
	}
}

// TestPrepareRejectsInconsistentModels checks the prepare-time validation: the output shape
// must match the computed convolution geometry, the bias zero point must be zero and the bias
// scale must be within 2 % (of the output scale) of input scale times filter scale.
func TestPrepareRejectsInconsistentModels(t *testing.T) {
	base := func() []tensorSpec {
		return []tensorSpec{
			{shape: []int{1, 5, 1, 1}, dtype: runtime.Int8, quant: quant(0, 1)},
			{shape: []int{1, 3, 1, 1}, dtype: runtime.Int8, constInt8: []int8{1, 0, -1}, quant: perChannel(0, 1)},
			{shape: []int{1}, dtype: runtime.Int32, constInt32: []int32{0}, quant: quant(0, 1)},
			{shape: []int{1, 3, 1, 1}, dtype: runtime.Int8, quant: quant(0, 1)},
		}
	}
	opts := &runtime.Conv2DOptions{Padding: runtime.PaddingValid, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1}
	cases := []struct {
		name   string
		mutate func(ts []tensorSpec)
		want   string
	}{
		{"output shape", func(ts []tensorSpec) { ts[3].shape = []int{1, 5, 1, 1} }, "output shape"},
		{"bias zero point", func(ts []tensorSpec) { ts[2].quant = quant(4, 1) }, "zero point 4"},
		{"bias scale", func(ts []tensorSpec) { ts[2].quant = quant(0, 1.1) }, "beyond 2 %"},
		{"filter zero point", func(ts []tensorSpec) {
			ts[1].quant = &runtime.Quantization{Scales: []float32{1}, ZeroPoints: []int32{5}}
		}, "filter zero point 5"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ts := base()
			tc.mutate(ts)
			err := tryNewSingleOp(t, tflite.BuiltinOperatorCONV_2D, opts, ts, []int{0, 1, 2}, []int{3})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected an error containing %q, got %v", tc.want, err)
			}
		})
	}
}

// tryNewSingleOp is newSingleOp returning the NewInterpreter error instead of failing.
func tryNewSingleOp(t *testing.T, code tflite.BuiltinOperator, opts any, tensors []tensorSpec, inputs, outputs []int) error {
	t.Helper()
	infos := make([]runtime.TensorInfo, len(tensors))
	for i, ts := range tensors {
		info := runtime.TensorInfo{Index: i, Name: "t" + string(rune('0'+i)), Shape: ts.shape, DType: ts.dtype, NumElements: 1, Quant: ts.quant}
		for _, d := range ts.shape {
			info.NumElements *= d
		}
		info.ByteSize = info.NumElements * ts.dtype.Size()
		if ts.constInt8 != nil {
			info.Const = toBytes(ts.constInt8)
		}
		if ts.constInt32 != nil {
			info.Const = make([]byte, 4*len(ts.constInt32))
			for k, v := range ts.constInt32 {
				info.Const[4*k] = byte(v)
				info.Const[4*k+1] = byte(v >> 8)
				info.Const[4*k+2] = byte(v >> 16)
				info.Const[4*k+3] = byte(v >> 24)
			}
		}
		infos[i] = info
	}
	op := runtime.Operator{Code: code, Name: tflite.EnumNamesBuiltinOperator[code], Inputs: inputs, Outputs: outputs, Options: opts, Variable: -1}
	m := &runtime.Model{Version: 3, InitSubgraph: -1, Subgraphs: []runtime.Subgraph{{Name: "main", Tensors: infos, Operators: []runtime.Operator{op}, Inputs: inputs[:1], Outputs: outputs}}}
	_, err := m.NewInterpreter()
	return err
}
