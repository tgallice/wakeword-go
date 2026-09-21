package kernels

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/tgallice/wakeword-go/runtime"
	"github.com/tgallice/wakeword-go/tflite"
)

// tensorSpec describes one tensor of a synthetic single-operator model.
type tensorSpec struct {
	shape []int
	dtype runtime.DType
	// constant data, nil for a dynamic tensor
	constInt8  []int8
	constInt32 []int32
	// quant overrides the default quantization (int8Quant for int8 tensors, none otherwise)
	quant *runtime.Quantization
}

// int8Quant is the shared quantization of every int8 tensor of the synthetic models; the copy
// kernels require identical parameters on inputs and outputs.
var int8Quant = &runtime.Quantization{Scales: []float32{0.5}, ZeroPoints: []int32{-128}}

// newSingleOp builds a one-operator model whose tensors are listed in order, returns an
// interpreter for it. Inputs and outputs are tensor indices.
func newSingleOp(t *testing.T, code tflite.BuiltinOperator, name string, opts any, tensors []tensorSpec, inputs, outputs []int) *runtime.Interpreter {
	t.Helper()
	infos := make([]runtime.TensorInfo, len(tensors))
	for i, ts := range tensors {
		info := runtime.TensorInfo{Index: i, Name: name + "_t" + string(rune('0'+i)), Shape: ts.shape, DType: ts.dtype, NumElements: 1}
		for _, d := range ts.shape {
			info.NumElements *= d
		}
		info.ByteSize = info.NumElements * ts.dtype.Size()
		switch {
		case ts.quant != nil:
			info.Quant = ts.quant
		case ts.dtype == runtime.Int8:
			info.Quant = int8Quant
		}
		switch {
		case ts.constInt8 != nil:
			info.Const = make([]byte, len(ts.constInt8))
			for k, v := range ts.constInt8 {
				info.Const[k] = byte(v)
			}
		case ts.constInt32 != nil:
			info.Const = make([]byte, 4*len(ts.constInt32))
			for k, v := range ts.constInt32 {
				binary.LittleEndian.PutUint32(info.Const[4*k:], uint32(v))
			}
		}
		if info.Const != nil && len(info.Const) != info.ByteSize {
			t.Fatalf("tensor %d: %d constant bytes for a %d-byte tensor", i, len(info.Const), info.ByteSize)
		}
		infos[i] = info
	}
	op := runtime.Operator{Subgraph: 0, Index: 0, Code: code, Name: name, Inputs: inputs, Outputs: outputs, Options: opts, Variable: -1}
	m := &runtime.Model{Version: 3, InitSubgraph: -1, Subgraphs: []runtime.Subgraph{{
		Index: 0, Name: "main", Tensors: infos, Operators: []runtime.Operator{op}, Inputs: inputs[:1], Outputs: outputs,
	}}}
	it, err := m.NewInterpreter()
	if err != nil {
		t.Fatalf("NewInterpreter: %v", err)
	}
	return it
}

func int8Data(n int, f func(i int) int8) []int8 {
	out := make([]int8, n)
	for i := range out {
		out[i] = f(i)
	}
	return out
}

func toBytes(v []int8) []byte {
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out
}

func assertBytes(t *testing.T, what string, got, want []byte) {
	t.Helper()
	if !bytes.Equal(got, want) {
		t.Errorf("%s:\n got  %v\n want %v", what, toInt8(got), toInt8(want))
	}
}

func toInt8(b []byte) []int8 {
	out := make([]int8, len(b))
	for i, x := range b {
		out[i] = int8(x)
	}
	return out
}

func TestReshapeCopies(t *testing.T) {
	in := int8Data(24, func(i int) int8 { return int8(i - 12) })
	it := newSingleOp(t, tflite.BuiltinOperatorRESHAPE, "RESHAPE", nil, []tensorSpec{
		{shape: []int{1, 3, 8}, dtype: runtime.Int8},
		{shape: []int{4}, dtype: runtime.Int32, constInt32: []int32{1, 3, 1, 8}},
		{shape: []int{1, 3, 1, 8}, dtype: runtime.Int8},
	}, []int{0, 1}, []int{2})
	if err := it.Tensor(0, 0).SetInt8(in); err != nil {
		t.Fatal(err)
	}
	if err := it.RunOperator(0, 0); err != nil {
		t.Fatal(err)
	}
	assertBytes(t, "reshape", it.Tensor(0, 2).Data, toBytes(in))
}

func TestConcatenation(t *testing.T) {
	// a: [1,2,1,3] with values 1..6, b: [1,1,1,3] with values 10..12.
	a := int8Data(6, func(i int) int8 { return int8(i + 1) })
	b := int8Data(3, func(i int) int8 { return int8(i + 10) })
	cases := []struct {
		name     string
		axis     int
		aShape   []int
		bShape   []int
		outShape []int
		want     []int8
	}{
		{"axis1_time", 1, []int{1, 2, 1, 3}, []int{1, 1, 1, 3}, []int{1, 3, 1, 3}, []int8{1, 2, 3, 4, 5, 6, 10, 11, 12}},
		{"axis-3_time", -3, []int{1, 2, 1, 3}, []int{1, 1, 1, 3}, []int{1, 3, 1, 3}, []int8{1, 2, 3, 4, 5, 6, 10, 11, 12}},
		{"axis-1_channels", -1, []int{1, 1, 1, 6}, []int{1, 1, 1, 3}, []int{1, 1, 1, 9}, []int8{1, 2, 3, 4, 5, 6, 10, 11, 12}},
		{"axis3_interleaved", 3, []int{1, 3, 1, 2}, []int{1, 3, 1, 1}, []int{1, 3, 1, 3}, []int8{1, 2, 10, 3, 4, 11, 5, 6, 12}},
		{"axis0_batch", 0, []int{2, 3}, []int{1, 3}, []int{3, 3}, []int8{1, 2, 3, 4, 5, 6, 10, 11, 12}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			it := newSingleOp(t, tflite.BuiltinOperatorCONCATENATION, "CONCATENATION",
				&runtime.ConcatenationOptions{Axis: tc.axis}, []tensorSpec{
					{shape: tc.aShape, dtype: runtime.Int8},
					{shape: tc.bShape, dtype: runtime.Int8},
					{shape: tc.outShape, dtype: runtime.Int8},
				}, []int{0, 1}, []int{2})
			if err := it.Tensor(0, 0).SetInt8(a); err != nil {
				t.Fatal(err)
			}
			if err := it.Tensor(0, 1).SetInt8(b); err != nil {
				t.Fatal(err)
			}
			if err := it.RunOperator(0, 0); err != nil {
				t.Fatal(err)
			}
			assertBytes(t, "concat", it.Tensor(0, 2).Data, toBytes(tc.want))
		})
	}
}

func TestStridedSlice(t *testing.T) {
	// Input [1,5,1,4]: value = 10*t + c for frame t (0..4) and channel c (0..3).
	in := int8Data(20, func(i int) int8 { return int8(10*(i/4) + i%4) })
	cases := []struct {
		name          string
		begin, end    []int32
		strides       []int32
		beginMask     int32
		endMask       int32
		outShape      []int
		want          []int8
		wantLoadError bool
	}{
		{
			// The microWakeWord pattern: keep the last 2 frames, masks 13 and 15.
			name: "keep_last_2_masks_13_15", begin: []int32{0, -2, 0, 0}, end: []int32{0, 0, 0, 4}, strides: []int32{1, 1, 1, 1},
			beginMask: 13, endMask: 15, outShape: []int{1, 2, 1, 4},
			want: []int8{30, 31, 32, 33, 40, 41, 42, 43},
		},
		{
			name: "keep_last_4_end_zero_masked", begin: []int32{0, -4, 0, 0}, end: []int32{0, 0, 0, 0}, strides: []int32{1, 1, 1, 1},
			beginMask: 13, endMask: 15, outShape: []int{1, 4, 1, 4},
			want: []int8{10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43},
		},
		{
			name: "negative_begin_beyond_size_clamps", begin: []int32{0, -9, 0, 0}, end: []int32{0, 0, 0, 0}, strides: []int32{1, 1, 1, 1},
			beginMask: 13, endMask: 15, outShape: []int{1, 5, 1, 4}, want: in,
		},
		{
			name: "explicit_begin_end_no_masks", begin: []int32{0, 1, 0, 1}, end: []int32{1, 3, 1, 3}, strides: []int32{1, 1, 1, 1},
			outShape: []int{1, 2, 1, 2}, want: []int8{11, 12, 21, 22},
		},
		{
			name: "negative_end", begin: []int32{0, 0, 0, 0}, end: []int32{1, -1, 1, -1}, strides: []int32{1, 1, 1, 1},
			outShape: []int{1, 4, 1, 3}, want: []int8{0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32},
		},
		{
			name: "stride_2_on_time", begin: []int32{0, 0, 0, 0}, end: []int32{1, 5, 1, 4}, strides: []int32{1, 2, 1, 1},
			outShape: []int{1, 3, 1, 4}, want: []int8{0, 1, 2, 3, 20, 21, 22, 23, 40, 41, 42, 43},
		},
		{
			name: "negative_stride_reverses", begin: []int32{0, 4, 0, 0}, end: []int32{1, 1, 1, 4}, strides: []int32{1, -1, 1, 1},
			outShape: []int{1, 3, 1, 4}, want: []int8{40, 41, 42, 43, 30, 31, 32, 33, 20, 21, 22, 23},
		},
		{
			name: "negative_stride_full_masks", begin: []int32{0, 0, 0, 0}, end: []int32{0, 0, 0, 0}, strides: []int32{1, -2, 1, 1},
			beginMask: 15, endMask: 15, outShape: []int{1, 3, 1, 4},
			want: []int8{40, 41, 42, 43, 20, 21, 22, 23, 0, 1, 2, 3},
		},
		{
			name: "empty_slice", begin: []int32{0, 3, 0, 0}, end: []int32{1, 3, 1, 4}, strides: []int32{1, 1, 1, 1},
			outShape: []int{1, 0, 1, 4}, want: []int8{},
		},
		{
			name: "output_shape_mismatch_is_an_error", begin: []int32{0, -2, 0, 0}, end: []int32{0, 0, 0, 0}, strides: []int32{1, 1, 1, 1},
			beginMask: 13, endMask: 15, outShape: []int{1, 3, 1, 4}, wantLoadError: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			it := newSingleOp(t, tflite.BuiltinOperatorSTRIDED_SLICE, "STRIDED_SLICE",
				&runtime.StridedSliceOptions{BeginMask: tc.beginMask, EndMask: tc.endMask}, []tensorSpec{
					{shape: []int{1, 5, 1, 4}, dtype: runtime.Int8},
					{shape: []int{4}, dtype: runtime.Int32, constInt32: tc.begin},
					{shape: []int{4}, dtype: runtime.Int32, constInt32: tc.end},
					{shape: []int{4}, dtype: runtime.Int32, constInt32: tc.strides},
					{shape: tc.outShape, dtype: runtime.Int8},
				}, []int{0, 1, 2, 3}, []int{4})
			if err := it.Tensor(0, 0).SetInt8(in); err != nil {
				t.Fatal(err)
			}
			err := it.RunOperator(0, 0)
			if tc.wantLoadError {
				if err == nil {
					t.Fatal("expected an error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			assertBytes(t, "slice", it.Tensor(0, 4).Data, toBytes(tc.want))
		})
	}
}

func TestSplitV(t *testing.T) {
	// Input [1,2,1,6]: value = 10*t + c.
	in := int8Data(12, func(i int) int8 { return int8(10*(i/6) + i%6) })
	cases := []struct {
		name   string
		sizes  []int32
		axis   int32
		shapes [][]int
		want   [][]int8
	}{
		{
			"channels_3_3_axis-1",
			[]int32{3, 3},
			-1,
			[][]int{{1, 2, 1, 3}, {1, 2, 1, 3}},
			[][]int8{{0, 1, 2, 10, 11, 12}, {3, 4, 5, 13, 14, 15}},
		},
		{
			"channels_2_4_axis3",
			[]int32{2, 4},
			3,
			[][]int{{1, 2, 1, 2}, {1, 2, 1, 4}},
			[][]int8{{0, 1, 10, 11}, {2, 3, 4, 5, 12, 13, 14, 15}},
		},
		{
			"remainder_wildcard",
			[]int32{4, -1},
			3,
			[][]int{{1, 2, 1, 4}, {1, 2, 1, 2}},
			[][]int8{{0, 1, 2, 3, 10, 11, 12, 13}, {4, 5, 14, 15}},
		},
		{
			"time_axis1",
			[]int32{1, 1},
			1,
			[][]int{{1, 1, 1, 6}, {1, 1, 1, 6}},
			[][]int8{{0, 1, 2, 3, 4, 5}, {10, 11, 12, 13, 14, 15}},
		},
		{
			"three_way",
			[]int32{1, 2, 3},
			-1,
			[][]int{{1, 2, 1, 1}, {1, 2, 1, 2}, {1, 2, 1, 3}},
			[][]int8{{0, 10}, {1, 2, 11, 12}, {3, 4, 5, 13, 14, 15}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			specs := []tensorSpec{
				{shape: []int{1, 2, 1, 6}, dtype: runtime.Int8},
				{shape: []int{len(tc.sizes)}, dtype: runtime.Int32, constInt32: tc.sizes},
				{shape: []int{}, dtype: runtime.Int32, constInt32: []int32{tc.axis}},
			}
			outputs := make([]int, len(tc.shapes))
			for i, s := range tc.shapes {
				outputs[i] = len(specs)
				specs = append(specs, tensorSpec{shape: s, dtype: runtime.Int8})
			}
			it := newSingleOp(t, tflite.BuiltinOperatorSPLIT_V, "SPLIT_V", &runtime.SplitVOptions{NumSplits: len(tc.sizes)},
				specs, []int{0, 1, 2}, outputs)
			if err := it.Tensor(0, 0).SetInt8(in); err != nil {
				t.Fatal(err)
			}
			if err := it.RunOperator(0, 0); err != nil {
				t.Fatal(err)
			}
			for i, idx := range outputs {
				assertBytes(t, "split output "+string(rune('0'+i)), it.Tensor(0, idx).Data, toBytes(tc.want[i]))
			}
		})
	}
}
