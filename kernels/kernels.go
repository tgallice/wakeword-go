// Package kernels implements the operator kernels of the runtime: the copy operators of
// Phase 2 and, in later phases, the int8 arithmetic operators. Importing the package registers
// every kernel; importers that only need the loader can skip it.
//
// Each kernel is a line-by-line port of the TensorFlow Lite reference implementation named in
// its comment. Kernels never allocate: every scratch value lives on the stack in fixed-size
// arrays bounded by the loader's maximum dimension count.
package kernels

import (
	"github.com/tgallice/wakeword-go/runtime"
	"github.com/tgallice/wakeword-go/tflite"
)

func init() {
	runtime.RegisterKernel(tflite.BuiltinOperatorRESHAPE, reshape)
	runtime.RegisterKernel(tflite.BuiltinOperatorCONCATENATION, concatenation)
	runtime.RegisterKernel(tflite.BuiltinOperatorSTRIDED_SLICE, stridedSlice)
	runtime.RegisterKernel(tflite.BuiltinOperatorSPLIT_V, splitV)
}

// maxDims bounds the rank of tensors handled by the kernels; the loader enforces it.
const maxDims = 8

// dims is a fixed-size shape holder used to keep kernels allocation-free.
type dims [maxDims]int

// elementSize returns the byte size of one element of a tensor.
func elementSize(t *runtime.Tensor) int { return t.Info.DType.Size() }

// product multiplies shape[from:to].
func product(shape []int, from, to int) int {
	n := 1
	for _, d := range shape[from:to] {
		n *= d
	}
	return n
}

// resolveAxis maps a possibly negative axis onto [0, rank).
func resolveAxis(axis, rank int) (int, bool) {
	if axis < 0 {
		axis += rank
	}
	return axis, axis >= 0 && axis < rank
}
