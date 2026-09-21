// Package runtime loads microWakeWord TFLite models and prepares them for execution.
//
// Load parses and validates a .tflite file against the closed operator set described in
// docs/spec.md section 5. Everything the interpreter needs (shapes, quantization parameters,
// constant buffers, operator options, state variables) is read from the file; nothing is
// hardcoded. Any operator, version, dtype or option outside the supported set makes Load fail
// with an error naming the offending operator and its index.
package runtime

import (
	"fmt"

	"github.com/tgallice/wakeword-go/tflite"
)

// DType is the element type of a tensor. Only the types used by microWakeWord v2 models are
// accepted; anything else is rejected at load time.
type DType uint8

// Supported tensor element types.
const (
	Int8 DType = iota + 1
	UInt8
	Int32
	// Resource tensors carry no data: they reference a state variable (see Variable).
	Resource
)

// String returns the TFLite name of the dtype.
func (d DType) String() string {
	switch d {
	case Int8:
		return "INT8"
	case UInt8:
		return "UINT8"
	case Int32:
		return "INT32"
	case Resource:
		return "RESOURCE"
	default:
		return fmt.Sprintf("DType(%d)", uint8(d))
	}
}

// Size returns the size in bytes of one element, or 0 for Resource.
func (d DType) Size() int {
	switch d {
	case Int8, UInt8:
		return 1
	case Int32:
		return 4
	default:
		return 0
	}
}

func dtypeFromTFLite(t tflite.TensorType) (DType, bool) {
	switch t {
	case tflite.TensorTypeINT8:
		return Int8, true
	case tflite.TensorTypeUINT8:
		return UInt8, true
	case tflite.TensorTypeINT32:
		return Int32, true
	case tflite.TensorTypeRESOURCE:
		return Resource, true
	default:
		return 0, false
	}
}

// Quantization holds the affine quantization parameters of a tensor.
//
// Per-tensor quantization has exactly one scale and one zero point. Per-channel quantization
// (convolution weights) has one scale and one zero point per index along QuantizedDimension.
type Quantization struct {
	Scales             []float32
	ZeroPoints         []int32
	QuantizedDimension int
}

// PerChannel reports whether the tensor uses per-channel quantization.
func (q *Quantization) PerChannel() bool {
	return q != nil && len(q.Scales) > 1
}

// Equal reports whether two quantizations are exactly identical.
func (q *Quantization) Equal(o *Quantization) bool {
	if q == nil || o == nil {
		return q == o
	}
	if len(q.Scales) != len(o.Scales) || len(q.ZeroPoints) != len(o.ZeroPoints) ||
		q.QuantizedDimension != o.QuantizedDimension {
		return false
	}
	for i := range q.Scales {
		if q.Scales[i] != o.Scales[i] {
			return false
		}
	}
	for i := range q.ZeroPoints {
		if q.ZeroPoints[i] != o.ZeroPoints[i] {
			return false
		}
	}
	return true
}
