package kernels

import (
	"fmt"
	"math"

	"github.com/tgallice/wakeword-go/runtime"
)

// quantizeParams is what the QUANTIZE prepare step computes once per interpreter
// (tensorflow/lite/kernels/quantize.cc, OpData for the requantize use case).
type quantizeParams struct {
	multiplier                      int32
	shift                           int
	inputZeroPoint, outputZeroPoint int32
}

// prepareQuantize computes the effective scale of an int8 to uint8 requantization
// (quantize.cc, Prepare): input_scale / output_scale in double, then QuantizeMultiplier.
func prepareQuantize(it *runtime.Interpreter, op *runtime.Operator) (any, error) {
	in, out := it.In(op, 0).Info, it.Out(op, 0).Info
	if in.DType != runtime.Int8 || out.DType != runtime.UInt8 {
		return nil, fmt.Errorf("only int8 to uint8 requantization is supported, got %s to %s", in.DType, out.DType)
	}
	if in.Quant == nil || out.Quant == nil || len(in.Quant.Scales) != 1 || len(out.Quant.Scales) != 1 {
		return nil, fmt.Errorf("input and output must have per-tensor quantization")
	}
	p := &quantizeParams{inputZeroPoint: in.Quant.ZeroPoints[0], outputZeroPoint: out.Quant.ZeroPoints[0]}
	effectiveOutputScale := float64(in.Quant.Scales[0]) / float64(out.Quant.Scales[0])
	p.multiplier, p.shift = QuantizeMultiplier(effectiveOutputScale)
	return p, nil
}

// quantize is the int8 to uint8 requantization
// (tensorflow/lite/kernels/internal/reference/requantize.h, Requantize). When the scales are
// equal and the zero points differ by exactly 128, the reference flips the sign bit; the general
// path requantizes with MultiplyByQuantizedMultiplier and clamps.
func quantize(it *runtime.Interpreter, op *runtime.Operator) error {
	p := it.Params(op).(*quantizeParams)
	input, output := it.In(op, 0).Int8(), it.Out(op, 0).Uint8()
	sameScale := p.multiplier == 1<<30 && p.shift == 1
	if sameScale && p.inputZeroPoint-p.outputZeroPoint == -128 {
		for i := range input {
			output[i] = uint8(input[i]) ^ 0x80
		}
		return nil
	}
	for i := range input {
		x := int32(input[i]) - p.inputZeroPoint
		y := MultiplyByQuantizedMultiplier(x, p.multiplier, p.shift) + p.outputZeroPoint
		output[i] = uint8(min(max(y, 0), math.MaxUint8))
	}
	return nil
}
