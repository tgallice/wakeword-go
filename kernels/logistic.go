package kernels

import (
	"fmt"
	"math"

	"github.com/tgallice/wakeword-go/runtime"
)

// logisticParams is what the LOGISTIC prepare step computes once per interpreter
// (tflite-micro kernels/logistic_common.cc, OpDataLogistic).
type logisticParams struct {
	inputZeroPoint   int32
	inputRangeRadius int32
	inputMultiplier  int32
	inputLeftShift   int
}

const (
	// logisticInputIntegerBits is the fixed-point format of the sigmoid input, shared by the
	// prepare step and the kernel (kInputIntegerBits in both files).
	logisticInputIntegerBits = 4
	// logisticOutputIntegerBits is the number of output bits (kOutputIntegerBits in logistic.h).
	logisticOutputIntegerBits = 8
	// logisticOutputZeroPoint and logisticOutputScale are the only int8 output quantization the
	// reference accepts: the sigmoid range [0, 1] mapped onto [-128, 127] in 1/256 steps.
	logisticOutputZeroPoint int32   = -128
	logisticOutputScale     float32 = 1.0 / 256
)

// calculateInputRadius mirrors quantization_util.cc CalculateInputRadius: the largest input
// (in the requantized domain) that the fixed-point sigmoid can take before saturating.
func calculateInputRadius(inputIntegerBits, inputLeftShift, totalSignedBits int) int32 {
	maxInputRescaled := 1.0 * float64((int64(1)<<inputIntegerBits)-1) *
		float64(int64(1)<<(totalSignedBits-inputIntegerBits)) /
		float64(int64(1)<<inputLeftShift)
	return int32(math.Floor(maxInputRescaled))
}

// prepareLogistic computes the int8 sigmoid parameters (logistic_common.cc,
// CalculateArithmeticOpDataLogistic). The output quantization must be zero point -128 and
// scale 1/256, as both TFLite (activations.cc, SigmoidPrepare) and tflite-micro require.
func prepareLogistic(it *runtime.Interpreter, op *runtime.Operator) (any, error) {
	in, out := it.In(op, 0).Info, it.Out(op, 0).Info
	if err := checkLogisticOutputQuantization(out); err != nil {
		return nil, err
	}
	p := &logisticParams{inputZeroPoint: in.Quant.ZeroPoints[0]}
	inputRealMultiplier := float64(in.Quant.Scales[0]) * float64(int32(1)<<(31-logisticInputIntegerBits))
	q, exp := math.Frexp(inputRealMultiplier)
	p.inputLeftShift = exp
	p.inputMultiplier = int32(math.Round(q * (1 << 31)))
	p.inputRangeRadius = calculateInputRadius(logisticInputIntegerBits, p.inputLeftShift, 31)
	return p, nil
}

// checkLogisticOutputQuantization rejects any int8 LOGISTIC output quantization other than
// the one the reference kernels accept.
func checkLogisticOutputQuantization(out *runtime.TensorInfo) error {
	if out.Quant == nil || len(out.Quant.Scales) != 1 {
		return fmt.Errorf("output tensor %s must have per-tensor quantization", out.Name)
	}
	if out.Quant.ZeroPoints[0] != logisticOutputZeroPoint {
		return fmt.Errorf("output zero point %d, the int8 sigmoid requires %d", out.Quant.ZeroPoints[0], logisticOutputZeroPoint)
	}
	if out.Quant.Scales[0] != logisticOutputScale {
		return fmt.Errorf("output scale %v, the int8 sigmoid requires 1/256", out.Quant.Scales[0])
	}
	return nil
}

// logistic is the int8 sigmoid (tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h,
// Logistic for int8). The input is requantized to 4 integer bits, passed through the gemmlowp
// fixed-point sigmoid, then rounded to the 8-bit output. Inputs beyond the range radius
// saturate to the output bounds.
//
// TFLite's reference resolver evaluates the int8 LOGISTIC through a 256-entry table built from
// the float sigmoid (activations.cc, LUTPopulate), while tflite-micro evaluates this fixed-point
// path. On the five microWakeWord v2 models both agree on all 256 inputs, and the parity traces
// match this implementation; the fixed-point path is kept because tflite-micro is what the
// models run on in production.
func logistic(it *runtime.Interpreter, op *runtime.Operator) error {
	p := it.Params(op).(*logisticParams)
	input, output := it.In(op, 0).Int8(), it.Out(op, 0).Int8()
	for i := range input {
		x := int32(input[i]) - p.inputZeroPoint
		switch {
		case x <= -p.inputRangeRadius:
			output[i] = math.MinInt8
		case x >= p.inputRangeRadius:
			output[i] = math.MaxInt8
		default:
			inputInQ4 := MultiplyByQuantizedMultiplier(x, p.inputMultiplier, p.inputLeftShift)
			outputInQ0 := logisticFixedPoint(inputInQ4, logisticInputIntegerBits)
			outputInQ23 := RoundingDivideByPOT(outputInQ0, 31-logisticOutputIntegerBits)
			outputInQ23 = min(max(outputInQ23+logisticOutputZeroPoint, math.MinInt8), math.MaxInt8)
			output[i] = int8(outputInQ23)
		}
	}
	return nil
}
