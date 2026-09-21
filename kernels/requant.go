package kernels

import "math"

// Fixed-point requantization helpers, ported from tensorflow/lite/kernels/internal/common.h,
// tensorflow/lite/kernels/internal/quantization_util.cc and gemmlowp fixedpoint/fixedpoint.h.
//
// The reference integer kernels compute an int32 accumulator and scale it to the output
// domain with MultiplyByQuantizedMultiplier. The multiplier is a Q0.31 significand with a
// power-of-two exponent produced by QuantizeMultiplier. TFLITE_SINGLE_ROUNDING is off in the
// default TensorFlow Lite and tflite-micro builds, so the double-rounding variant is the one
// reproduced here: a saturating rounding doubling high multiply followed by a rounding
// division by a power of two.

// QuantizeMultiplier splits a real multiplier into a Q0.31 significand and a base-two exponent
// such that multiplier is approximately significand * 2^(shift-31). A multiplier of zero yields
// (0, 0), and a multiplier below 2^-31 is flushed to (0, 0) as the reference does
// (quantization_util.cc, QuantizeMultiplier).
func QuantizeMultiplier(multiplier float64) (significand int32, shift int) {
	if multiplier == 0 {
		return 0, 0
	}
	q, exp := math.Frexp(multiplier)
	qFixed := int64(math.Round(q * (1 << 31)))
	if qFixed == 1<<31 {
		qFixed /= 2
		exp++
	}
	if exp < -31 {
		return 0, 0
	}
	return int32(qFixed), exp
}

// SaturatingRoundingDoublingHighMul returns the high 32 bits of 2*a*b, rounded to nearest,
// saturating the single overflowing case INT32_MIN * INT32_MIN to INT32_MAX
// (gemmlowp fixedpoint.h).
func SaturatingRoundingDoublingHighMul(a, b int32) int32 {
	overflow := a == math.MinInt32 && b == math.MinInt32
	ab := int64(a) * int64(b)
	var nudge int64
	if ab >= 0 {
		nudge = 1 << 30
	} else {
		nudge = 1 - (1 << 30)
	}
	// Truncating division, as in C.
	abX2High32 := int32((ab + nudge) / (1 << 31))
	if overflow {
		return math.MaxInt32
	}
	return abX2High32
}

// RoundingDivideByPOT divides x by 2^exponent, rounding to nearest with ties away from zero
// (gemmlowp fixedpoint.h). exponent must be in [0, 31].
func RoundingDivideByPOT(x int32, exponent int) int32 {
	mask := int32((int64(1) << exponent) - 1)
	remainder := x & mask
	threshold := mask >> 1
	if x < 0 {
		threshold++
	}
	result := x >> exponent
	if remainder > threshold {
		result++
	}
	return result
}

// MultiplyByQuantizedMultiplier scales x by significand * 2^(shift-31) with the double-rounding
// arithmetic of the reference kernels (common.h, MultiplyByQuantizedMultiplier with
// TFLITE_SINGLE_ROUNDING off). A positive shift is applied to x before the multiplication and
// wraps like the int32 multiplication of the C code; a negative shift is a rounding right
// shift of the product.
func MultiplyByQuantizedMultiplier(x, significand int32, shift int) int32 {
	leftShift, rightShift := 0, 0
	if shift > 0 {
		leftShift = shift
	} else {
		rightShift = -shift
	}
	return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x*(1<<leftShift), significand), rightShift)
}
