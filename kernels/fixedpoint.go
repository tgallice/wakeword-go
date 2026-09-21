package kernels

import "math"

// Fixed-point transcendental functions ported from gemmlowp fixedpoint/fixedpoint.h, the
// arithmetic behind the int8 LOGISTIC reference kernel. A fixed-point number with n integer
// bits is an int32 raw value scaled by 2^(31-n). Every operation below keeps the exact integer
// types, wrapping and saturation of the C++ templates, instantiated for int32 raw values.

// int32Max is gemmlowp's FixedPoint<int32, 0>::One(): the largest representable value.
const int32Max = math.MaxInt32

// roundingHalfSum returns (a+b)/2 rounded to the nearest integer, ties away from zero
// (fixedpoint.h, RoundingHalfSum for int32).
func roundingHalfSum(a, b int32) int32 {
	sum := int64(a) + int64(b)
	sign := int64(1)
	if sum < 0 {
		sign = -1
	}
	return int32((sum + sign) / 2)
}

// saturatingRoundingMultiplyByPOT multiplies x by 2^exponent: a saturating left shift for a
// positive exponent, a rounding right shift for a negative one (fixedpoint.h,
// SaturatingRoundingMultiplyByPOT).
func saturatingRoundingMultiplyByPOT(x int32, exponent int) int32 {
	switch {
	case exponent == 0:
		return x
	case exponent < 0:
		return RoundingDivideByPOT(x, -exponent)
	}
	threshold := int32((1 << (31 - exponent)) - 1)
	if x > threshold {
		return math.MaxInt32
	}
	if x < -threshold {
		return math.MinInt32
	}
	return x << exponent
}

// expOnIntervalBetweenNegativeOneQuarterAnd0Excl returns exp(a) for a in [-1/4, 0), both in
// fixed point with 0 integer bits (fixedpoint.h,
// exp_on_interval_between_negative_one_quarter_and_0_excl). The Taylor expansion is taken
// around -1/8 with the change of variable x = a + 1/8.
func expOnIntervalBetweenNegativeOneQuarterAnd0Excl(a int32) int32 {
	const (
		constantTerm   int32 = 1895147668 // exp(-1/8)
		constant1Over3 int32 = 715827883  // 1/3
	)
	x := a + 1<<28
	x2 := SaturatingRoundingDoublingHighMul(x, x)
	x3 := SaturatingRoundingDoublingHighMul(x2, x)
	x4 := SaturatingRoundingDoublingHighMul(x2, x2)
	x4Over4 := saturatingRoundingMultiplyByPOT(x4, -2)
	x4Over24PlusX3Over6PlusX2Over2 := saturatingRoundingMultiplyByPOT(
		SaturatingRoundingDoublingHighMul(x4Over4+x3, constant1Over3)+x2, -1)
	return constantTerm + SaturatingRoundingDoublingHighMul(constantTerm, x+x4Over24PlusX3Over6PlusX2Over2)
}

// expBarrelShifterMultipliers are the 0-integer-bit representations of exp(-2^k) for
// k = -2 .. 4, applied by expOnNegativeValues for each set bit of the integer part of -a.
var expBarrelShifterMultipliers = [...]struct {
	exponent   int
	multiplier int32
}{
	{-2, 1672461947}, // exp(-1/4)
	{-1, 1302514674}, // exp(-1/2)
	{0, 790015084},   // exp(-1)
	{1, 290630308},   // exp(-2)
	{2, 39332535},    // exp(-4)
	{3, 720401},      // exp(-8)
	{4, 242},         // exp(-16)
}

// expOnNegativeValues returns exp(a) with 0 integer bits for a <= 0 given with integerBits
// integer bits (fixedpoint.h, exp_on_negative_values).
func expOnNegativeValues(a int32, integerBits int) int32 {
	fractionalBits := 31 - integerBits
	oneQuarter := int32(1) << (fractionalBits - 2)
	mask := oneQuarter - 1
	aModQuarterMinusOneQuarter := (a & mask) - oneQuarter
	result := expOnIntervalBetweenNegativeOneQuarterAnd0Excl(
		saturatingRoundingMultiplyByPOT(aModQuarterMinusOneQuarter, integerBits))
	remainder := aModQuarterMinusOneQuarter - a
	for _, m := range expBarrelShifterMultipliers {
		if integerBits > m.exponent {
			shift := fractionalBits + m.exponent
			if remainder&(1<<shift) != 0 {
				result = SaturatingRoundingDoublingHighMul(result, m.multiplier)
			}
		}
	}
	if integerBits > 5 {
		clamp := -(int32(1) << (36 - integerBits))
		if a < clamp {
			result = 0
		}
	}
	if a == 0 {
		result = int32Max
	}
	return result
}

// oneOverOnePlusXForXIn01 returns 1/(1+a) for a in (0, 1), both with 0 integer bits
// (fixedpoint.h, one_over_one_plus_x_for_x_in_0_1), by three Newton-Raphson steps carried
// with 2 integer bits.
func oneOverOnePlusXForXIn01(a int32) int32 {
	const (
		constant48Over17    int32 = 1515870810  // 48/17 with 2 integer bits
		constantNeg32Over17 int32 = -1010580540 // -32/17 with 2 integer bits
		oneWith2IntegerBits int32 = 1 << 29
	)
	halfDenominator := roundingHalfSum(a, int32Max)
	x := constant48Over17 + SaturatingRoundingDoublingHighMul(halfDenominator, constantNeg32Over17)
	for range 3 {
		halfDenominatorTimesX := SaturatingRoundingDoublingHighMul(halfDenominator, x)
		oneMinusHalfDenominatorTimesX := oneWith2IntegerBits - halfDenominatorTimesX
		x += saturatingRoundingMultiplyByPOT(SaturatingRoundingDoublingHighMul(x, oneMinusHalfDenominatorTimesX), 2)
	}
	// Rescale<0>(ExactMulByPot<-1>(x)): x/2 read with 1 integer bit, then brought to 0.
	return saturatingRoundingMultiplyByPOT(x, 1)
}

// logisticFixedPoint returns 1/(1+exp(-a)) with 0 integer bits for a given with integerBits
// integer bits (fixedpoint.h, logistic and logistic_on_positive_values).
func logisticFixedPoint(a int32, integerBits int) int32 {
	const oneHalf int32 = 1 << 30
	if a == 0 {
		return oneHalf
	}
	absInput := a
	if a < 0 {
		absInput = -a
	}
	resultIfPositive := oneOverOnePlusXForXIn01(expOnNegativeValues(-absInput, integerBits))
	if a > 0 {
		return resultIfPositive
	}
	return int32Max - resultIfPositive
}
