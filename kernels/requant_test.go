package kernels

import (
	"math"
	"testing"
)

// Expected values were computed with a Python transcription of the C reference (gemmlowp
// SaturatingRoundingDoublingHighMul and RoundingDivideByPOT, quantization_util.cc
// QuantizeMultiplier) using Python's unbounded integers and truncating division.

func TestSaturatingRoundingDoublingHighMul(t *testing.T) {
	cases := []struct {
		name string
		a, b int32
		want int32
	}{
		{"int32min squared saturates", math.MinInt32, math.MinInt32, math.MaxInt32},
		{"2^30 squared", 1 << 30, 1 << 30, 1 << 29},
		{"5 * 0.5 rounds half up", 5, 1 << 30, 3},
		{"-5 * 0.5 rounds half toward positive", -5, 1 << 30, -2},
		{"7 * 0.5", 7, 1 << 30, 4},
		{"-7 * 0.5", -7, 1 << 30, -3},
		{"int32max squared", math.MaxInt32, math.MaxInt32, math.MaxInt32 - 1},
		{"int32min * int32max does not saturate", math.MinInt32, math.MaxInt32, -math.MaxInt32},
		{"negative multiplier", 3, -(1 << 30), -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := SaturatingRoundingDoublingHighMul(tc.a, tc.b); got != tc.want {
				t.Errorf("SaturatingRoundingDoublingHighMul(%d, %d) = %d, want %d", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

func TestRoundingDivideByPOT(t *testing.T) {
	cases := []struct {
		name     string
		x        int32
		exponent int
		want     int32
	}{
		{"2.5 rounds away from zero", 5, 1, 3},
		{"-2.5 rounds away from zero", -5, 1, -3},
		{"exact 2", 4, 1, 2},
		{"-1.5 rounds away from zero", -3, 1, -2},
		{"1.75 rounds up", 7, 2, 2},
		{"int32max by 2^31", math.MaxInt32, 31, 1},
		{"int32min by 2^31", math.MinInt32, 31, -1},
		{"-1 by 2^31", -1, 31, 0},
		{"exponent 0 is identity", 6, 0, 6},
		{"-1.5 by 4", -6, 2, -2},
		{"-1.75 by 4", -7, 2, -2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := RoundingDivideByPOT(tc.x, tc.exponent); got != tc.want {
				t.Errorf("RoundingDivideByPOT(%d, %d) = %d, want %d", tc.x, tc.exponent, got, tc.want)
			}
		})
	}
}

func TestQuantizeMultiplier(t *testing.T) {
	cases := []struct {
		name        string
		multiplier  float64
		significand int32
		shift       int
	}{
		{"zero", 0, 0, 0},
		{"one", 1.0, 1 << 30, 1},
		{"half", 0.5, 1 << 30, 0},
		{"three quarters", 0.75, 1610612736, 0},
		{"three", 3.0, 1610612736, 2},
		{"2^-40 flushes to zero", math.Ldexp(1, -40), 0, 0},
		{"2^-31 is the smallest kept", math.Ldexp(1, -31), 1 << 30, -30},
		{"2^-32 is still kept", math.Ldexp(1, -32), 1 << 30, -31},
		{"just below one rounds to 2^31 and renormalizes", 0.9999999999, 1 << 30, 1},
		{"model-like scale", 0.10196078568696976 * 0.0123 / 0.05, 1723646193, -5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, sh := QuantizeMultiplier(tc.multiplier)
			if s != tc.significand || sh != tc.shift {
				t.Errorf("QuantizeMultiplier(%v) = (%d, %d), want (%d, %d)", tc.multiplier, s, sh, tc.significand, tc.shift)
			}
		})
	}
}

func TestMultiplyByQuantizedMultiplier(t *testing.T) {
	cases := []struct {
		name        string
		x           int32
		significand int32
		shift       int
		want        int32
	}{
		{"multiplier one is identity", 1000, 1 << 30, 1, 1000},
		{"half of even", 1000, 1 << 30, 0, 500},
		{"half of odd rounds up", 1001, 1 << 30, 0, 501},
		{"half of negative odd rounds toward positive", -1001, 1 << 30, 0, -500},
		{"0.75 * 2^-3", 1000, 1610612736, -3, 94},
		{"0.75 * 2^-3 negative", -1000, 1610612736, -3, -94},
		{"half times 2^2", 123456789, 1 << 30, 2, 246913578},
		{"left shift wraps like int32", 1 << 30, math.MaxInt32, 1, -math.MaxInt32},
		{"int32min times almost one", math.MinInt32, math.MaxInt32, 0, -math.MaxInt32},
		{"right shift 31 of a small value", 1, 1 << 30, -31, 0},
		{"zero stays zero", 0, 12345, -5, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := MultiplyByQuantizedMultiplier(tc.x, tc.significand, tc.shift); got != tc.want {
				t.Errorf("MultiplyByQuantizedMultiplier(%d, %d, %d) = %d, want %d", tc.x, tc.significand, tc.shift, got, tc.want)
			}
		})
	}
}
