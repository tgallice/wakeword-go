package kernels

import (
	"math"
	"strings"
	"testing"

	"github.com/tgallice/wakeword-go/runtime"
	"github.com/tgallice/wakeword-go/tflite"
)

// runUnary builds a one-operator model with a single dynamic input and output and runs it.
func runUnary(t *testing.T, code tflite.BuiltinOperator, in, out tensorSpec, input []int8) []byte {
	t.Helper()
	name := tflite.EnumNamesBuiltinOperator[code]
	it := newSingleOp(t, code, name, nil, []tensorSpec{in, out}, []int{0}, []int{1})
	if err := it.Input(0).SetInt8(input); err != nil {
		t.Fatal(err)
	}
	if err := it.RunOperator(0, 0); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return it.Output(0).Data
}

// sigmoidOutQuant is the only output quantization the int8 sigmoid accepts.
var sigmoidOutQuant = quant(-128, 1.0/256)

func TestLogisticParameters(t *testing.T) {
	// Expectations follow logistic_common.cc by hand: with a power-of-two input scale the
	// multiplier is exactly 2^30 and the radius is 15 * 2^27 / 2^shift.
	cases := []struct {
		scale      float32
		zeroPoint  int32
		multiplier int32
		shift      int
		radius     int32
		note       string
	}{
		{1.0 / 16, 0, 1 << 30, 24, 120, "power of two scale"},
		{0.1895558089017868, 24, 1628272000, 25, 60, "alexa"},
		{0.10491189360618591, -5, 1802372608, 24, 120, "vad"},
	}
	for _, c := range cases {
		t.Run(c.note, func(t *testing.T) {
			in := tensorSpec{shape: []int{1, 1}, dtype: runtime.Int8, quant: quant(c.zeroPoint, c.scale)}
			out := tensorSpec{shape: []int{1, 1}, dtype: runtime.Int8, quant: sigmoidOutQuant}
			it := newSingleOp(t, tflite.BuiltinOperatorLOGISTIC, "LOGISTIC", nil, []tensorSpec{in, out}, []int{0}, []int{1})
			p := it.Params(&it.Model().Main().Operators[0]).(*logisticParams)
			if p.inputZeroPoint != c.zeroPoint || p.inputMultiplier != c.multiplier || p.inputLeftShift != c.shift || p.inputRangeRadius != c.radius {
				t.Errorf("got zero point %d, multiplier %d, shift %d, radius %d; want %d, %d, %d, %d",
					p.inputZeroPoint, p.inputMultiplier, p.inputLeftShift, p.inputRangeRadius,
					c.zeroPoint, c.multiplier, c.shift, c.radius)
			}
		})
	}
}

func TestLogisticValues(t *testing.T) {
	// Expected values were computed with an independent implementation of the same fixed-point
	// formulas and cross-checked against the float sigmoid rounded to 1/256: with these scales
	// both agree on every input. The radius cases exercise the saturation branches.
	cases := []struct {
		note      string
		scale     float32
		zeroPoint int32
		inputs    []int8
		want      []int8
	}{
		{
			"scale 1/16, zero point 0 (radius 120)",
			1.0 / 16, 0,
			[]int8{-128, -121, -120, -119, -64, -32, -16, -8, -1, 0, 1, 8, 16, 32, 64, 119, 120, 127},
			[]int8{-128, -128, -128, -128, -123, -97, -59, -31, -4, 0, 4, 31, 59, 97, 123, 127, 127, 127},
		},
		{
			"alexa dense output (radius 60)",
			0.1895558089017868, 24,
			[]int8{-128, -37, -36, -35, -8, 8, 16, 23, 24, 25, 32, 40, 56, 83, 84, 85, 127},
			[]int8{-128, -128, -128, -128, -127, -116, -82, -12, 0, 12, 82, 116, 127, 127, 127, 127, 127},
		},
		{
			"vad dense output (radius 120)",
			0.10491189360618591, -5,
			[]int8{-128, -125, -124, -69, -37, -21, -13, -6, -5, -4, 3, 11, 27, 59, 114, 115, 127},
			[]int8{-128, -128, -128, -128, -119, -88, -51, -7, 0, 7, 51, 88, 119, 127, 127, 127, 127},
		},
	}
	for _, c := range cases {
		t.Run(c.note, func(t *testing.T) {
			n := len(c.inputs)
			in := tensorSpec{shape: []int{1, n}, dtype: runtime.Int8, quant: quant(c.zeroPoint, c.scale)}
			out := tensorSpec{shape: []int{1, n}, dtype: runtime.Int8, quant: sigmoidOutQuant}
			got := toInt8(runUnary(t, tflite.BuiltinOperatorLOGISTIC, in, out, c.inputs))
			for i := range c.inputs {
				if got[i] != c.want[i] {
					t.Errorf("input %d: expected %d, got %d", c.inputs[i], c.want[i], got[i])
				}
			}
		})
	}
}

func TestLogisticMonotonicAndSymmetric(t *testing.T) {
	// For a zero point of 0 the sigmoid of -x is 1 - sigmoid(x): in the output domain
	// out(x) + out(-x) is 0, or -1 when the fixed-point One() (2^31 - 1 rather than 2^31) tips
	// the rounding of the negative branch.
	inputs := make([]int8, 256)
	for i := range inputs {
		inputs[i] = int8(i - 128)
	}
	in := tensorSpec{shape: []int{1, 256}, dtype: runtime.Int8, quant: quant(0, 0.05)}
	out := tensorSpec{shape: []int{1, 256}, dtype: runtime.Int8, quant: sigmoidOutQuant}
	got := toInt8(runUnary(t, tflite.BuiltinOperatorLOGISTIC, in, out, inputs))
	for i := 1; i < 256; i++ {
		if got[i] < got[i-1] {
			t.Errorf("not monotonic at input %d: %d then %d", inputs[i], got[i-1], got[i])
		}
	}
	if got[128] != 0 {
		t.Errorf("sigmoid(0) should map to 0 (value 1/2), got %d", got[128])
	}
	for x := 1; x < 128; x++ {
		if sum := int(got[128+x]) + int(got[128-x]); sum != 0 && sum != -1 {
			t.Errorf("input %d: out(x) %d + out(-x) %d is neither 0 nor -1", x, got[128+x], got[128-x])
		}
	}
	if got[0] != math.MinInt8 || got[255] != math.MaxInt8 {
		t.Errorf("extremes should saturate: got %d and %d", got[0], got[255])
	}
}

func TestLogisticRejectsWrongOutputQuantization(t *testing.T) {
	in := tensorSpec{shape: []int{1, 1}, dtype: runtime.Int8, quant: quant(0, 0.05)}
	for _, c := range []struct {
		note string
		out  *runtime.Quantization
		want string
	}{
		{"zero point", quant(0, 1.0/256), "zero point 0"},
		{"scale", quant(-128, 0.5), "scale 0.5"},
	} {
		t.Run(c.note, func(t *testing.T) {
			out := tensorSpec{shape: []int{1, 1}, dtype: runtime.Int8, quant: c.out}
			err := tryNewSingleOp(t, tflite.BuiltinOperatorLOGISTIC, nil, []tensorSpec{in, out}, []int{0}, []int{1})
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("expected an error containing %q, got %v", c.want, err)
			}
		})
	}
}

func TestQuantizeInt8ToUint8(t *testing.T) {
	inputs := []int8{-128, -100, -1, 0, 5, 6, 50, 127}
	cases := []struct {
		note    string
		in, out *runtime.Quantization
		want    []uint8
	}{
		{
			// Same scale, zero points -128 and 0: the reference flips the sign bit.
			"same scale, shift by 128",
			quant(-128, 1.0/256), quant(0, 1.0/256),
			[]uint8{0, 28, 127, 128, 133, 134, 178, 255},
		},
		{
			// Effective scale 0.4 with offsets 5 and 3; expectations from the double-rounding
			// MultiplyByQuantizedMultiplier arithmetic, clamped to [0, 255].
			"general requantization",
			quant(5, 0.02), quant(3, 0.05),
			[]uint8{0, 0, 0, 1, 3, 4, 21, 52},
		},
	}
	for _, c := range cases {
		t.Run(c.note, func(t *testing.T) {
			in := tensorSpec{shape: []int{1, len(inputs)}, dtype: runtime.Int8, quant: c.in}
			out := tensorSpec{shape: []int{1, len(inputs)}, dtype: runtime.UInt8, quant: c.out}
			got := runUnary(t, tflite.BuiltinOperatorQUANTIZE, in, out, inputs)
			for i := range inputs {
				if got[i] != c.want[i] {
					t.Errorf("input %d: expected %d, got %d", inputs[i], c.want[i], got[i])
				}
			}
		})
	}
}
