package frontend

import (
	"math"
	"testing"
)

func logFloat(x uint32) float64 { return math.Log(float64(x)) }

// TestFilterbankTables checks the band layout produced by the default
// configuration against the values of filterbank_util.cc: DC excluded, bins 5
// to 240 used, 41 accumulators (40 channels plus the unweighted tail).
func TestFilterbankTables(t *testing.T) {
	fb, err := newFilterbank(40, 125, 7500, 16000, 257)
	if err != nil {
		t.Fatal(err)
	}
	if fb.startIndex != 5 || fb.endIndex != 241 {
		t.Fatalf("start %d end %d, want 5 and 241", fb.startIndex, fb.endIndex)
	}
	wantStarts := []int16{
		4, 6, 8, 8, 10, 12, 14, 16, 18, 22, 24, 26, 30, 32, 36, 38, 42, 46, 50, 54, 58,
		64, 68, 74, 78, 84, 90, 98, 104, 112, 120, 128, 136, 146, 154, 166, 176, 188, 200, 212, 226,
	}
	wantWidths := []int16{
		4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
		8, 8, 8, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16,
	}
	for i := range wantStarts {
		if fb.channelFrequencyStarts[i] != wantStarts[i] || fb.channelWidths[i] != wantWidths[i] {
			t.Fatalf("channel %d: start %d width %d, want %d and %d", i,
				fb.channelFrequencyStarts[i], fb.channelWidths[i], wantStarts[i], wantWidths[i])
		}
	}
	if len(fb.weights) != 316 || len(fb.unweights) != 316 {
		t.Fatalf("%d weights, want 316", len(fb.weights))
	}
	// Weight and unweight of a used bin sum to full scale (up to rounding);
	// padding bins are zero on both sides.
	for i := range fb.weights {
		w, u := int(fb.weights[i]), int(fb.unweights[i])
		if w == 0 && u == 0 {
			continue
		}
		if s := w + u; s < (1<<filterbankBits)-1 || s > (1<<filterbankBits)+1 {
			t.Fatalf("weight %d: %d + %d = %d, want about %d", i, w, u, s, 1<<filterbankBits)
		}
	}
	if _, err := newFilterbank(40, 125, 9000, 16000, 257); err == nil {
		t.Fatal("a top band above the spectrum must be rejected")
	}
}

func TestSqrt(t *testing.T) {
	for _, n := range []uint64{
		0, 1, 2, 3, 4, 15, 16, 17, 255, 256, 65535, 65536, 1 << 31, 1<<32 - 1,
		1 << 32, 1<<32 + 1, 1 << 40, 123456789012, 1<<64 - 1,
	} {
		got := float64(sqrt64(n))
		// Integer square root rounded to nearest, within one unit of the float
		// result (the C rounding rule is approximate and saturates at 32 bits).
		want := math.Round(math.Sqrt(float64(n)))
		if d := got - want; d < -1 || d > 1 {
			t.Errorf("sqrt64(%d) = %v, want about %v", n, got, want)
		}
	}
	if got := sqrt64(1 << 62); got != 1<<31 {
		t.Errorf("sqrt64(2^62) = %d, want 2^31", got)
	}
	if got := sqrt32(1 << 30); got != 1<<15 {
		t.Errorf("sqrt32(2^30) = %d, want 2^15", got)
	}
}

func TestNoiseReductionSetup(t *testing.T) {
	nr := newNoiseReduction(10, 0.025, 0.06, 0.05, 2)
	if nr.evenSmoothing != 409 || nr.oddSmoothing != 983 || nr.minSignalRemaining != 819 {
		t.Fatalf("smoothing %d %d floor %d, want 409 983 819", nr.evenSmoothing, nr.oddSmoothing, nr.minSignalRemaining)
	}
	// With a zero estimate the signal passes almost untouched: the noise
	// estimate takes only a fraction of it.
	signal := []uint32{1000, 1000}
	nr.apply(signal)
	if signal[0] != 975 || signal[1] != 940 {
		t.Fatalf("first frame %v, want [975 940]", signal)
	}
	// (1000 << 10) * 409 >> 14 and (1000 << 10) * 983 >> 14, truncated.
	if nr.estimate[0] != 25562 || nr.estimate[1] != 61437 {
		t.Fatalf("estimate %v, want [25562 61437]", nr.estimate[:2])
	}
	// A stationary signal converges to the floor: 1000 * 819 >> 14 = 49.
	for range 2000 {
		signal[0], signal[1] = 1000, 1000
		nr.apply(signal)
	}
	if signal[0] != 49 || signal[1] != 49 {
		t.Fatalf("converged %v, want [49 49]", signal)
	}
	nr.reset()
	if nr.estimate[0] != 0 {
		t.Fatal("reset did not clear the estimate")
	}
}

func TestPcanGainTable(t *testing.T) {
	estimate := make([]uint32, 40)
	p := newPcanGainControl(0.95, 80, 21, estimate, 40, 10, 3)
	if p.snrShift != 6 {
		t.Fatalf("snr shift %d, want 6", p.snrShift)
	}
	// Leading entries of the gain table computed by pcan_gain_control_util.cc
	// for the default configuration.
	want := map[int]int16{
		0: 32636, 1: 32633, 2: 32630, 3: -6, 4: 0, 6: 32624, 7: -12, 8: 0,
		10: 32612, 11: -23, 12: -2, 46: 23707, 47: -6265, 48: 1230, 50: 18672, 51: -7458, 52: 1952,
	}
	for i, v := range want {
		if p.gainLUT[i] != v {
			t.Errorf("gain lut[%d] = %d, want %d", i, p.gainLUT[i], v)
		}
	}
	// The interpolated gain is monotonically decreasing in the noise estimate.
	prev := wideDynamicFunction(0, p.gainLUT)
	for x := uint32(1); x < 1<<20; x = x*3/2 + 1 {
		g := wideDynamicFunction(x, p.gainLUT)
		if g > prev {
			t.Fatalf("gain increases at %d: %d after %d", x, g, prev)
		}
		prev = g
	}
	if pcanShrink(0) != 0 || pcanShrink(8191) != (8191*8191)>>20 || pcanShrink(8192) != 128-64 {
		t.Fatal("pcanShrink boundaries")
	}
}

func TestFixedLog(t *testing.T) {
	// ln(x) * 2^6 with the C approximation, checked against float math with a
	// small tolerance: the table interpolation is accurate to a few units.
	for _, x := range []uint32{2, 3, 7, 100, 1000, 65536, 1 << 20, 1<<32 - 1} {
		got := float64(fixedLog(x, 6))
		want := 64 * logFloat(x)
		if d := got - want; d < -2 || d > 2 {
			t.Errorf("fixedLog(%d) = %v, want about %v", x, got, want)
		}
	}
	l := &logScaler{enableLog: true, scaleShift: 6}
	out := make([]uint16, 3)
	l.apply([]uint32{0, 1, 1 << 31}, out, 0)
	if out[0] != 0 || out[1] != 0 {
		t.Fatalf("values up to 1 must give 0, got %v", out)
	}
	if out[2] != uint16(fixedLog(1<<31, 6)) || out[2] < 1370 || out[2] > 1380 {
		t.Fatalf("log of 2^31: %d, want 64 * ln(2^31)", out[2])
	}
	l.apply([]uint32{1 << 28}, out, 3)
	if out[0] != uint16(fixedLog(1<<31, 6)) {
		t.Fatalf("positive correction: %d", out[0])
	}
	l.apply([]uint32{1 << 29}, out, -1)
	if out[0] != uint16(fixedLog(1<<28, 6)) {
		t.Fatalf("negative correction: %d", out[0])
	}
	l.enableLog = false
	l.apply([]uint32{70000, 5}, out, 0)
	if out[0] != 65535 || out[1] != 5 {
		t.Fatalf("log disabled: %v", out[:2])
	}
}
