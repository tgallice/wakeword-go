package frontend

import "testing"

func TestKissFFTSetup(t *testing.T) {
	f, err := newFFT(480)
	if err != nil {
		t.Fatal(err)
	}
	if f.fftSize != 512 || len(f.output) != 257 {
		t.Fatalf("fft size %d bins %d", f.fftSize, len(f.output))
	}
	// 256 = 4 * 4 * 4 * 4, the factor list kf_factor produces.
	want := []int{4, 64, 4, 16, 4, 4, 4, 1}
	got := f.fftr.substate.factors
	if len(got) != len(want) {
		t.Fatalf("factors %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("factors %v, want %v", got, want)
		}
	}
	// Twiddle 0 is 1, twiddle nfft/4 is -i (rounded to Q15).
	tw := f.fftr.substate.twiddles
	if tw[0] != (cpx{32767, 0}) || tw[64] != (cpx{0, -32767}) {
		t.Fatalf("twiddles %v %v", tw[0], tw[64])
	}
	// First super twiddle, as kiss_fftr_alloc computes it for nfft 256.
	if f.fftr.superTwiddles[0] != (cpx{-402, -32765}) {
		t.Fatalf("super twiddle 0 = %v", f.fftr.superTwiddles[0])
	}
	if _, err := newKissFFT(96); err == nil {
		t.Fatal("radix 3 sizes must be rejected")
	}
	if _, err := newKissFFTR(9); err == nil {
		t.Fatal("odd sizes must be rejected")
	}
}

// TestFFTDC checks a constant input: with the per stage scaling of the fixed
// point transform, a DC level of 8192 lands entirely in bin 0 with the same
// value, and every other bin is exactly zero.
func TestFFTDC(t *testing.T) {
	f, err := newFFT(512)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]int16, 512)
	for i := range in {
		in[i] = 8192
	}
	f.compute(in, 0)
	if f.output[0] != (cpx{8192, 0}) {
		t.Fatalf("bin 0 = %v, want {8192 0}", f.output[0])
	}
	for k := 1; k < len(f.output); k++ {
		if f.output[k] != (cpx{0, 0}) {
			t.Fatalf("bin %d = %v, want zero", k, f.output[k])
		}
	}
}

// TestFFTImpulse checks a unit impulse: its spectrum is flat and real. The
// fixed point rounding may move a bin by one unit.
func TestFFTImpulse(t *testing.T) {
	f, err := newFFT(512)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]int16, 512)
	in[0] = 16384
	f.compute(in, 0)
	level := f.output[0].r
	if level < 30 || level > 34 {
		t.Fatalf("impulse level %d, want about 32 (16384 scaled by 1/512)", level)
	}
	for k, bin := range f.output {
		if d := int(bin.r) - int(level); d < -1 || d > 1 || bin.i < -1 || bin.i > 1 {
			t.Fatalf("bin %d = %v, want a flat real spectrum near %d", k, bin, level)
		}
	}
}

// TestFFTInputShift checks the unsigned shift of the input with wraparound.
func TestFFTInputShift(t *testing.T) {
	f, err := newFFT(4)
	if err != nil {
		t.Fatal(err)
	}
	f.compute([]int16{1, -1, 0x4000, 3}, 2)
	want := []int16{4, -4, 0, 12}
	for i, v := range want {
		if f.input[i] != v {
			t.Fatalf("input[%d] = %d, want %d", i, f.input[i], v)
		}
	}
}

func TestSround(t *testing.T) {
	tests := []struct {
		in   int32
		want int16
	}{
		{0, 0},
		{1 << 14, 1},
		{(1 << 14) - 1, 0},
		{-(1 << 14), 0},
		{-(1 << 14) - 1, -1},
		{32767 << 15, 32767},
		{(32768 << 15), -32768}, // truncation to int16, as the C cast
	}
	for _, tc := range tests {
		if got := sround(tc.in); got != tc.want {
			t.Errorf("sround(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
	// C_FIXDIV by 2 multiplies by 16383/32768, so 32767 maps to 16383.
	if got := divScalar(32767, 2); got != 16383 {
		t.Errorf("divScalar(32767, 2) = %d, want 16383", got)
	}
}
