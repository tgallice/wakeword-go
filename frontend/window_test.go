package frontend

import "testing"

func TestWindowCoefficients(t *testing.T) {
	w := newWindow(30, 10, 16000)
	if w.size != 480 || w.step != 160 || len(w.coefficients) != 480 {
		t.Fatalf("size %d step %d coefficients %d", w.size, w.step, len(w.coefficients))
	}
	// Leading values of the Q12 Hann window, as computed by window_util.cc.
	want := []int16{0, 0, 1, 2, 4, 5, 7, 10, 13, 16, 19, 23, 27, 32, 37, 42, 48, 54, 60, 66}
	for i, v := range want {
		if w.coefficients[i] != v {
			t.Errorf("coefficient[%d] = %d, want %d", i, w.coefficients[i], v)
		}
	}
	// The window is symmetric around its center up to the float32 rounding of
	// the C computation, and peaks at full scale there.
	for i := range w.size / 2 {
		if d := int(w.coefficients[i]) - int(w.coefficients[w.size-1-i]); d < -1 || d > 1 {
			t.Errorf("coefficient[%d] = %d, mirror %d", i, w.coefficients[i], w.coefficients[w.size-1-i])
		}
	}
	if c := w.coefficients[w.size/2]; c != 1<<windowBits {
		t.Errorf("center coefficient %d, want %d", c, 1<<windowBits)
	}
	for i := 1; i < w.size/2; i++ {
		if w.coefficients[i] < w.coefficients[i-1] {
			t.Fatalf("coefficients not monotonic at %d", i)
		}
	}
}

func TestWindowProcess(t *testing.T) {
	w := newWindow(30, 10, 16000)
	full := make([]int16, 480)
	for i := range full {
		full[i] = 1000
	}
	ok, consumed := w.process(full[:100])
	if ok || consumed != 100 || w.inputUsed != 100 {
		t.Fatalf("partial: ok %v consumed %d used %d", ok, consumed, w.inputUsed)
	}
	ok, consumed = w.process(full)
	if !ok || consumed != 380 || w.inputUsed != 320 {
		t.Fatalf("complete: ok %v consumed %d used %d", ok, consumed, w.inputUsed)
	}
	// Center sample times the full scale coefficient is unchanged, edges vanish.
	if w.output[240] != 1000 || w.output[0] != 0 {
		t.Fatalf("output center %d edge %d", w.output[240], w.output[0])
	}
	if w.maxAbsOutputValue != 1000 {
		t.Fatalf("max abs %d, want 1000", w.maxAbsOutputValue)
	}
	// Negative input: the product is truncated toward minus infinity by the
	// arithmetic shift and the maximum tracks the absolute value.
	for i := range full {
		full[i] = -1000
	}
	w.reset()
	if ok, _ = w.process(full); !ok {
		t.Fatal("expected a window")
	}
	if w.output[240] != -1000 || w.maxAbsOutputValue != 1000 {
		t.Fatalf("negative: center %d max abs %d", w.output[240], w.maxAbsOutputValue)
	}
	w.reset()
	if w.inputUsed != 0 || w.maxAbsOutputValue != 0 || w.output[240] != 0 {
		t.Fatal("reset did not clear the state")
	}
}
