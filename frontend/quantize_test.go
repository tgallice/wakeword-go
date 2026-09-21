package frontend

import "testing"

func TestQuantizeFeature(t *testing.T) {
	tests := []struct {
		feature uint16
		want    int8
	}{
		{0, -128},
		{1, -128},  // (256 + 333) / 666 = 0
		{2, -127},  // (512 + 333) / 666 = 1
		{333, 0},   // 128 - 128
		{664, 127}, // (169984 + 333) / 666 = 255
		{665, 127}, // 255
		{666, 127}, // 256 - 128 = 128, clamped to 127
		{669, 127},
		{65535, 127},
	}
	for _, tc := range tests {
		if got := QuantizeFeature(tc.feature); got != tc.want {
			t.Errorf("QuantizeFeature(%d) = %d, want %d", tc.feature, got, tc.want)
		}
	}
	// The whole int8 range is reachable and monotonic over the useful range.
	prev := QuantizeFeature(0)
	for v := uint16(1); v <= 700; v++ {
		got := QuantizeFeature(v)
		if got < prev {
			t.Fatalf("not monotonic at %d: %d after %d", v, got, prev)
		}
		prev = got
	}
}

func TestQuantizeFeatures(t *testing.T) {
	src := []uint16{0, 333, 665, 700}
	dst := make([]int8, 4)
	QuantizeFeatures(dst, src)
	want := []int8{-128, 0, 127, 127}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst = %v, want %v", dst, want)
		}
	}
}
