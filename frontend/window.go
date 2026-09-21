package frontend

import "math"

// windowBits is the fixed point precision of the window coefficients
// (kFrontendWindowBits in window.h).
const windowBits = 12

// window is the port of window.cc and window_util.cc: a sliding buffer of
// size samples advanced by step samples, multiplied by a Q12 Hann window.
type window struct {
	size         int
	step         int
	coefficients []int16

	input             []int16
	inputUsed         int
	output            []int16
	maxAbsOutputValue int16
}

func newWindow(sizeMS, stepMS, sampleRate int) *window {
	w := &window{
		size: sizeMS * sampleRate / 1000,
		step: stepMS * sampleRate / 1000,
	}
	w.coefficients = make([]int16, w.size)
	w.input = make([]int16, w.size)
	w.output = make([]int16, w.size)

	// Same float32 arithmetic as WindowPopulateState: the explicit float32
	// conversions round every intermediate result and prevent fused operations.
	arg := float32(math.Pi) * 2.0 / float32(w.size)
	for i := range w.size {
		x := float32(arg * (float32(i) + 0.5))
		floatValue := float32(0.5 - float32(0.5*cosf(x)))
		w.coefficients[i] = int16(floorf(float32(floatValue*(1<<windowBits)) + 0.5))
	}
	return w
}

// process copies as many samples as fit into the input buffer and reports how
// many were consumed. It returns true when a full window has been produced in
// w.output, after which the input buffer is shifted down by step samples.
func (w *window) process(samples []int16) (ok bool, consumed int) {
	consumed = min(w.size-w.inputUsed, len(samples))
	copy(w.input[w.inputUsed:], samples[:consumed])
	w.inputUsed += consumed
	if w.inputUsed < w.size {
		return false, consumed
	}

	var maxAbs int16
	for i, in := range w.input {
		// (int32)input * coefficient >> 12, truncated to int16 as in C.
		newValue := int16((int32(in) * int32(w.coefficients[i])) >> windowBits)
		w.output[i] = newValue
		if newValue < 0 {
			newValue = -newValue
		}
		if newValue > maxAbs {
			maxAbs = newValue
		}
	}
	copy(w.input, w.input[w.step:])
	w.inputUsed -= w.step
	w.maxAbsOutputValue = maxAbs
	return true, consumed
}

func (w *window) reset() {
	clear(w.input)
	clear(w.output)
	w.inputUsed = 0
	w.maxAbsOutputValue = 0
}

// cosf and floorf are the float32 counterparts of the C library calls used at
// initialization. Computing in float64 and rounding once gives the correctly
// rounded float32 result.
func cosf(x float32) float32   { return float32(math.Cos(float64(x))) }
func floorf(x float32) float32 { return float32(math.Floor(float64(x))) }
