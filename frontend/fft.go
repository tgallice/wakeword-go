package frontend

// fft is the port of fft.cc and fft_util.cc: it scales the windowed frame up
// by a per frame shift, zero pads it to the next power of two and runs the
// fixed point real FFT.
type fft struct {
	inputSize int
	fftSize   int
	input     []int16
	output    []cpx // fftSize/2 + 1 bins
	fftr      *kissFFTR
}

func newFFT(inputSize int) (*fft, error) {
	f := &fft{inputSize: inputSize, fftSize: 1}
	for f.fftSize < inputSize {
		f.fftSize <<= 1
	}
	f.input = make([]int16, f.fftSize)
	f.output = make([]cpx, f.fftSize/2+1)
	fftr, err := newKissFFTR(f.fftSize)
	if err != nil {
		return nil, err
	}
	f.fftr = fftr
	return f, nil
}

// compute is FftCompute: input is shifted left by inputScaleShift (as an
// unsigned 16 bit quantity, wrapping like the C cast) before the transform.
func (f *fft) compute(input []int16, inputScaleShift int) {
	for i := range f.inputSize {
		f.input[i] = int16(uint16(input[i]) << uint(inputScaleShift))
	}
	clear(f.input[f.inputSize:])
	f.fftr.transform(f.input, f.output)
}

func (f *fft) reset() {
	clear(f.input)
	clear(f.output)
}
