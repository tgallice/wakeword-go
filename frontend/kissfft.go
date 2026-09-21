package frontend

import (
	"fmt"
	"math"
)

// This file is a port of kissfft (kiss_fft.c and tools/kiss_fftr.c, Mark
// Borgerding, BSD license) compiled with FIXED_POINT=16, as embedded in the
// tflite-micro microfrontend. Samples are int16, products are int32, and every
// butterfly stage scales its inputs down by the radix so the transform never
// overflows. The rounding macros (sround, C_FIXDIV, C_MUL) are reproduced
// exactly; they are part of the observable result.

// cpx is kiss_fft_cpx with kiss_fft_scalar = int16.
type cpx struct {
	r, i int16
}

const (
	fracBits = 15
	sampMax  = 32767
	// kissPi is the literal used by kiss_fftr_alloc for the super twiddles.
	kissPi = 3.14159265358979323846264338327
	// twiddlePi is the literal used by kiss_fft_alloc for the twiddles.
	twiddlePi = 3.141592653589793238462643383279502884197169399375105820974944
)

// sround rounds a Q15 product back to int16 (the sround macro).
func sround(x int32) int16 {
	return int16((x + (1 << (fracBits - 1))) >> fracBits)
}

// smul is the int32 product of two int16 (the smul macro).
func smul(a, b int16) int32 {
	return int32(a) * int32(b)
}

// cMul is C_MUL: complex product with rounding of each component.
func cMul(a, b cpx) cpx {
	return cpx{
		r: sround(smul(a.r, b.r) - smul(a.i, b.i)),
		i: sround(smul(a.r, b.i) + smul(a.i, b.r)),
	}
}

// divScalar is DIVSCALAR: x scaled by SAMP_MAX/k (integer division) with rounding.
func divScalar(x int16, k int32) int16 {
	return sround(int32(x) * (sampMax / k))
}

// cFixDiv is C_FIXDIV: scale both components of c down by div.
func cFixDiv(c *cpx, div int32) {
	c.r = divScalar(c.r, div)
	c.i = divScalar(c.i, div)
}

// cAdd and cSub wrap on int16 exactly like the C assignments do.
func cAdd(a, b cpx) cpx { return cpx{a.r + b.r, a.i + b.i} }
func cSub(a, b cpx) cpx { return cpx{a.r - b.r, a.i - b.i} }

// kissFFT is kiss_fft_state: a complex FFT of nfft points.
type kissFFT struct {
	nfft     int
	factors  []int // radix, remaining length pairs, as kf_factor fills them
	twiddles []cpx
}

// newKissFFT is kiss_fft_alloc for a forward transform.
func newKissFFT(nfft int) (*kissFFT, error) {
	st := &kissFFT{nfft: nfft, twiddles: make([]cpx, nfft)}
	for i := range nfft {
		phase := -2 * twiddlePi * float64(i) / float64(nfft)
		st.twiddles[i] = cexp(phase)
	}
	st.factors = factor(nfft)
	for p := 0; p < len(st.factors); p += 2 {
		if radix := st.factors[p]; radix != 2 && radix != 4 {
			return nil, fmt.Errorf("frontend: fft size %d needs radix %d, only 2 and 4 are ported", nfft, radix)
		}
	}
	return st, nil
}

// cexp is kf_cexp in fixed point: floor(0.5 + SAMP_MAX * cos/sin(phase)),
// computed in float64 and truncated to int16 as the C assignment does.
func cexp(phase float64) cpx {
	return cpx{
		r: int16(math.Floor(0.5 + sampMax*math.Cos(phase))),
		i: int16(math.Floor(0.5 + sampMax*math.Sin(phase))),
	}
}

// factor is kf_factor: powers of 4, then 2, then odd primes.
func factor(n int) []int {
	var out []int
	p := 4
	floorSqrt := int(math.Floor(math.Sqrt(float64(n))))
	for {
		for n%p != 0 {
			switch p {
			case 4:
				p = 2
			case 2:
				p = 3
			default:
				p += 2
			}
			if p > floorSqrt {
				p = n
			}
		}
		n /= p
		out = append(out, p, n)
		if n <= 1 {
			return out
		}
	}
}

// bfly2 is kf_bfly2.
func (st *kissFFT) bfly2(fout []cpx, fstride, m int) {
	tw := 0
	for k := range m {
		cFixDiv(&fout[k], 2)
		cFixDiv(&fout[k+m], 2)
		t := cMul(fout[k+m], st.twiddles[tw])
		tw += fstride
		fout[k+m] = cSub(fout[k], t)
		fout[k] = cAdd(fout[k], t)
	}
}

// bfly4 is kf_bfly4 for a forward transform.
func (st *kissFFT) bfly4(fout []cpx, fstride, m int) {
	var scratch [6]cpx
	tw1, tw2, tw3 := 0, 0, 0
	m2, m3 := 2*m, 3*m
	for k := range m {
		cFixDiv(&fout[k], 4)
		cFixDiv(&fout[k+m], 4)
		cFixDiv(&fout[k+m2], 4)
		cFixDiv(&fout[k+m3], 4)

		scratch[0] = cMul(fout[k+m], st.twiddles[tw1])
		scratch[1] = cMul(fout[k+m2], st.twiddles[tw2])
		scratch[2] = cMul(fout[k+m3], st.twiddles[tw3])

		scratch[5] = cSub(fout[k], scratch[1])
		fout[k] = cAdd(fout[k], scratch[1])
		scratch[3] = cAdd(scratch[0], scratch[2])
		scratch[4] = cSub(scratch[0], scratch[2])
		fout[k+m2] = cSub(fout[k], scratch[3])
		tw1 += fstride
		tw2 += fstride * 2
		tw3 += fstride * 3
		fout[k] = cAdd(fout[k], scratch[3])

		fout[k+m].r = scratch[5].r + scratch[4].i
		fout[k+m].i = scratch[5].i - scratch[4].r
		fout[k+m3].r = scratch[5].r - scratch[4].i
		fout[k+m3].i = scratch[5].i + scratch[4].r
	}
}

// work is kf_work: the recursive mixed radix decimation in time. fin is read
// starting at index fidx with stride fstride*inStride; fout receives p*m points.
func (st *kissFFT) work(fout, fin []cpx, fidx, fstride, inStride int, factors []int) {
	p, m := factors[0], factors[1]
	rest := factors[2:]
	if m == 1 {
		for j := range p {
			fout[j] = fin[fidx+j*fstride*inStride]
		}
	} else {
		for j := range p {
			st.work(fout[j*m:(j+1)*m], fin, fidx+j*fstride*inStride, fstride*p, inStride, rest)
		}
	}
	switch p {
	case 2:
		st.bfly2(fout, fstride, m)
	case 4:
		st.bfly4(fout, fstride, m)
	}
}

// transform is kiss_fft: fout must not alias fin.
func (st *kissFFT) transform(fin, fout []cpx) {
	st.work(fout, fin, 0, 1, 1, st.factors)
}

// kissFFTR is kiss_fftr_state: a real FFT of nfft points built on a complex
// FFT of nfft/2 points.
type kissFFTR struct {
	substate      *kissFFT
	tmpbuf        []cpx
	superTwiddles []cpx
	packed        []cpx // the real input viewed as nfft/2 complex points
}

// newKissFFTR is kiss_fftr_alloc for a forward transform of nfft real points.
func newKissFFTR(nfft int) (*kissFFTR, error) {
	if nfft&1 != 0 {
		return nil, fmt.Errorf("frontend: real fft size must be even, got %d", nfft)
	}
	ncfft := nfft >> 1
	sub, err := newKissFFT(ncfft)
	if err != nil {
		return nil, err
	}
	st := &kissFFTR{
		substate:      sub,
		tmpbuf:        make([]cpx, ncfft),
		superTwiddles: make([]cpx, ncfft/2),
		packed:        make([]cpx, ncfft),
	}
	for i := range ncfft / 2 {
		phase := -kissPi * (float64(i+1)/float64(ncfft) + 0.5)
		st.superTwiddles[i] = cexp(phase)
	}
	return st, nil
}

// transform is kiss_fftr: timedata has nfft samples, freqdata receives
// nfft/2+1 bins.
func (st *kissFFTR) transform(timedata []int16, freqdata []cpx) {
	ncfft := st.substate.nfft
	for k := range ncfft {
		st.packed[k] = cpx{r: timedata[2*k], i: timedata[2*k+1]}
	}
	st.substate.transform(st.packed, st.tmpbuf)

	tdc := st.tmpbuf[0]
	cFixDiv(&tdc, 2)
	freqdata[0].r = tdc.r + tdc.i
	freqdata[ncfft].r = tdc.r - tdc.i
	freqdata[0].i = 0
	freqdata[ncfft].i = 0

	for k := 1; k <= ncfft/2; k++ {
		fpk := st.tmpbuf[k]
		fpnk := cpx{r: st.tmpbuf[ncfft-k].r, i: -st.tmpbuf[ncfft-k].i}
		cFixDiv(&fpk, 2)
		cFixDiv(&fpnk, 2)

		f1k := cAdd(fpk, fpnk)
		f2k := cSub(fpk, fpnk)
		tw := cMul(f2k, st.superTwiddles[k-1])

		// HALF_OF works on the int promoted sum, so add in int32 before halving.
		freqdata[k].r = int16((int32(f1k.r) + int32(tw.r)) >> 1)
		freqdata[k].i = int16((int32(f1k.i) + int32(tw.i)) >> 1)
		freqdata[ncfft-k].r = int16((int32(f1k.r) - int32(tw.r)) >> 1)
		freqdata[ncfft-k].i = int16((int32(tw.i) - int32(f1k.i)) >> 1)
	}
}
