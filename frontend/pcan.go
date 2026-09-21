package frontend

import "math"

const (
	pcanSnrBits    = 12
	pcanOutputBits = 6
	// wideDynamicFunctionBits is the number of bit intervals covered by the
	// gain lookup table (kWideDynamicFunctionBits).
	wideDynamicFunctionBits    = 32
	wideDynamicFunctionLUTSize = 4*wideDynamicFunctionBits - 3
	int16Max                   = 0x7FFF
)

// pcanGainControl is the port of pcan_gain_control.cc and
// pcan_gain_control_util.cc: per channel automatic gain normalization driven
// by the noise estimate, with a piecewise quadratic lookup table of the gain.
type pcanGainControl struct {
	noiseEstimate []uint32 // shared with the noise reduction stage
	numChannels   int
	gainLUT       []int16 // indexed with the C offset of -6 already applied
	snrShift      int
}

// pcanGainLookupFunction is PcanGainLookupFunction, in float32 like the C code.
func pcanGainLookupFunction(strength, offset float32, gainBits, inputBits int, x uint32) int16 {
	xAsFloat := float32(float32(x) / float32(uint32(1)<<uint(inputBits)))
	gainAsFloat := float32(float32(uint32(1)<<uint(gainBits)) * powf(float32(xAsFloat+offset), -strength))
	if gainAsFloat > int16Max {
		return int16Max
	}
	return int16(float32(gainAsFloat + 0.5))
}

func powf(x, y float32) float32 { return float32(math.Pow(float64(x), float64(y))) }

func newPcanGainControl(strength, offset float32, gainBits int, noiseEstimate []uint32,
	numChannels, smoothingBits, inputCorrectionBits int,
) *pcanGainControl {
	p := &pcanGainControl{
		noiseEstimate: noiseEstimate,
		numChannels:   numChannels,
		gainLUT:       make([]int16, wideDynamicFunctionLUTSize),
		snrShift:      gainBits - inputCorrectionBits - pcanSnrBits,
	}
	inputBits := smoothingBits - inputCorrectionBits
	p.gainLUT[0] = pcanGainLookupFunction(strength, offset, gainBits, inputBits, 0)
	p.gainLUT[1] = pcanGainLookupFunction(strength, offset, gainBits, inputBits, 1)
	// The C code offsets the pointer by -6 and writes lut[4*interval + k];
	// the same slots are addressed here as 4*interval - 6 + k.
	for interval := 2; interval <= wideDynamicFunctionBits; interval++ {
		x0 := uint32(1) << uint(interval-1)
		x1 := x0 + (x0 >> 1)
		x2 := 2 * x0
		if interval == wideDynamicFunctionBits {
			x2 = x0 + (x0 - 1)
		}
		y0 := pcanGainLookupFunction(strength, offset, gainBits, inputBits, x0)
		y1 := pcanGainLookupFunction(strength, offset, gainBits, inputBits, x1)
		y2 := pcanGainLookupFunction(strength, offset, gainBits, inputBits, x2)

		diff1 := int32(y1) - int32(y0)
		diff2 := int32(y2) - int32(y0)
		a1 := 4*diff1 - diff2
		a2 := diff2 - a1

		base := 4*interval - 6
		p.gainLUT[base] = y0
		p.gainLUT[base+1] = int16(a1)
		p.gainLUT[base+2] = int16(a2)
	}
	return p
}

// wideDynamicFunction is WideDynamicFunction: the interpolated gain for a
// noise estimate x.
func wideDynamicFunction(x uint32, lut []int16) int16 {
	if x <= 2 {
		return lut[x]
	}
	interval := mostSignificantBit32(x)
	base := 4*interval - 6

	var frac int16
	if interval < 11 {
		frac = int16((x << uint(11-interval)) & 0x3FF)
	} else {
		frac = int16((x >> uint(interval-11)) & 0x3FF)
	}

	result := (int32(lut[base+2]) * int32(frac)) >> 5
	result += int32(uint32(int32(lut[base+1])) << 5)
	result *= int32(frac)
	result = (result + (1 << 14)) >> 15
	result += int32(lut[base])
	return int16(result)
}

// pcanShrink is PcanShrink.
func pcanShrink(x uint32) uint32 {
	if x < (2 << pcanSnrBits) {
		return (x * x) >> (2 + 2*pcanSnrBits - pcanOutputBits)
	}
	return (x >> (pcanSnrBits - pcanOutputBits)) - (1 << pcanOutputBits)
}

// apply is PcanGainControlApply, in place on signal.
func (p *pcanGainControl) apply(signal []uint32) {
	for i := range p.numChannels {
		gain := uint32(int32(wideDynamicFunction(p.noiseEstimate[i], p.gainLUT)))
		snr := uint32((uint64(signal[i]) * uint64(gain)) >> uint(p.snrShift))
		signal[i] = pcanShrink(snr)
	}
}
