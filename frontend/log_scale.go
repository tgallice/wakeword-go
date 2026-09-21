package frontend

const (
	logSegmentsLog2 = 7
	logScale        = 65536
	logScaleLog2    = 16
	// logCoeff converts log2 to the natural logarithm in Q16 (kLogCoeff).
	logCoeff  = 45426
	uint16Max = 0xFFFF
)

// logLUT is kLogLut: the correction of the linear log2 approximation over
// kLogSegments segments of the fraction, plus padding.
var logLUT = [...]uint16{
	0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163,
	2329, 2490, 2646, 2797, 2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848,
	3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714, 4791, 4864, 4934,
	5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
	5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633,
	5626, 5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370,
	5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762,
	4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
	3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659,
	2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224,
	1094, 963, 830, 695, 559, 421, 282, 142, 0, 0,
}

// logScaler is the port of log_scale.cc: a fixed point natural logarithm.
type logScaler struct {
	enableLog  bool
	scaleShift int
}

// log2FractionPart is Log2FractionPart.
func log2FractionPart(x, log2x uint32) uint32 {
	frac := int32(int64(x) - (int64(1) << log2x))
	if log2x < logScaleLog2 {
		frac <<= logScaleLog2 - log2x
	} else {
		frac >>= log2x - logScaleLog2
	}
	baseSeg := uint32(frac >> (logScaleLog2 - logSegmentsLog2))
	const segUnit = uint32(1<<logScaleLog2) >> logSegmentsLog2

	c0 := int32(logLUT[baseSeg])
	c1 := int32(logLUT[baseSeg+1])
	segBase := int32(segUnit * baseSeg)
	relPos := ((c1 - c0) * (frac - segBase)) >> logScaleLog2
	return uint32(frac + c0 + relPos)
}

// fixedLog is Log: the natural logarithm of x scaled by 2^scaleShift.
func fixedLog(x uint32, scaleShift int) uint32 {
	integer := uint32(mostSignificantBit32(x) - 1)
	fraction := log2FractionPart(x, integer)
	log2 := (integer << logScaleLog2) + fraction
	const round = logScale / 2
	loge := uint32((uint64(logCoeff)*uint64(log2) + round) >> logScaleLog2)
	return ((loge << uint(scaleShift)) + round) >> logScaleLog2
}

// apply is LogScaleApply: signal is corrected by correctionBits, logged and
// saturated to uint16 into output.
func (l *logScaler) apply(signal []uint32, output []uint16, correctionBits int) {
	for i, value := range signal {
		if l.enableLog {
			if correctionBits < 0 {
				value >>= uint(-correctionBits)
			} else {
				value <<= uint(correctionBits)
			}
			if value > 1 {
				value = fixedLog(value, l.scaleShift)
			} else {
				value = 0
			}
		}
		if value < uint16Max {
			output[i] = uint16(value)
		} else {
			output[i] = uint16Max
		}
	}
}
