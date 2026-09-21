package frontend

// noiseReductionBits is the fixed point precision of the smoothing factors
// (kNoiseReductionBits).
const noiseReductionBits = 14

// noiseReduction is the port of noise_reduction.cc: a per channel low pass
// estimate of the stationary noise, subtracted from the signal with a floor.
type noiseReduction struct {
	smoothingBits      int
	evenSmoothing      uint16
	oddSmoothing       uint16
	minSignalRemaining uint16
	numChannels        int
	estimate           []uint32
}

func newNoiseReduction(smoothingBits int, even, odd, minRemaining float32, numChannels int) *noiseReduction {
	return &noiseReduction{
		smoothingBits: smoothingBits,
		// float32 product truncated to uint16, as the C assignment does.
		oddSmoothing:       uint16(float32(odd * (1 << noiseReductionBits))),
		evenSmoothing:      uint16(float32(even * (1 << noiseReductionBits))),
		minSignalRemaining: uint16(float32(minRemaining * (1 << noiseReductionBits))),
		numChannels:        numChannels,
		estimate:           make([]uint32, numChannels),
	}
}

// apply is NoiseReductionApply, in place on signal.
func (nr *noiseReduction) apply(signal []uint32) {
	for i := range nr.numChannels {
		smoothing := uint32(nr.evenSmoothing)
		if i&1 != 0 {
			smoothing = uint32(nr.oddSmoothing)
		}
		oneMinusSmoothing := uint32(1<<noiseReductionBits) - smoothing

		// Update the estimate of the noise.
		signalScaledUp := signal[i] << uint(nr.smoothingBits)
		estimate := uint32((uint64(signalScaledUp)*uint64(smoothing) +
			uint64(nr.estimate[i])*uint64(oneMinusSmoothing)) >> noiseReductionBits)
		nr.estimate[i] = estimate

		// Make sure that we can't get a negative value for the signal - estimate.
		if estimate > signalScaledUp {
			estimate = signalScaledUp
		}

		floor := uint32((uint64(signal[i]) * uint64(nr.minSignalRemaining)) >> noiseReductionBits)
		subtracted := (signalScaledUp - estimate) >> uint(nr.smoothingBits)
		if subtracted > floor {
			signal[i] = subtracted
		} else {
			signal[i] = floor
		}
	}
}

func (nr *noiseReduction) reset() {
	clear(nr.estimate)
}
