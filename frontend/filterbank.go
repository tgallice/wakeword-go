package frontend

import (
	"fmt"
	"math"
)

const (
	// filterbankBits is the fixed point precision of the weights (kFilterbankBits).
	filterbankBits = 12
	// filterbankIndexAlignment and filterbankChannelBlockSize control the
	// padding of the weight arrays (filterbank_util.cc).
	filterbankIndexAlignment   = 4
	filterbankChannelBlockSize = 4
)

// filterbank is the port of filterbank.cc and filterbank_util.cc: the mel
// triangular filters applied to the FFT energy, accumulated in 64 bits.
type filterbank struct {
	numChannels int
	startIndex  int
	endIndex    int

	channelFrequencyStarts []int16
	channelWeightStarts    []int16
	channelWidths          []int16
	weights                []int16
	unweights              []int16
	work                   []uint64
}

// freqToMel is FreqToMel in float32.
func freqToMel(freq float32) float32 {
	return float32(1127.0 * log1pf(float32(freq/700.0)))
}

func log1pf(x float32) float32 { return float32(math.Log1p(float64(x))) }

func newFilterbank(numChannels int, lower, upper float32, sampleRate, spectrumSize int) (*filterbank, error) {
	fb := &filterbank{numChannels: numChannels}
	numChannelsPlus1 := numChannels + 1

	// Index alignment in int16 units.
	indexAlignment := filterbankIndexAlignment / 2

	fb.channelFrequencyStarts = make([]int16, numChannelsPlus1)
	fb.channelWeightStarts = make([]int16, numChannelsPlus1)
	fb.channelWidths = make([]int16, numChannelsPlus1)
	fb.work = make([]uint64, numChannelsPlus1)

	centerMelFreqs := make([]float32, numChannelsPlus1)
	actualChannelStarts := make([]int16, numChannelsPlus1)
	actualChannelWidths := make([]int16, numChannelsPlus1)

	// CalculateCenterFrequencies.
	melLow := freqToMel(lower)
	melHi := freqToMel(upper)
	melSpan := float32(melHi - melLow)
	melSpacing := float32(melSpan / float32(numChannelsPlus1))
	for i := range numChannelsPlus1 {
		centerMelFreqs[i] = float32(melLow + float32(melSpacing*float32(i+1)))
	}

	// Always exclude DC.
	hzPerSbin := float32(float32(0.5*float32(sampleRate)) / (float32(spectrumSize) - 1))
	fb.startIndex = int(float32(1.5 + float32(lower/hzPerSbin)))
	fb.endIndex = 0

	chanFreqIndexStart := fb.startIndex
	weightIndexStart := 0
	needsZeros := false

	for ch := range numChannelsPlus1 {
		// Keep jumping frequencies until we overshoot the bound on this channel.
		freqIndex := chanFreqIndexStart
		for freqToMel(float32(float32(freqIndex)*hzPerSbin)) <= centerMelFreqs[ch] {
			freqIndex++
		}

		width := freqIndex - chanFreqIndexStart
		actualChannelStarts[ch] = int16(chanFreqIndexStart)
		actualChannelWidths[ch] = int16(width)

		if width == 0 {
			// This channel gets no frequency: point it at a block of zero
			// weights at the start of the arrays, shifting earlier channels once.
			fb.channelFrequencyStarts[ch] = 0
			fb.channelWeightStarts[ch] = 0
			fb.channelWidths[ch] = filterbankChannelBlockSize
			if !needsZeros {
				needsZeros = true
				for j := range ch {
					fb.channelWeightStarts[j] += filterbankChannelBlockSize
				}
				weightIndexStart += filterbankChannelBlockSize
			}
		} else {
			alignedStart := (chanFreqIndexStart / indexAlignment) * indexAlignment
			alignedWidth := chanFreqIndexStart - alignedStart + width
			paddedWidth := (((alignedWidth - 1) / filterbankChannelBlockSize) + 1) * filterbankChannelBlockSize

			fb.channelFrequencyStarts[ch] = int16(alignedStart)
			fb.channelWeightStarts[ch] = int16(weightIndexStart)
			fb.channelWidths[ch] = int16(paddedWidth)
			weightIndexStart += paddedWidth
		}
		chanFreqIndexStart = freqIndex
	}

	fb.weights = make([]int16, weightIndexStart)
	fb.unweights = make([]int16, weightIndexStart)

	// Second pass: the weights of the frequencies that belong to a channel.
	for ch := range numChannelsPlus1 {
		frequency := int(actualChannelStarts[ch])
		numFrequencies := int(actualChannelWidths[ch])
		frequencyOffset := frequency - int(fb.channelFrequencyStarts[ch])
		weightStart := int(fb.channelWeightStarts[ch])
		denomVal := melLow
		if ch != 0 {
			denomVal = centerMelFreqs[ch-1]
		}
		for j := 0; j < numFrequencies; j, frequency = j+1, frequency+1 {
			num := float32(centerMelFreqs[ch] - freqToMel(float32(float32(frequency)*hzPerSbin)))
			den := float32(centerMelFreqs[ch] - denomVal)
			weight := float32(num / den)
			weightIndex := weightStart + frequencyOffset + j
			fb.weights[weightIndex], fb.unweights[weightIndex] = quantizeFilterbankWeights(weight)
		}
		if frequency > fb.endIndex {
			fb.endIndex = frequency
		}
	}

	if fb.endIndex >= spectrumSize {
		return nil, fmt.Errorf("frontend: filterbank end index %d is above spectrum size %d", fb.endIndex, spectrumSize)
	}
	return fb, nil
}

// quantizeFilterbankWeights is QuantizeFilterbankWeights.
func quantizeFilterbankWeights(w float32) (weight, unweight int16) {
	weight = int16(floorf(float32(w*(1<<filterbankBits)) + 0.5))
	unweight = int16(floorf(float32(float32(1.0-w)*(1<<filterbankBits)) + 0.5))
	return weight, unweight
}

// convertToEnergy is FilterbankConvertFftComplexToEnergy: squared magnitude
// of the bins in [startIndex, endIndex), computed in int32 with wraparound and
// stored as int32 like the C code (which aliases the FFT output buffer).
func (fb *filterbank) convertToEnergy(fftOutput []cpx, energy []int32) {
	for i := fb.startIndex; i < fb.endIndex; i++ {
		re := int32(fftOutput[i].r)
		im := int32(fftOutput[i].i)
		energy[i] = re*re + im*im
	}
}

// accumulate is FilterbankAccumulateChannels. The int32 energy is converted to
// uint64 through sign extension, exactly as the C cast does.
func (fb *filterbank) accumulate(energy []int32) {
	var weightAccumulator, unweightAccumulator uint64
	for i := range fb.numChannels + 1 {
		magnitudes := energy[fb.channelFrequencyStarts[i]:]
		ws := int(fb.channelWeightStarts[i])
		weights := fb.weights[ws:]
		unweights := fb.unweights[ws:]
		width := int(fb.channelWidths[i])
		for j := range width {
			m := uint64(int64(magnitudes[j]))
			weightAccumulator += uint64(int64(weights[j])) * m
			unweightAccumulator += uint64(int64(unweights[j])) * m
		}
		fb.work[i] = weightAccumulator
		weightAccumulator = unweightAccumulator
		unweightAccumulator = 0
	}
}

// sqrtScaled is FilterbankSqrt: the integer square root of each channel
// accumulator, shifted down by scaleDownShift.
func (fb *filterbank) sqrtScaled(scaleDownShift int, output []uint32) {
	for i := range fb.numChannels {
		output[i] = sqrt64(fb.work[i+1]) >> uint(scaleDownShift)
	}
}

func (fb *filterbank) reset() {
	clear(fb.work)
}

// sqrt32 is Sqrt32 in filterbank.cc.
func sqrt32(num uint32) uint16 {
	if num == 0 {
		return 0
	}
	var res uint32
	maxBitNumber := 32 - mostSignificantBit32(num)
	maxBitNumber |= 1
	bit := uint32(1) << uint(31-maxBitNumber)
	iterations := (31-maxBitNumber)/2 + 1
	for ; iterations > 0; iterations-- {
		if num >= res+bit {
			num -= res + bit
			res = (res >> 1) + bit
		} else {
			res >>= 1
		}
		bit >>= 2
	}
	if num > res && res != 0xFFFF {
		res++
	}
	return uint16(res)
}

// sqrt64 is Sqrt64 in filterbank.cc, including its 32 bit shortcut.
func sqrt64(num uint64) uint32 {
	if num>>32 == 0 {
		return uint32(sqrt32(uint32(num)))
	}
	var res uint64
	maxBitNumber := 64 - mostSignificantBit64(num)
	maxBitNumber |= 1
	bit := uint64(1) << uint(63-maxBitNumber)
	iterations := (63-maxBitNumber)/2 + 1
	for ; iterations > 0; iterations-- {
		if num >= res+bit {
			num -= res + bit
			res = (res >> 1) + bit
		} else {
			res >>= 1
		}
		bit >>= 2
	}
	if num > res && res != 0xFFFFFFFF {
		res++
	}
	return uint32(res)
}
