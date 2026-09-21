package frontend

// Frontend turns PCM samples into feature frames. It is the port of
// FrontendProcessSamples in frontend.cc and holds the whole pipeline state.
// A Frontend is not safe for concurrent use.
type Frontend struct {
	cfg Config

	win  *window
	fft  *fft
	fb   *filterbank
	nr   *noiseReduction
	pcan *pcanGainControl // nil when disabled
	log  *logScaler

	energy         []int32
	scaled         []uint32
	features       []uint16
	correctionBits int
}

// New returns a Frontend with the default (microWakeWord) configuration.
func New() *Frontend {
	f, err := NewWithConfig(DefaultConfig())
	if err != nil {
		// The default configuration is validated by the tests; a failure here
		// is a programming error, not a runtime condition.
		panic(err)
	}
	return f
}

// NewWithConfig returns a Frontend for cfg. All tables and buffers are
// allocated here; Process and Reset do not allocate.
func NewWithConfig(cfg Config) (*Frontend, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	f := &Frontend{cfg: cfg}

	f.win = newWindow(cfg.WindowSizeMS, cfg.WindowStepMS, cfg.SampleRate)

	fft, err := newFFT(f.win.size)
	if err != nil {
		return nil, err
	}
	f.fft = fft

	spectrumSize := fft.fftSize/2 + 1
	fb, err := newFilterbank(cfg.NumChannels, cfg.LowerBandLimit, cfg.UpperBandLimit, cfg.SampleRate, spectrumSize)
	if err != nil {
		return nil, err
	}
	f.fb = fb

	f.nr = newNoiseReduction(cfg.NoiseSmoothingBits, cfg.NoiseEvenSmoothing, cfg.NoiseOddSmoothing,
		cfg.NoiseMinSignalRemaining, cfg.NumChannels)

	// The FFT scales its result down by fft_size, the filterbank weights are
	// Q12: the log stage compensates for both.
	f.correctionBits = mostSignificantBit32(uint32(fft.fftSize)) - 1 - (filterbankBits / 2)

	if cfg.PCANEnabled {
		f.pcan = newPcanGainControl(cfg.PCANStrength, cfg.PCANOffset, cfg.PCANGainBits,
			f.nr.estimate, cfg.NumChannels, cfg.NoiseSmoothingBits, f.correctionBits)
	}

	f.log = &logScaler{enableLog: cfg.LogEnabled, scaleShift: cfg.LogScaleShift}

	f.energy = make([]int32, spectrumSize)
	f.scaled = make([]uint32, cfg.NumChannels)
	f.features = make([]uint16, cfg.NumChannels)
	f.Reset()
	return f, nil
}

// Config returns the configuration the Frontend was built with.
func (f *Frontend) Config() Config { return f.cfg }

// NumChannels returns the number of features per frame.
func (f *Frontend) NumChannels() int { return f.cfg.NumChannels }

// WindowSize returns the analysis window length in samples.
func (f *Frontend) WindowSize() int { return f.win.size }

// WindowStep returns the hop between frames in samples.
func (f *Frontend) WindowStep() int { return f.win.step }

// Process feeds samples to the frontend and reports how many were consumed:
// at most what is needed to complete the current window, so a caller should
// loop over its buffer. When a window completes, the returned slice holds one
// frame of NumChannels features; it aliases an internal buffer that is
// overwritten by the next call. Otherwise features is nil.
func (f *Frontend) Process(samples []int16) (features []uint16, consumed int) {
	ok, consumed := f.win.process(samples)
	if !ok {
		return nil, consumed
	}

	// Scale the input up so the fixed point FFT keeps as much resolution as
	// possible; the shift is undone after the filterbank square root.
	inputShift := 15 - mostSignificantBit32(uint32(int32(f.win.maxAbsOutputValue)))
	f.fft.compute(f.win.output, inputShift)

	f.fb.convertToEnergy(f.fft.output, f.energy)
	f.fb.accumulate(f.energy)
	f.fb.sqrtScaled(inputShift, f.scaled)

	f.nr.apply(f.scaled)
	if f.pcan != nil {
		f.pcan.apply(f.scaled)
	}
	f.log.apply(f.scaled, f.features, f.correctionBits)
	return f.features, consumed
}

// Reset clears all state (window, FFT, filterbank, noise estimate), as
// FrontendReset does.
func (f *Frontend) Reset() {
	f.win.reset()
	f.fft.reset()
	f.fb.reset()
	f.nr.reset()
}
