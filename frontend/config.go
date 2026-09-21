// Package frontend is a pure Go port of the tflite-micro audio microfrontend
// (tensorflow/lite/experimental/microfrontend/lib). It turns 16 kHz mono PCM
// into frames of log mel filterbank features, bit for bit identical to the C
// reference used to train the microWakeWord models and embedded in ESPHome.
//
// The pipeline per frame is: Hann window (Q12), fixed point real FFT (kissfft
// with 16 bit samples), squared magnitude, mel filterbank accumulation, integer
// square root, noise reduction, per channel automatic gain control (PCAN) and a
// fixed point natural logarithm. Every stage works on the same integer widths as
// the C code; float math only happens once, at construction, to build tables.
package frontend

import "fmt"

// Config holds the microfrontend parameters. The defaults are the values shared
// by microWakeWord (training), pymicro-features (oracle) and ESPHome (runtime);
// a model only works with the configuration it was trained with.
type Config struct {
	// SampleRate is the PCM sample rate in Hz.
	SampleRate int
	// WindowSizeMS is the analysis window length in milliseconds.
	WindowSizeMS int
	// WindowStepMS is the hop between two frames in milliseconds.
	WindowStepMS int

	// NumChannels is the number of mel filterbank channels (features per frame).
	NumChannels int
	// LowerBandLimit is the lowest filterbank frequency in Hz.
	LowerBandLimit float32
	// UpperBandLimit is the highest filterbank frequency in Hz.
	UpperBandLimit float32

	// NoiseSmoothingBits scales the signal up before noise estimation.
	NoiseSmoothingBits int
	// NoiseEvenSmoothing is the noise estimate smoothing factor for even channels.
	NoiseEvenSmoothing float32
	// NoiseOddSmoothing is the noise estimate smoothing factor for odd channels.
	NoiseOddSmoothing float32
	// NoiseMinSignalRemaining is the floor kept after noise subtraction.
	NoiseMinSignalRemaining float32

	// PCANEnabled turns the per channel gain normalization on.
	PCANEnabled bool
	// PCANStrength is the gain normalization exponent (0 disables, 1 full strength).
	PCANStrength float32
	// PCANOffset is added to the noise estimate in the normalization denominator.
	PCANOffset float32
	// PCANGainBits is the number of fractional bits of the gain.
	PCANGainBits int

	// LogEnabled applies the fixed point logarithm to the output.
	LogEnabled bool
	// LogScaleShift is the output scale shift of the logarithm.
	LogScaleShift int
}

// DefaultConfig returns the microWakeWord configuration (docs/spec.md, section 8).
func DefaultConfig() Config {
	return Config{
		SampleRate:              16000,
		WindowSizeMS:            30,
		WindowStepMS:            10,
		NumChannels:             40,
		LowerBandLimit:          125.0,
		UpperBandLimit:          7500.0,
		NoiseSmoothingBits:      10,
		NoiseEvenSmoothing:      0.025,
		NoiseOddSmoothing:       0.06,
		NoiseMinSignalRemaining: 0.05,
		PCANEnabled:             true,
		PCANStrength:            0.95,
		PCANOffset:              80.0,
		PCANGainBits:            21,
		LogEnabled:              true,
		LogScaleShift:           6,
	}
}

func (c Config) validate() error {
	switch {
	case c.SampleRate <= 0:
		return fmt.Errorf("frontend: sample rate must be positive, got %d", c.SampleRate)
	case c.WindowSizeMS <= 0 || c.WindowStepMS <= 0:
		return fmt.Errorf("frontend: window size and step must be positive, got %d ms and %d ms",
			c.WindowSizeMS, c.WindowStepMS)
	case c.WindowStepMS > c.WindowSizeMS:
		return fmt.Errorf("frontend: window step (%d ms) larger than window size (%d ms)",
			c.WindowStepMS, c.WindowSizeMS)
	case c.NumChannels <= 0:
		return fmt.Errorf("frontend: number of channels must be positive, got %d", c.NumChannels)
	case c.LowerBandLimit < 0 || c.UpperBandLimit <= c.LowerBandLimit:
		return fmt.Errorf("frontend: invalid band limits %g Hz to %g Hz", c.LowerBandLimit, c.UpperBandLimit)
	case c.NoiseSmoothingBits < 0 || c.NoiseSmoothingBits > 31:
		return fmt.Errorf("frontend: noise smoothing bits out of range: %d", c.NoiseSmoothingBits)
	case c.PCANGainBits < 0 || c.PCANGainBits > 31:
		return fmt.Errorf("frontend: PCAN gain bits out of range: %d", c.PCANGainBits)
	case c.LogScaleShift < 0 || c.LogScaleShift > 31:
		return fmt.Errorf("frontend: log scale shift out of range: %d", c.LogScaleShift)
	}
	return nil
}
