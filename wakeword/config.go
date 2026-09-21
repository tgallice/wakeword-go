package wakeword

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/tgallice/wakeword-go/frontend"
)

// ModelConfig is the manifest that ships next to a microWakeWord model (the .json file of
// esphome/micro-wake-word-models). Only version 2 "micro" manifests are accepted.
type ModelConfig struct {
	Type             string      `json:"type"`
	WakeWord         string      `json:"wake_word"`
	Author           string      `json:"author"`
	Website          string      `json:"website"`
	Model            string      `json:"model"`
	TrainedLanguages []string    `json:"trained_languages"`
	Version          int         `json:"version"`
	Micro            MicroConfig `json:"micro"`
}

// MicroConfig holds the runtime parameters of a version 2 manifest.
type MicroConfig struct {
	// ProbabilityCutoff is the detection threshold as a float in (0, 1].
	ProbabilityCutoff float64 `json:"probability_cutoff"`
	// SlidingWindowSize is the number of recent probabilities averaged for a detection.
	SlidingWindowSize int `json:"sliding_window_size"`
	// FeatureStepSize is the frame step in milliseconds; it must match the frontend step.
	FeatureStepSize int `json:"feature_step_size"`
	// TensorArenaSize is the tflite-micro arena size; informational here.
	TensorArenaSize int `json:"tensor_arena_size"`
	// MinimumESPHomeVersion is informational.
	MinimumESPHomeVersion string `json:"minimum_esphome_version"`
}

// ParseModelConfig decodes and validates a model manifest.
func ParseModelConfig(data []byte) (ModelConfig, error) {
	var c ModelConfig
	if err := json.Unmarshal(data, &c); err != nil {
		return ModelConfig{}, fmt.Errorf("model config: %w", err)
	}
	if err := c.Validate(); err != nil {
		return ModelConfig{}, err
	}
	return c, nil
}

// Validate checks the fields the detector relies on.
func (c ModelConfig) Validate() error {
	var errs []error
	if c.Type != "micro" {
		errs = append(errs, fmt.Errorf("model config: type %q, want \"micro\"", c.Type))
	}
	if c.Version != 2 {
		errs = append(errs, fmt.Errorf("model config: version %d, want 2", c.Version))
	}
	if c.WakeWord == "" {
		errs = append(errs, errors.New("model config: missing wake_word"))
	}
	if !(c.Micro.ProbabilityCutoff > 0 && c.Micro.ProbabilityCutoff <= 1) {
		errs = append(errs, fmt.Errorf("model config: probability_cutoff %v outside (0, 1]", c.Micro.ProbabilityCutoff))
	}
	if c.Micro.SlidingWindowSize < 1 {
		errs = append(errs, fmt.Errorf("model config: sliding_window_size %d, want at least 1", c.Micro.SlidingWindowSize))
	}
	if step := frontend.DefaultConfig().WindowStepMS; c.Micro.FeatureStepSize != step {
		errs = append(errs, fmt.Errorf("model config: feature_step_size %d ms, the frontend produces a frame every %d ms",
			c.Micro.FeatureStepSize, step))
	}
	return errors.Join(errs...)
}

// CutoffUint8 converts a probability in (0, 1] to the uint8 threshold compared against the
// model output, exactly as the ESPHome component does when generating its code:
// `quantized_probability_cutoff = int(probability_cutoff * 255)` in
// components/micro_wake_word/__init__.py, that is a truncation toward zero, not a rounding.
// 0.97 gives 247 and 0.5 gives 127.
func CutoffUint8(probability float64) uint8 {
	v := int(probability * 255)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// CutoffUint8 returns the manifest threshold as a uint8.
func (c ModelConfig) CutoffUint8() uint8 {
	return CutoffUint8(c.Micro.ProbabilityCutoff)
}
