// Package wakeword detects a wake word in 16 kHz mono PCM audio with a microWakeWord v2
// model, reproducing the decision logic of the ESPHome micro_wake_word component.
//
// A Detector chains the frontend (frontend package), the ESPHome feature quantization and the
// model (runtime package), then applies the streaming decision rules of docs/spec.md
// section 9: a sliding window of probabilities, a warm-up period, and a refractory period
// after each detection. An optional VAD model gates the detections.
package wakeword

import (
	"errors"
	"fmt"

	"github.com/tgallice/wakeword-go/frontend"
)

// Event reports a wake word decision.
type Event struct {
	// WakeWord is the manifest wake_word of the model that fired.
	WakeWord string
	// AverageProbability and MaxProbability are over the sliding window, in 0 to 255.
	AverageProbability uint8
	MaxProbability     uint8
	// Frame is the index of the feature frame (one every WindowStep samples) that produced
	// the event, counted from the last Reset.
	Frame int
	// Sample is the position in samples, from the last Reset, of the end of the audio window
	// that produced the frame: WindowSize + Frame * WindowStep.
	Sample int
	// BlockedByVAD is set when the wake word model fired but the VAD model reported no
	// speech: ESPHome logs it and does not trigger. The wake word model is not reset in
	// that case, so it may fire again on the next invocation.
	BlockedByVAD bool
}

// Options tune a Detector beyond the manifest values.
type Options struct {
	// ProbabilityCutoff overrides the manifest probability_cutoff when non-zero.
	ProbabilityCutoff float64
	// SlidingWindowSize overrides the manifest sliding_window_size when non-zero.
	SlidingWindowSize int
	// VADModel and VADConfig configure an optional VAD model (a version 2 manifest whose
	// wake_word is "vad"). Both must be set together.
	VADModel  []byte
	VADConfig []byte
	// VADProbabilityCutoff and VADSlidingWindowSize override the VAD manifest values when non-zero.
	VADProbabilityCutoff float64
	VADSlidingWindowSize int
	// OnProbability, when set, is called after every wake word model invocation with the
	// frame index and the raw output (0 to 255). Debugging and threshold tuning only.
	OnProbability func(frame int, probability uint8)
}

// Detector runs one wake word model, and optionally a VAD model, over audio or features.
// It is not safe for concurrent use.
type Detector struct {
	config    ModelConfig
	vadConfig ModelConfig
	model     *streamingModel
	vad       *streamingModel

	fe         *frontend.Frontend
	frame      []int8
	windowSize int
	windowStep int
	frames     int
	events     []Event
	onProb     func(frame int, probability uint8)
}

// NewDetector builds a detector from a model file, its manifest and options.
func NewDetector(model, configJSON []byte, opts Options) (*Detector, error) {
	cfg, err := ParseModelConfig(configJSON)
	if err != nil {
		return nil, err
	}
	if (opts.VADModel == nil) != (opts.VADConfig == nil) {
		return nil, errors.New("wakeword: VADModel and VADConfig must be set together")
	}
	scorer, err := newModelScorer(model)
	if err != nil {
		return nil, fmt.Errorf("wakeword: model %q: %w", cfg.WakeWord, err)
	}
	if err := scorer.probe(); err != nil {
		return nil, fmt.Errorf("wakeword: model %q: %w", cfg.WakeWord, err)
	}
	var vadScorer Scorer
	vadCfg := ModelConfig{}
	if opts.VADModel != nil {
		vadCfg, err = ParseModelConfig(opts.VADConfig)
		if err != nil {
			return nil, fmt.Errorf("wakeword: vad: %w", err)
		}
		vs, err := newModelScorer(opts.VADModel)
		if err != nil {
			return nil, fmt.Errorf("wakeword: vad model: %w", err)
		}
		if err := vs.probe(); err != nil {
			return nil, fmt.Errorf("wakeword: vad model: %w", err)
		}
		vadScorer = vs
	}
	return newDetector(cfg, scorer, vadCfg, vadScorer, opts)
}

// newDetector assembles a detector from scorers; tests use it with scripted scorers.
func newDetector(cfg ModelConfig, scorer Scorer, vadCfg ModelConfig, vadScorer Scorer, opts Options) (*Detector, error) {
	cutoff, window := cfg.Micro.ProbabilityCutoff, cfg.Micro.SlidingWindowSize
	if opts.ProbabilityCutoff != 0 {
		cutoff = opts.ProbabilityCutoff
	}
	if opts.SlidingWindowSize != 0 {
		window = opts.SlidingWindowSize
	}
	if !(cutoff > 0 && cutoff <= 1) {
		return nil, fmt.Errorf("wakeword: probability cutoff %v outside (0, 1]", cutoff)
	}
	model, err := newStreamingModel(scorer, CutoffUint8(cutoff), window)
	if err != nil {
		return nil, err
	}
	fe := frontend.New()
	if scorer.FeatureSize() != fe.NumChannels() {
		return nil, fmt.Errorf("wakeword: model wants %d features per frame, the frontend produces %d",
			scorer.FeatureSize(), fe.NumChannels())
	}
	d := &Detector{
		config:     cfg,
		model:      model,
		fe:         fe,
		frame:      make([]int8, fe.NumChannels()),
		windowSize: fe.WindowSize(),
		windowStep: fe.WindowStep(),
		events:     make([]Event, 0, 2),
		onProb:     opts.OnProbability,
	}
	if vadScorer != nil {
		vadCutoff, vadWindow := vadCfg.Micro.ProbabilityCutoff, vadCfg.Micro.SlidingWindowSize
		if opts.VADProbabilityCutoff != 0 {
			vadCutoff = opts.VADProbabilityCutoff
		}
		if opts.VADSlidingWindowSize != 0 {
			vadWindow = opts.VADSlidingWindowSize
		}
		if !(vadCutoff > 0 && vadCutoff <= 1) {
			return nil, fmt.Errorf("wakeword: vad probability cutoff %v outside (0, 1]", vadCutoff)
		}
		if vadScorer.FeatureSize() != scorer.FeatureSize() {
			return nil, fmt.Errorf("wakeword: vad model wants %d features per frame, the wake word model %d",
				vadScorer.FeatureSize(), scorer.FeatureSize())
		}
		d.vad, err = newStreamingModel(vadScorer, CutoffUint8(vadCutoff), vadWindow)
		if err != nil {
			return nil, err
		}
		d.vadConfig = vadCfg
	}
	return d, nil
}

// Config returns the wake word model manifest.
func (d *Detector) Config() ModelConfig { return d.config }

// HasVAD reports whether a VAD model gates the detections.
func (d *Detector) HasVAD() bool { return d.vad != nil }

// Cutoff returns the uint8 threshold in use for the wake word model.
func (d *Detector) Cutoff() uint8 { return d.model.cutoff }

// SlidingWindowSize returns the window size in use for the wake word model.
func (d *Detector) SlidingWindowSize() int { return d.model.window }

// Stride returns how many frames the model scores per invocation.
func (d *Detector) Stride() int { return d.model.stride }

// FeatureSize returns the number of features per frame.
func (d *Detector) FeatureSize() int { return d.model.featureSize }

// Frames returns the number of frames fed since the last Reset.
func (d *Detector) Frames() int { return d.frames }

// LastProbability returns the most recent wake word model output (0 to 255) and whether a
// new invocation happened during the last FeedFeatures call. It exists for debugging and
// threshold tuning; the detection decision uses the sliding window, not this value alone.
func (d *Detector) LastProbability() (p uint8, fresh bool) {
	return d.model.ring[d.model.lastN], d.model.lastInvoked
}

// FeedFeatures feeds one frame of quantized int8 features (as produced by
// frontend.QuantizeFeatures) to the models and reports a detection, if any. This is the
// per-frame path of ESPHome's inference task: every model scores the frame, then the
// wake word decision is taken and gated by the VAD.
func (d *Detector) FeedFeatures(frame []int8) (Event, bool, error) {
	if err := d.model.feed(frame); err != nil {
		return Event{}, false, err
	}
	if d.vad != nil {
		if err := d.vad.feed(frame); err != nil {
			return Event{}, false, err
		}
	}
	if d.onProb != nil && d.model.lastInvoked {
		d.onProb(d.frames, d.model.ring[d.model.lastN])
	}
	ev, ok := d.process()
	d.frames++
	return ev, ok, nil
}

// process is MicroWakeWord::process_probabilities_ for the single wake word model.
func (d *Detector) process() (Event, bool) {
	speech := true
	if d.vad != nil {
		speech, _, _ = d.vad.determineVAD()
	}
	if !d.model.unprocessed {
		return Event{}, false
	}
	detected, avg, maxP := d.model.determineWakeWord()
	if !detected {
		return Event{}, false
	}
	ev := Event{
		WakeWord:           d.config.WakeWord,
		AverageProbability: avg,
		MaxProbability:     maxP,
		Frame:              d.frames,
		Sample:             d.windowSize + d.frames*d.windowStep,
	}
	if speech {
		d.model.resetProbabilities()
	} else {
		ev.BlockedByVAD = true
	}
	return ev, true
}

// Feed runs PCM samples (16 kHz, mono, int16) through the frontend and the models. The
// returned slice aliases an internal buffer that the next call overwrites; copy the events
// to keep them. Samples are consumed entirely: the frontend buffers a partial window.
func (d *Detector) Feed(pcm []int16) ([]Event, error) {
	d.events = d.events[:0]
	for len(pcm) > 0 {
		features, n := d.fe.Process(pcm)
		pcm = pcm[n:]
		if features == nil {
			continue
		}
		frontend.QuantizeFeatures(d.frame, features)
		ev, ok, err := d.FeedFeatures(d.frame)
		if err != nil {
			return d.events, err
		}
		if ok {
			d.events = append(d.events, ev)
		}
	}
	return d.events, nil
}

// Reset restarts the frontend, the models and the frame counter.
func (d *Detector) Reset() error {
	d.fe.Reset()
	d.frames = 0
	if err := d.model.reset(); err != nil {
		return err
	}
	if d.vad != nil {
		if err := d.vad.reset(); err != nil {
			return err
		}
	}
	return nil
}
