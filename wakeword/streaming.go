package wakeword

import "fmt"

// minSlicesBeforeDetection is ESPHome's MIN_SLICES_BEFORE_DETECTION (streaming_model.h): the
// number of sub-threshold frames a model must see, after start or after a detection, before
// it may report again.
const minSlicesBeforeDetection = 100

// streamingModel is a port of ESPHome's StreamingModel (components/micro_wake_word/
// streaming_model.cpp): a stride buffer of frames, an Invoke when the buffer is full, a ring
// of the most recent probabilities and the ignore-windows counter that implements the warm-up
// and the refractory period after a detection.
type streamingModel struct {
	scorer      Scorer
	input       []int8
	stride      int
	featureSize int
	cutoff      uint8
	window      int

	strideStep    int
	ring          []uint8
	lastN         int
	ignoreWindows int
	unprocessed   bool
}

func newStreamingModel(s Scorer, cutoff uint8, window int) (*streamingModel, error) {
	if window < 1 {
		return nil, fmt.Errorf("wakeword: sliding window size %d, want at least 1", window)
	}
	if s.Stride() < 1 || s.FeatureSize() < 1 || len(s.Input()) != s.Stride()*s.FeatureSize() {
		return nil, fmt.Errorf("wakeword: scorer input %d values for stride %d x %d features",
			len(s.Input()), s.Stride(), s.FeatureSize())
	}
	return &streamingModel{
		scorer:        s,
		input:         s.Input(),
		stride:        s.Stride(),
		featureSize:   s.FeatureSize(),
		cutoff:        cutoff,
		window:        window,
		ring:          make([]uint8, window),
		ignoreWindows: -minSlicesBeforeDetection,
	}, nil
}

// feed is StreamingModel::perform_streaming_inference for one frame: copy the frame into the
// stride buffer, invoke when the buffer is full, push the probability into the ring, and let
// the ignore-windows counter climb toward zero while the latest probability is below the
// cutoff.
func (m *streamingModel) feed(frame []int8) error {
	if len(frame) != m.featureSize {
		return fmt.Errorf("wakeword: frame has %d features, model wants %d", len(frame), m.featureSize)
	}
	m.strideStep %= m.stride
	copy(m.input[m.strideStep*m.featureSize:], frame)
	m.strideStep++

	if m.strideStep >= m.stride {
		p, err := m.scorer.Invoke()
		if err != nil {
			return err
		}
		m.lastN++
		if m.lastN == m.window {
			m.lastN = 0
		}
		m.ring[m.lastN] = p
		m.unprocessed = true
	}
	if m.ring[m.lastN] < m.cutoff {
		// Only climb while below the cutoff: a sustained high probability keeps the model
		// in its cool-off period and avoids duplicate detections.
		m.ignoreWindows = min(m.ignoreWindows+1, 0)
	}
	return nil
}

// stats returns the sum, average and maximum of the probability ring.
func (m *streamingModel) stats() (sum uint32, avg, maxP uint8) {
	for _, p := range m.ring {
		sum += uint32(p)
		maxP = max(maxP, p)
	}
	return sum, uint8(sum / uint32(m.window)), maxP
}

// determineWakeWord is WakeWordModel::determine_detected: nothing while the ignore-windows
// counter is negative, otherwise the ring sum against cutoff times window. It also clears
// the unprocessed flag, so a caller only evaluates a new probability once.
func (m *streamingModel) determineWakeWord() (detected bool, avg, maxP uint8) {
	if m.ignoreWindows < 0 {
		return false, 0, 0
	}
	sum, avg, maxP := m.stats()
	m.unprocessed = false
	return sum > uint32(m.cutoff)*uint32(m.window), avg, maxP
}

// determineVAD is VADModel::determine_detected: the same threshold without the
// ignore-windows gate.
func (m *streamingModel) determineVAD() (detected bool, avg, maxP uint8) {
	sum, avg, maxP := m.stats()
	return sum > uint32(m.cutoff)*uint32(m.window), avg, maxP
}

// resetProbabilities is StreamingModel::reset_probabilities: ring zeroed and the cool-off
// period restarted. It does not touch the model's streaming state.
func (m *streamingModel) resetProbabilities() {
	for i := range m.ring {
		m.ring[i] = 0
	}
	m.ignoreWindows = -minSlicesBeforeDetection
}

// reset restarts the model as freshly loaded: streaming state, stride buffer and ring.
func (m *streamingModel) reset() error {
	if err := m.scorer.Reset(); err != nil {
		return err
	}
	m.strideStep = 0
	m.lastN = 0
	m.unprocessed = false
	m.resetProbabilities()
	return nil
}
