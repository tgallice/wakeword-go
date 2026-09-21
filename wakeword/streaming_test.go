package wakeword

import (
	"errors"
	"strings"
	"testing"
)

// scriptScorer replays a list of probabilities, one per invocation, repeating the last one.
type scriptScorer struct {
	stride   int
	features int
	input    []int8
	probs    []uint8
	calls    int
	resets   int
	err      error
}

func newScriptScorer(stride int, probs ...uint8) *scriptScorer {
	return &scriptScorer{stride: stride, features: 40, input: make([]int8, stride*40), probs: probs}
}

func (s *scriptScorer) Stride() int      { return s.stride }
func (s *scriptScorer) FeatureSize() int { return s.features }
func (s *scriptScorer) Input() []int8    { return s.input }
func (s *scriptScorer) Reset() error     { s.resets++; s.calls = 0; return nil }

func (s *scriptScorer) Invoke() (uint8, error) {
	if s.err != nil {
		return 0, s.err
	}
	i := min(s.calls, len(s.probs)-1)
	s.calls++
	return s.probs[i], nil
}

// repeat builds a probability script from (count, value) runs; the last value repeats forever.
func repeat(runs ...int) []uint8 {
	var out []uint8
	for i := 0; i+1 < len(runs); i += 2 {
		for range runs[i] {
			out = append(out, uint8(runs[i+1]))
		}
	}
	return out
}

func testConfig(word string, cutoff float64, window int) ModelConfig {
	return ModelConfig{Type: "micro", WakeWord: word, Version: 2, Micro: MicroConfig{
		ProbabilityCutoff: cutoff, SlidingWindowSize: window, FeatureStepSize: 10,
	}}
}

// run feeds n frames of a silent feature vector and returns the frames at which events
// fired, with the events themselves.
func run(t *testing.T, d *Detector, n int) ([]int, []Event) {
	t.Helper()
	frame := make([]int8, d.FeatureSize())
	for i := range frame {
		frame[i] = -128
	}
	var at []int
	var evs []Event
	for i := range n {
		ev, ok, err := d.FeedFeatures(frame)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if ok {
			if ev.Frame != i {
				t.Errorf("event at frame %d carries Frame %d", i, ev.Frame)
			}
			at = append(at, i)
			evs = append(evs, ev)
		}
	}
	return at, evs
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestDecisionLogic(t *testing.T) {
	cases := []struct {
		name   string
		cutoff float64
		window int
		stride int
		probs  []uint8
		frames int
		want   []int
	}{
		{
			// 100 sub-cutoff windows of warm-up, then five 255s fill the window: one event,
			// and the sustained 255 never lets the cool-off counter climb again.
			name: "warm-up then one detection, sustained high", cutoff: 0.97, window: 5, stride: 1,
			probs: repeat(100, 0, 1, 255), frames: 300, want: []int{104},
		},
		{
			// After the detection, 100 sub-cutoff windows re-arm the model and a second burst fires.
			name: "refractory then recovery", cutoff: 0.97, window: 5, stride: 1,
			probs: repeat(100, 0, 5, 255, 100, 0, 1, 255), frames: 300, want: []int{104, 209},
		},
		{
			// A burst within the refractory period (fewer than 100 low windows) does not fire.
			name: "burst inside refractory period", cutoff: 0.97, window: 5, stride: 1,
			probs: repeat(100, 0, 5, 255, 50, 0, 5, 255, 100, 0, 1, 255), frames: 400, want: []int{104, 264},
		},
		{
			// High probabilities before the warm-up completes never count, and they do not
			// advance the warm-up either.
			name: "high before warm-up is ignored", cutoff: 0.97, window: 5, stride: 1,
			probs: repeat(5, 255, 100, 0, 1, 255), frames: 200, want: []int{109},
		},
		{
			// Sum exactly equal to cutoff * window does not detect: five 127s give 635 = 127 * 5.
			// One 128 replacing a 127 gives 636 and fires.
			name: "sum equal to threshold does not fire", cutoff: 0.5, window: 5, stride: 1,
			probs: repeat(100, 0, 5, 127, 1, 128), frames: 120, want: []int{105},
		},
		{
			// Window of 1: a single probability above the cutoff fires right after warm-up.
			name: "window one", cutoff: 0.97, window: 1, stride: 1,
			probs: repeat(100, 0, 1, 248), frames: 120, want: []int{100},
		},
		{
			// Window of 1 with a probability equal to the cutoff: 247 > 247 is false.
			name: "window one at cutoff", cutoff: 0.97, window: 1, stride: 1,
			probs: repeat(100, 0, 1, 247), frames: 120, want: nil,
		},
		{
			// Stride 3: invocations on frames 2, 5, 8, ... while the cool-off counter climbs on
			// every frame. 34 zero invocations cover frames 0 to 101; the five 255s land on
			// frames 104, 107, 110, 113 and 116.
			name: "stride three", cutoff: 0.97, window: 5, stride: 3,
			probs: repeat(34, 0, 1, 255), frames: 300, want: []int{116},
		},
		{
			// Nothing ever crosses the cutoff.
			name: "never", cutoff: 0.97, window: 5, stride: 1,
			probs: repeat(1, 200), frames: 500, want: nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sc := newScriptScorer(tc.stride, tc.probs...)
			d, err := newDetector(testConfig("test", tc.cutoff, tc.window), sc, ModelConfig{}, nil, Options{})
			if err != nil {
				t.Fatalf("newDetector: %v", err)
			}
			got, evs := run(t, d, tc.frames)
			if !equalInts(got, tc.want) {
				t.Fatalf("events at frames %v, want %v", got, tc.want)
			}
			for _, ev := range evs {
				if ev.WakeWord != "test" || ev.BlockedByVAD {
					t.Errorf("event %+v: wrong word or blocked", ev)
				}
				if ev.Sample != d.windowSize+ev.Frame*d.windowStep {
					t.Errorf("event sample %d, want %d", ev.Sample, d.windowSize+ev.Frame*d.windowStep)
				}
			}
			if wantCalls := tc.frames / tc.stride; sc.calls != wantCalls {
				t.Errorf("%d invocations for %d frames of stride %d, want %d", sc.calls, tc.frames, tc.stride, wantCalls)
			}
		})
	}
}

func TestDetectionEventProbabilities(t *testing.T) {
	sc := newScriptScorer(1, repeat(100, 0, 1, 100, 1, 120, 1, 130, 1, 140, 1, 200)...)
	d, err := newDetector(testConfig("w", 0.5, 5), sc, ModelConfig{}, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	at, evs := run(t, d, 105)
	// Partial sums 100, 220, 350, 490 stay at or below 127 * 5 = 635; the fifth value gives
	// 690 and fires with average 138 (690 / 5) and max 200.
	if !equalInts(at, []int{104}) {
		t.Fatalf("events at %v, want [104]", at)
	}
	if evs[0].AverageProbability != 138 || evs[0].MaxProbability != 200 {
		t.Errorf("average %d max %d, want 138 and 200", evs[0].AverageProbability, evs[0].MaxProbability)
	}
}

func TestVADGate(t *testing.T) {
	cases := []struct {
		name        string
		vadProbs    []uint8
		frames      int
		wantAt      []int
		wantBlocked []bool
	}{
		{
			// VAD reports speech from the start (no warm-up for the VAD): the detection passes.
			name: "speech", vadProbs: repeat(1, 255), frames: 200,
			wantAt: []int{104}, wantBlocked: []bool{false},
		},
		{
			// No speech ever: the model fires and is blocked, and since a blocked model is not
			// reset it fires again on every new probability while its window stays high.
			name: "no speech", vadProbs: repeat(1, 0), frames: 108,
			wantAt: []int{104, 105, 106, 107}, wantBlocked: []bool{true, true, true, true},
		},
		{
			// Speech starts late: blocked until the VAD window (5, cutoff 127) sums above 635,
			// which takes three 255s (frames 106, 107, 108), then the real detection and reset.
			name: "late speech", vadProbs: repeat(106, 0, 1, 255), frames: 300,
			wantAt: []int{104, 105, 106, 107, 108}, wantBlocked: []bool{true, true, true, true, false},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sc := newScriptScorer(1, repeat(100, 0, 1, 255)...)
			vad := newScriptScorer(1, tc.vadProbs...)
			d, err := newDetector(testConfig("w", 0.97, 5), sc, testConfig("vad", 0.5, 5), vad, Options{})
			if err != nil {
				t.Fatal(err)
			}
			if !d.HasVAD() {
				t.Fatal("HasVAD false")
			}
			at, evs := run(t, d, tc.frames)
			if !equalInts(at, tc.wantAt) {
				t.Fatalf("events at %v, want %v", at, tc.wantAt)
			}
			for i, ev := range evs {
				if ev.BlockedByVAD != tc.wantBlocked[i] {
					t.Errorf("event %d blocked %v, want %v", i, ev.BlockedByVAD, tc.wantBlocked[i])
				}
			}
		})
	}
}

func TestOptionsOverrideManifest(t *testing.T) {
	sc := newScriptScorer(1, repeat(100, 0, 1, 130)...)
	// Manifest cutoff 0.97 would never fire on 130; the override to 0.5 (127) fires once the
	// window of 3 holds three 130s: 390 > 381.
	d, err := newDetector(testConfig("w", 0.97, 5), sc, ModelConfig{}, nil,
		Options{ProbabilityCutoff: 0.5, SlidingWindowSize: 3})
	if err != nil {
		t.Fatal(err)
	}
	if d.Cutoff() != 127 || d.SlidingWindowSize() != 3 {
		t.Fatalf("cutoff %d window %d, want 127 and 3", d.Cutoff(), d.SlidingWindowSize())
	}
	at, _ := run(t, d, 120)
	if !equalInts(at, []int{102}) {
		t.Errorf("events at %v, want [102]", at)
	}
	if _, err := newDetector(testConfig("w", 0.97, 5), sc, ModelConfig{}, nil, Options{ProbabilityCutoff: 1.5}); err == nil {
		t.Error("cutoff 1.5 accepted")
	}
}

func TestResetRestartsEverything(t *testing.T) {
	sc := newScriptScorer(3, repeat(34, 0, 1, 255)...)
	d, err := newDetector(testConfig("w", 0.97, 5), sc, ModelConfig{}, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if at, _ := run(t, d, 118); !equalInts(at, []int{116}) {
		t.Fatalf("events at %v before reset, want [116]", at)
	}
	if err := d.Reset(); err != nil {
		t.Fatal(err)
	}
	if sc.resets != 1 || d.Frames() != 0 || d.model.strideStep != 0 || d.model.lastN != 0 ||
		d.model.ignoreWindows != -minSlicesBeforeDetection || d.model.unprocessed {
		t.Errorf("state after reset: resets %d frames %d stride %d lastN %d ignore %d unprocessed %v",
			sc.resets, d.Frames(), d.model.strideStep, d.model.lastN, d.model.ignoreWindows, d.model.unprocessed)
	}
	for i, p := range d.model.ring {
		if p != 0 {
			t.Errorf("ring[%d] = %d after reset", i, p)
		}
	}
	// The same script replays identically.
	if at, _ := run(t, d, 118); !equalInts(at, []int{116}) {
		t.Errorf("events at %v after reset, want [116]", at)
	}
}

func TestFeedFeaturesErrors(t *testing.T) {
	sc := newScriptScorer(1, 0)
	d, err := newDetector(testConfig("w", 0.97, 5), sc, ModelConfig{}, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := d.FeedFeatures(make([]int8, 39)); err == nil || !strings.Contains(err.Error(), "39 features") {
		t.Errorf("short frame: %v", err)
	}
	boom := errors.New("boom")
	sc.err = boom
	if _, _, err := d.FeedFeatures(make([]int8, 40)); !errors.Is(err, boom) {
		t.Errorf("scorer error not propagated: %v", err)
	}
}

func TestFeedFeaturesDoesNotAllocate(t *testing.T) {
	sc := newScriptScorer(3, repeat(34, 0, 1, 255)...)
	vad := newScriptScorer(3, repeat(1, 255)...)
	d, err := newDetector(testConfig("w", 0.97, 5), sc, testConfig("vad", 0.5, 5), vad, Options{})
	if err != nil {
		t.Fatal(err)
	}
	frame := make([]int8, 40)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := d.FeedFeatures(frame); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Errorf("FeedFeatures allocates %v times per frame", allocs)
	}
}
