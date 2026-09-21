package wakeword

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tgallice/wakeword-go/internal/wav"
)

// audioFixture describes one file of testdata/audio/manifest.json: a small, committed
// subset of the synthetic TTS corpus produced by tools/corpus/generate.py.
type audioFixture struct {
	File          string  `json:"file"`
	Subset        string  `json:"subset"`
	Word          string  `json:"word"`
	Voice         string  `json:"voice"`
	Text          string  `json:"text"`
	ExpectedStart float64 `json:"expected_start_s"`
	ExpectedEnd   float64 `json:"expected_end_s"`
}

// prerollSamples is fed before every fixture so the ESPHome warm-up (100 invocations,
// 3 s) is over when the clip starts; the evaluator in cmd/wakeword-eval does the same.
const prerollSamples = 3500 * wav.SampleRate / 1000

// TestSyntheticAudioFixtures is functional, not parity: with the manifest settings, every
// positive fixture must fire once within 1.5 s after the wake word and every negative or
// near-miss fixture must stay silent, for its own model. Probabilities are not asserted.
func TestSyntheticAudioFixtures(t *testing.T) {
	raw, err := os.ReadFile("../testdata/audio/manifest.json")
	if err != nil {
		t.Skipf("no audio fixtures: %v", err)
	}
	var fixtures []audioFixture
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	preroll := make([]int16, prerollSamples)
	for i := range preroll {
		preroll[i] = int16(i%5 - 2) // a few LSB of dither, as an idle microphone
	}
	detectors := map[string]*Detector{}
	for _, name := range []string{"alexa", "hey_jarvis", "hey_mycroft", "okay_nabu"} {
		model, config := loadModel(t, name)
		d, err := NewDetector(model, config, Options{})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		detectors[name] = d
	}
	for _, fx := range fixtures {
		t.Run(fx.File, func(t *testing.T) {
			pcm, err := wav.ReadFile(filepath.Join("../testdata/audio", fx.File))
			if err != nil {
				t.Fatal(err)
			}
			words := []string{fx.Word}
			if fx.Word == "" { // a negative is checked against every model
				words = []string{"alexa", "hey_jarvis", "hey_mycroft", "okay_nabu"}
			}
			for _, word := range words {
				d := detectors[word]
				if d == nil {
					t.Fatalf("no model for word %q", word)
				}
				times := detectionTimes(t, d, preroll, pcm)
				switch fx.Subset {
				case "positive", "embedded":
					if len(times) != 1 {
						t.Fatalf("%q by %s: want exactly one detection, got %v", fx.Text, fx.Voice, times)
					}
					if times[0] < fx.ExpectedStart || times[0] > fx.ExpectedEnd+1.5 {
						t.Fatalf("%q by %s: detection at %.2f s, wake word spans %.2f to %.2f s",
							fx.Text, fx.Voice, times[0], fx.ExpectedStart, fx.ExpectedEnd)
					}
				case "near_miss", "negative":
					if len(times) != 0 {
						t.Fatalf("%q by %s: model %s fired at %v", fx.Text, fx.Voice, word, times)
					}
				default:
					t.Fatalf("unknown subset %q", fx.Subset)
				}
			}
		})
	}
}

// detectionTimes feeds the preroll then the clip in 100 ms chunks and returns the times of
// the detections not blocked by a VAD, in seconds relative to the clip.
func detectionTimes(t *testing.T, d *Detector, preroll, pcm []int16) []float64 {
	t.Helper()
	if err := d.Reset(); err != nil {
		t.Fatal(err)
	}
	var times []float64
	for _, p := range [][]int16{preroll, pcm} {
		for off := 0; off < len(p); off += 1600 {
			evs, err := d.Feed(p[off:min(off+1600, len(p))])
			if err != nil {
				t.Fatal(err)
			}
			for _, ev := range evs {
				if !ev.BlockedByVAD {
					times = append(times, float64(ev.Sample-prerollSamples)/wav.SampleRate)
				}
			}
		}
	}
	return times
}
