package wakeword

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tgallice/wakeword-go/runtime"
)

// skipWithoutKernels skips a test whose model invocation needs kernels that are not
// implemented (all standard microWakeWord operators should be implemented).
func skipWithoutKernels(t *testing.T, err error) {
	t.Helper()
	if errors.Is(err, runtime.ErrNoKernel) {
		t.Skipf("kernel not implemented: %v", err)
	}
}

func loadModel(t *testing.T, name string) (model, config []byte) {
	t.Helper()
	return readFile(t, filepath.Join(modelsDir, name+".tflite")), readFile(t, filepath.Join(modelsDir, name+".json"))
}

func TestModelScorerShapes(t *testing.T) {
	for _, name := range []string{"alexa", "hey_jarvis", "hey_mycroft", "okay_nabu", "vad"} {
		t.Run(name, func(t *testing.T) {
			model, _ := loadModel(t, name)
			s, err := newModelScorer(model)
			if err != nil {
				t.Fatalf("newModelScorer: %v", err)
			}
			if s.Stride() != 3 || s.FeatureSize() != 40 || len(s.Input()) != 120 {
				t.Errorf("stride %d features %d input %d, want 3, 40, 120", s.Stride(), s.FeatureSize(), len(s.Input()))
			}
			err = s.probe()
			skipWithoutKernels(t, err)
			if err != nil {
				t.Fatalf("probe: %v", err)
			}
			p, err := s.Invoke()
			if err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			t.Logf("silent input scores %d", p)
		})
	}
}

func TestNewDetectorRejects(t *testing.T) {
	model, config := loadModel(t, "hey_jarvis")
	v1 := readFile(t, "../testdata/models/v1/okay_nabu.tflite")
	cases := []struct {
		name  string
		model []byte
		cfg   []byte
		opts  Options
		want  string
	}{
		{"bad config", model, []byte(`{}`), Options{}, "model config"},
		{"v1 model", v1, config, Options{}, "unsupported operators"},
		{"not a model", []byte("hello"), config, Options{}, "not a TFLite model"},
		{"vad model without config", model, config, Options{VADModel: model}, "VADModel and VADConfig"},
		{"vad config without model", model, config, Options{VADConfig: config}, "VADModel and VADConfig"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewDetector(tc.model, tc.cfg, tc.opts)
			skipWithoutKernels(t, err)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %v does not mention %q", err, tc.want)
			}
		})
	}
}

type frontendGolden struct {
	Signal string `json:"signal"`
	PCM    string `json:"pcm"`
	Chunks []struct {
		SamplesRead  int     `json:"samples_read"`
		FeaturesInt8 []int8  `json:"features_int8"`
		FeaturesU16  []int32 `json:"features_u16"`
	} `json:"chunks"`
}

func loadFrontendGolden(t *testing.T, path string) (frontendGolden, []int16) {
	t.Helper()
	var g frontendGolden
	if err := json.Unmarshal(readFile(t, path), &g); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	raw, err := base64.StdEncoding.DecodeString(g.PCM)
	if err != nil {
		t.Fatalf("%s: pcm: %v", path, err)
	}
	pcm := make([]int16, len(raw)/2)
	for i := range pcm {
		pcm[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return g, pcm
}

// TestReplayFrontendFeaturesRealModel runs the golden frontend signals through a real model
// twice, as precomputed int8 frames through FeedFeatures and as PCM through Feed, and checks
// that both paths agree on the frames fed and the events raised.
func TestReplayFrontendFeaturesRealModel(t *testing.T) {
	model, config := loadModel(t, "hey_jarvis")
	vadModel, vadConfig := loadModel(t, "vad")
	files, err := filepath.Glob("../testdata/frontend/*.json")
	if err != nil || len(files) == 0 {
		t.Fatalf("no frontend golden files: %v", err)
	}
	for _, withVAD := range []bool{false, true} {
		opts := Options{}
		if withVAD {
			opts.VADModel, opts.VADConfig = vadModel, vadConfig
		}
		d, err := NewDetector(model, config, opts)
		skipWithoutKernels(t, err)
		if err != nil {
			t.Fatalf("NewDetector: %v", err)
		}
		for _, path := range files {
			g, pcm := loadFrontendGolden(t, path)
			t.Run(g.Signal+map[bool]string{false: "", true: "+vad"}[withVAD], func(t *testing.T) {
				if err := d.Reset(); err != nil {
					t.Fatal(err)
				}
				var viaFeatures []Event
				frames := 0
				for i, c := range g.Chunks {
					if c.FeaturesInt8 == nil {
						continue
					}
					ev, ok, err := d.FeedFeatures(c.FeaturesInt8)
					if err != nil {
						t.Fatalf("chunk %d: %v", i, err)
					}
					frames++
					if ok {
						viaFeatures = append(viaFeatures, ev)
					}
				}
				if d.Frames() != frames {
					t.Errorf("Frames %d, want %d", d.Frames(), frames)
				}

				if err := d.Reset(); err != nil {
					t.Fatal(err)
				}
				var viaPCM []Event
				for off := 0; off < len(pcm); off += 1000 {
					evs, err := d.Feed(pcm[off:min(off+1000, len(pcm))])
					if err != nil {
						t.Fatalf("Feed at %d: %v", off, err)
					}
					viaPCM = append(viaPCM, evs...)
				}
				if d.Frames() != frames {
					t.Errorf("Feed produced %d frames, FeedFeatures %d", d.Frames(), frames)
				}
				if len(viaPCM) != len(viaFeatures) {
					t.Fatalf("events via PCM %v, via features %v", viaPCM, viaFeatures)
				}
				for i := range viaPCM {
					if viaPCM[i] != viaFeatures[i] {
						t.Errorf("event %d via PCM %+v, via features %+v", i, viaPCM[i], viaFeatures[i])
					}
				}
				t.Logf("%d frames, %d events", frames, len(viaFeatures))
			})
		}
	}
}

func TestFeedDoesNotAllocateWithRealModel(t *testing.T) {
	model, config := loadModel(t, "hey_jarvis")
	d, err := NewDetector(model, config, Options{})
	skipWithoutKernels(t, err)
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]int16, 1600) // 100 ms, 10 frames
	allocs := testing.AllocsPerRun(20, func() {
		if _, err := d.Feed(pcm); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Errorf("Feed allocates %v times per 100 ms of audio", allocs)
	}
}
