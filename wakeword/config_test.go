package wakeword

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const modelsDir = "../testdata/models/v2"

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return b
}

func TestParseModelConfigV2Manifests(t *testing.T) {
	cases := []struct {
		name   string
		word   string
		cutoff uint8
		window int
	}{
		{"alexa", "Alexa", 0, 0},
		{"hey_jarvis", "Hey Jarvis", 0, 0},
		{"hey_mycroft", "Hey Mycroft", 0, 0},
		{"okay_nabu", "Okay Nabu", 247, 5},
		{"vad", "vad", 127, 5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := ParseModelConfig(readFile(t, filepath.Join(modelsDir, tc.name+".json")))
			if err != nil {
				t.Fatalf("ParseModelConfig: %v", err)
			}
			if cfg.WakeWord != tc.word {
				t.Errorf("wake_word %q, want %q", cfg.WakeWord, tc.word)
			}
			if cfg.Version != 2 || cfg.Type != "micro" {
				t.Errorf("type %q version %d, want micro 2", cfg.Type, cfg.Version)
			}
			if cfg.Micro.FeatureStepSize != 10 {
				t.Errorf("feature_step_size %d, want 10", cfg.Micro.FeatureStepSize)
			}
			if tc.cutoff != 0 && cfg.CutoffUint8() != tc.cutoff {
				t.Errorf("cutoff uint8 %d, want %d", cfg.CutoffUint8(), tc.cutoff)
			}
			if tc.window != 0 && cfg.Micro.SlidingWindowSize != tc.window {
				t.Errorf("sliding_window_size %d, want %d", cfg.Micro.SlidingWindowSize, tc.window)
			}
		})
	}
}

func TestCutoffUint8TruncatesLikeESPHome(t *testing.T) {
	cases := []struct {
		p    float64
		want uint8
	}{
		{0.97, 247}, // 247.35 truncated, as int(0.97 * 255) in Python
		{0.5, 127},  // 127.5 truncated, not rounded to 128
		{1, 255},
		{0.999, 254},
		{0.001, 0},
		{0, 0},
		{-1, 0},
		{2, 255},
	}
	for _, tc := range cases {
		if got := CutoffUint8(tc.p); got != tc.want {
			t.Errorf("CutoffUint8(%v) = %d, want %d", tc.p, got, tc.want)
		}
	}
}

func TestParseModelConfigRejects(t *testing.T) {
	valid := `{"type":"micro","wake_word":"x","version":2,"micro":{"probability_cutoff":0.9,"sliding_window_size":5,"feature_step_size":10}}`
	cases := []struct {
		name string
		json string
		want string
	}{
		{"not json", `{`, "model config"},
		{"version 1", strings.Replace(valid, `"version":2`, `"version":1`, 1), "version 1, want 2"},
		{"type", strings.Replace(valid, `"micro","wake`, `"tflite","wake`, 1), `type "tflite"`},
		{"cutoff zero", strings.Replace(valid, `0.9`, `0`, 1), "probability_cutoff 0"},
		{"cutoff above one", strings.Replace(valid, `0.9`, `1.5`, 1), "probability_cutoff 1.5"},
		{"window zero", strings.Replace(valid, `"sliding_window_size":5`, `"sliding_window_size":0`, 1), "sliding_window_size 0"},
		{"step", strings.Replace(valid, `"feature_step_size":10`, `"feature_step_size":20`, 1), "feature_step_size 20"},
		{"no word", strings.Replace(valid, `"wake_word":"x",`, ``, 1), "missing wake_word"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseModelConfig([]byte(tc.json))
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err, tc.want)
			}
		})
	}
	if _, err := ParseModelConfig([]byte(valid)); err != nil {
		t.Errorf("valid manifest rejected: %v", err)
	}
}
