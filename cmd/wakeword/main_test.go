package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tgallice/wakeword-go/internal/wav"
	"github.com/tgallice/wakeword-go/runtime"
)

const modelsDir = "../../testdata/models/v2"

func sine(n int, hz, amp float64) []int16 {
	pcm := make([]int16, n)
	for i := range pcm {
		pcm[i] = int16(amp * math.Sin(2*math.Pi*hz*float64(i)/wav.SampleRate))
	}
	return pcm
}

func wavFile(t *testing.T, pcm []int16) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "in.wav")
	var buf bytes.Buffer
	if err := wav.Write(&buf, pcm); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadWAVRoundTrip(t *testing.T) {
	pcm := sine(3200, 440, 8000)
	var buf bytes.Buffer
	if err := wav.Write(&buf, pcm); err != nil {
		t.Fatal(err)
	}
	got, err := wav.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(pcm) {
		t.Fatalf("%d samples, want %d", len(got), len(pcm))
	}
	for i := range pcm {
		if got[i] != pcm[i] {
			t.Fatalf("sample %d = %d, want %d", i, got[i], pcm[i])
		}
	}
}

func TestReadWAVSkipsChunksAndStreamingSize(t *testing.T) {
	pcm := sine(160, 440, 1000)
	var buf bytes.Buffer
	if err := wav.Write(&buf, pcm); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	// Insert a LIST chunk of odd size (padded) between fmt and data, and mark the data size as
	// unknown (0xFFFFFFFF), as streaming writers do.
	list := append([]byte("LIST"), 3, 0, 0, 0, 'a', 'b', 'c', 0)
	out := append([]byte{}, b[:36]...)
	out = append(out, list...)
	out = append(out, b[36:40]...)
	out = append(out, 0xFF, 0xFF, 0xFF, 0xFF)
	out = append(out, b[44:]...)
	got, err := wav.Read(bytes.NewReader(out))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(pcm) || got[10] != pcm[10] {
		t.Errorf("%d samples, want %d", len(got), len(pcm))
	}
}

func TestReadWAVRejects(t *testing.T) {
	good := func() []byte {
		var buf bytes.Buffer
		if err := wav.Write(&buf, sine(160, 440, 1000)); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	cases := []struct {
		name string
		mut  func([]byte) []byte
		want string
	}{
		{"empty", func(b []byte) []byte { return nil }, "not a RIFF/WAVE"},
		{"not riff", func(b []byte) []byte { copy(b, "RIFX"); return b }, "not a RIFF/WAVE"},
		{"stereo", func(b []byte) []byte { b[22] = 2; return b }, "2 channels"},
		{"44100", func(b []byte) []byte { binary.LittleEndian.PutUint32(b[24:], 44100); return b }, "44100 Hz"},
		{"8 bit", func(b []byte) []byte { b[34] = 8; return b }, "8 bits"},
		{"float", func(b []byte) []byte { b[20] = 3; return b }, "format tag 3"},
		{"truncated data", func(b []byte) []byte { return b[:100] }, "data chunk"},
		{"no data", func(b []byte) []byte { return b[:36] }, "no data chunk"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := wav.Read(bytes.NewReader(tc.mut(good())))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %v does not mention %q", err, tc.want)
			}
		})
	}
}

func TestFeaturesCommand(t *testing.T) {
	path := wavFile(t, sine(16000, 440, 8000)) // 1 s: 98 frames
	var stdout, stderr bytes.Buffer
	if code := run([]string{"features", path}, nil, &stdout, &stderr); code != exitOK {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	lines := strings.Split(strings.TrimSpace(stdout.String()), "\n")
	if len(lines) != 98 {
		t.Fatalf("%d lines, want 98", len(lines))
	}
	if !strings.HasPrefix(lines[0], "0 t=0.030s u16:") || !strings.Contains(lines[0], " | int8:") {
		t.Errorf("first line %q", lines[0])
	}
	if !strings.HasPrefix(lines[97], "97 t=1.000s u16:") {
		t.Errorf("last line %q", lines[97])
	}
	if fields := strings.Fields(lines[5]); len(fields) != 2+1+40+2+40 {
		t.Errorf("line has %d fields, want %d: %q", len(fields), 2+1+40+2+40, lines[5])
	}

	// The same audio as raw samples on stdin gives the same output.
	raw := make([]byte, 32000)
	for i, s := range sine(16000, 440, 8000) {
		binary.LittleEndian.PutUint16(raw[2*i:], uint16(s))
	}
	var stdout2 bytes.Buffer
	if code := run([]string{"features", "--raw"}, bytes.NewReader(raw), &stdout2, &stderr); code != exitOK {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	if stdout2.String() != stdout.String() {
		t.Error("raw stdin output differs from WAV output")
	}
}

func TestUsageErrors(t *testing.T) {
	wav := wavFile(t, sine(1600, 440, 1000))
	model := filepath.Join(modelsDir, "hey_jarvis.tflite")
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"no args", nil, "usage:"},
		{"unknown", []string{"frobnicate"}, "unknown command"},
		{"detect without model", []string{"detect", wav}, "--model is required"},
		{"detect without input", []string{"detect", "--model", model}, "exactly one of"},
		{"detect two inputs", []string{"detect", "--model", model, "--raw", wav}, "exactly one of"},
		{"detect features and wav", []string{"detect", "--model", model, "--features", "x.json", wav}, "exactly one of"},
		{"features without input", []string{"features"}, "exactly one of"},
		{"features two args", []string{"features", wav, wav}, "too many arguments"},
		{"bad flag", []string{"detect", "--bogus"}, "flag provided but not defined"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := run(tc.args, strings.NewReader(""), &stdout, &stderr); code != exitUsage {
				t.Errorf("exit %d, want %d (stderr %q)", code, exitUsage, stderr.String())
			}
			if !strings.Contains(stderr.String(), tc.want) {
				t.Errorf("stderr %q does not mention %q", stderr.String(), tc.want)
			}
		})
	}
	var stdout, stderr bytes.Buffer
	if code := run([]string{"help"}, nil, &stdout, &stderr); code != exitOK {
		t.Errorf("help exit %d", code)
	}
}

func TestDetectRuntimeErrors(t *testing.T) {
	model := filepath.Join(modelsDir, "hey_jarvis.tflite")
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"missing model file", []string{"detect", "--model", "nope.tflite", "--raw"}, "nope.tflite"},
		{"missing config", []string{"detect", "--model", model, "--config", "nope.json", "--raw"}, "nope.json"},
		{
			"v1 model",
			[]string{"detect", "--model", "../../testdata/models/v1/okay_nabu.tflite", "--config", filepath.Join(modelsDir, "okay_nabu.json"), "--raw"},
			"unsupported operators",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := run(tc.args, strings.NewReader(""), &stdout, &stderr); code != exitRuntime {
				t.Errorf("exit %d, want %d (stderr %q)", code, exitRuntime, stderr.String())
			}
			if !strings.Contains(stderr.String(), tc.want) {
				t.Errorf("stderr %q does not mention %q", stderr.String(), tc.want)
			}
		})
	}
}

// skipWithoutKernels skips when the detector cannot be built because an operator kernel
// is missing (all standard microWakeWord operators should be implemented).
func skipWithoutKernels(t *testing.T, stderr string) {
	t.Helper()
	if strings.Contains(stderr, runtime.ErrNoKernel.Error()) {
		t.Skipf("kernel not implemented: %s", strings.TrimSpace(stderr))
	}
}

func TestDetectReplaysGoldenFeatures(t *testing.T) {
	files, err := filepath.Glob("../../testdata/frontend/*.json")
	if err != nil || len(files) == 0 {
		t.Fatalf("no golden files: %v", err)
	}
	model := filepath.Join(modelsDir, "hey_jarvis.tflite")
	vad := filepath.Join(modelsDir, "vad.tflite")
	for _, path := range files {
		t.Run(filepath.Base(path), func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := run([]string{"detect", "--model", model, "--vad", vad, "--verbose", "--features", path},
				nil, &stdout, &stderr)
			skipWithoutKernels(t, stderr.String())
			if code != exitOK {
				t.Fatalf("exit %d: %s", code, stderr.String())
			}
			for _, line := range strings.Split(strings.TrimSpace(stdout.String()), "\n") {
				if line != "" && !strings.HasPrefix(line, "t=") {
					t.Errorf("unexpected output line %q", line)
				}
			}
			t.Logf("%s: %q", filepath.Base(path), strings.TrimSpace(stdout.String()))
		})
	}
}

func TestDetectOnWAV(t *testing.T) {
	model := filepath.Join(modelsDir, "hey_jarvis.tflite")
	wav := wavFile(t, sine(32000, 440, 8000))
	var stdout, stderr bytes.Buffer
	code := run([]string{"detect", "--model", model, wav}, nil, &stdout, &stderr)
	skipWithoutKernels(t, stderr.String())
	if code != exitOK {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	if strings.Contains(stdout.String(), "detected") {
		t.Errorf("a 440 Hz sine is not the wake word: %s", stdout.String())
	}
}
