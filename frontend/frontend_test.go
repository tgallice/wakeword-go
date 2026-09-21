package frontend

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// goldenSignal is one file of testdata/frontend, produced by
// tools/oracle/gen_frontend_golden.py with pymicro-features.
type goldenSignal struct {
	Signal       string `json:"signal"`
	SampleRate   int    `json:"sample_rate"`
	ChunkSamples int    `json:"chunk_samples"`
	PCM          string `json:"pcm"`
	Chunks       []struct {
		SamplesRead  int      `json:"samples_read"`
		FeaturesU16  []uint16 `json:"features_u16"`
		FeaturesInt8 []int8   `json:"features_int8"`
	} `json:"chunks"`
}

func loadGolden(t *testing.T, path string) (goldenSignal, []int16) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var g goldenSignal
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	raw, err := base64.StdEncoding.DecodeString(g.PCM)
	if err != nil {
		t.Fatalf("decode pcm of %s: %v", path, err)
	}
	pcm := make([]int16, len(raw)/2)
	for i := range pcm {
		pcm[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	return g, pcm
}

func goldenFiles(t *testing.T) []string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join("..", "testdata", "frontend", "*.json"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no golden files found: %v", err)
	}
	return files
}

// TestGoldenSignals replays every oracle signal block by block and requires
// exact equality of consumption, frame presence, every uint16 feature and its
// int8 conversion.
func TestGoldenSignals(t *testing.T) {
	var totalFrames, totalValues int
	for _, path := range goldenFiles(t) {
		g, pcm := loadGolden(t, path)
		t.Run(g.Signal, func(t *testing.T) {
			if g.SampleRate != DefaultConfig().SampleRate {
				t.Fatalf("golden sample rate %d, frontend expects %d", g.SampleRate, DefaultConfig().SampleRate)
			}
			f := New()
			step := g.ChunkSamples
			int8Buf := make([]int8, f.NumChannels())
			for b, chunk := range g.Chunks {
				block := pcm[b*step : (b+1)*step]
				features, consumed := f.Process(block)
				if consumed != chunk.SamplesRead {
					t.Fatalf("block %d: consumed %d samples, want %d", b, consumed, chunk.SamplesRead)
				}
				if (features != nil) != (chunk.FeaturesU16 != nil) {
					t.Fatalf("block %d: frame produced %v, want %v", b, features != nil, chunk.FeaturesU16 != nil)
				}
				if features == nil {
					continue
				}
				totalFrames++
				if len(features) != len(chunk.FeaturesU16) {
					t.Fatalf("block %d: %d features, want %d", b, len(features), len(chunk.FeaturesU16))
				}
				for ch, want := range chunk.FeaturesU16 {
					totalValues++
					if features[ch] != want {
						t.Fatalf("signal %s block %d channel %d: got %d, want %d", g.Signal, b, ch, features[ch], want)
					}
				}
				QuantizeFeatures(int8Buf, features)
				for ch, want := range chunk.FeaturesInt8 {
					if int8Buf[ch] != want {
						t.Fatalf("signal %s block %d channel %d: int8 got %d, want %d", g.Signal, b, ch, int8Buf[ch], want)
					}
				}
			}
		})
	}
	t.Logf("%d frames, %d feature values compared", totalFrames, totalValues)
}

// TestProcessLargeBuffer checks that feeding a whole signal at once, looping
// on the consumed count, yields the same frames as block by block feeding.
func TestProcessLargeBuffer(t *testing.T) {
	for _, path := range goldenFiles(t) {
		g, pcm := loadGolden(t, path)
		t.Run(g.Signal, func(t *testing.T) {
			f := New()
			var got [][]uint16
			rest := pcm
			for len(rest) > 0 {
				features, consumed := f.Process(rest)
				if consumed <= 0 {
					t.Fatalf("no progress with %d samples left", len(rest))
				}
				rest = rest[consumed:]
				if features != nil {
					got = append(got, append([]uint16(nil), features...))
				}
			}
			var want [][]uint16
			for _, c := range g.Chunks {
				if c.FeaturesU16 != nil {
					want = append(want, c.FeaturesU16)
				}
			}
			if len(got) != len(want) {
				t.Fatalf("%d frames, want %d", len(got), len(want))
			}
			for i := range want {
				for ch := range want[i] {
					if got[i][ch] != want[i][ch] {
						t.Fatalf("frame %d channel %d: got %d, want %d", i, ch, got[i][ch], want[i][ch])
					}
				}
			}
		})
	}
}

// TestReset checks that Reset restores the initial state: the same signal
// processed twice on one Frontend gives the same frames both times.
func TestReset(t *testing.T) {
	_, pcm := loadGolden(t, filepath.Join("..", "testdata", "frontend", "white_noise.json"))
	f := New()
	run := func() [][]uint16 {
		var frames [][]uint16
		for off := 0; off+160 <= len(pcm); off += 160 {
			if features, _ := f.Process(pcm[off : off+160]); features != nil {
				frames = append(frames, append([]uint16(nil), features...))
			}
		}
		return frames
	}
	first := run()
	f.Reset()
	second := run()
	if len(first) != len(second) || len(first) == 0 {
		t.Fatalf("%d frames then %d", len(first), len(second))
	}
	for i := range first {
		for ch := range first[i] {
			if first[i][ch] != second[i][ch] {
				t.Fatalf("frame %d channel %d differs after Reset: %d vs %d", i, ch, first[i][ch], second[i][ch])
			}
		}
	}
	// Reset half way through a window must drop the pending samples.
	f.Reset()
	f.Process(pcm[:160])
	f.Reset()
	if features, consumed := f.Process(pcm[:160]); features != nil || consumed != 160 {
		t.Fatalf("after Reset: frame %v consumed %d, want no frame and 160", features != nil, consumed)
	}
}

func TestProcessConsumption(t *testing.T) {
	f := New()
	size, step := f.WindowSize(), f.WindowStep()
	if size != 480 || step != 160 {
		t.Fatalf("window %d/%d, want 480/160", size, step)
	}
	buf := make([]int16, 1000)
	// First call: only a full window worth is taken.
	features, consumed := f.Process(buf)
	if features == nil || consumed != size {
		t.Fatalf("first call: frame %v consumed %d, want frame and %d", features != nil, consumed, size)
	}
	// Then one step per frame.
	features, consumed = f.Process(buf)
	if features == nil || consumed != step {
		t.Fatalf("second call: frame %v consumed %d, want frame and %d", features != nil, consumed, step)
	}
	// A short buffer is fully consumed without a frame.
	features, consumed = f.Process(buf[:10])
	if features != nil || consumed != 10 {
		t.Fatalf("short call: frame %v consumed %d, want no frame and 10", features != nil, consumed)
	}
	// An empty buffer consumes nothing.
	if features, consumed = f.Process(nil); features != nil || consumed != 0 {
		t.Fatalf("empty call: frame %v consumed %d", features != nil, consumed)
	}
}

func TestProcessNoAllocation(t *testing.T) {
	_, pcm := loadGolden(t, filepath.Join("..", "testdata", "frontend", "white_noise.json"))
	f := New()
	off := 0
	allocs := testing.AllocsPerRun(200, func() {
		f.Process(pcm[off : off+160])
		off = (off + 160) % (len(pcm) - 160)
	})
	if allocs != 0 {
		t.Fatalf("Process allocates %v times per call, want 0", allocs)
	}
	allocs = testing.AllocsPerRun(20, f.Reset)
	if allocs != 0 {
		t.Fatalf("Reset allocates %v times per call, want 0", allocs)
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{"zero sample rate", func(c *Config) { c.SampleRate = 0 }},
		{"zero window", func(c *Config) { c.WindowSizeMS = 0 }},
		{"step above size", func(c *Config) { c.WindowStepMS = c.WindowSizeMS + 1 }},
		{"zero channels", func(c *Config) { c.NumChannels = 0 }},
		{"inverted bands", func(c *Config) { c.UpperBandLimit = c.LowerBandLimit }},
		{"negative lower band", func(c *Config) { c.LowerBandLimit = -1 }},
		{"smoothing bits", func(c *Config) { c.NoiseSmoothingBits = 40 }},
		{"gain bits", func(c *Config) { c.PCANGainBits = -1 }},
		{"log shift", func(c *Config) { c.LogScaleShift = 32 }},
		{"band above nyquist", func(c *Config) { c.UpperBandLimit = 9000 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tc.mutate(&cfg)
			if _, err := NewWithConfig(cfg); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
	f, err := NewWithConfig(DefaultConfig())
	if err != nil {
		t.Fatalf("default config rejected: %v", err)
	}
	if f.Config() != DefaultConfig() || f.NumChannels() != 40 {
		t.Fatal("configuration not preserved")
	}
	cfg := DefaultConfig()
	cfg.PCANEnabled = false
	f, err = NewWithConfig(cfg)
	if err != nil {
		t.Fatalf("PCAN disabled: %v", err)
	}
	if f.pcan != nil {
		t.Fatal("PCAN stage built while disabled")
	}
}

func BenchmarkProcess(b *testing.B) {
	data, err := os.ReadFile(filepath.Join("..", "testdata", "frontend", "white_noise.json"))
	if err != nil {
		b.Fatal(err)
	}
	var g goldenSignal
	if err := json.Unmarshal(data, &g); err != nil {
		b.Fatal(err)
	}
	raw, _ := base64.StdEncoding.DecodeString(g.PCM)
	pcm := make([]int16, len(raw)/2)
	for i := range pcm {
		pcm[i] = int16(binary.LittleEndian.Uint16(raw[2*i:]))
	}
	f := New()
	off := 0
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		f.Process(pcm[off : off+160])
		off = (off + 160) % (len(pcm) - 160)
	}
	// One block of 160 samples is 10 ms of audio: the ratio of CPU time to
	// audio time, per block.
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/1e7, "cpu/realtime")
}
