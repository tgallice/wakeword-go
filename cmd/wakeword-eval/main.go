// Command wakeword-eval measures a microWakeWord model on the synthetic corpus produced by
// tools/corpus/generate.py: recall on positives, false accepts per hour on negatives and
// the near-miss acceptance rate, for a sweep of cutoffs, with and without the VAD model.
//
//	wakeword-eval --corpus tools/corpus/corpus --models testdata/models/v2 [--word okay_nabu]
//	              [--cutoffs 0.5,0.8,0.9,0.97,0.99] [--preroll 3.5] [--jobs 8]
//
// Every file is fed after a preroll of near-silence (a few LSB of dither) so that the
// ESPHome warm-up of 100 invocations (3 s) is over before the audio starts; a detection
// is then never lost to the warm-up. Event times are reported relative to the file.
//
// The detector is the real wakeword package: nothing is reimplemented here.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/tgallice/wakeword-go/internal/wav"
	"github.com/tgallice/wakeword-go/wakeword"
)

// manifest mirrors tools/corpus/generate.py output.
type manifest struct {
	Entries []entry `json:"entries"`
}

type entry struct {
	File          string  `json:"file"`
	Subset        string  `json:"subset"`
	Word          string  `json:"word"`
	Voice         string  `json:"voice"`
	Duration      float64 `json:"duration_s"`
	ExpectedStart float64 `json:"expected_start_s"`
	ExpectedEnd   float64 `json:"expected_end_s"`
	Noise         string  `json:"noise"`
	SNR           float64 `json:"snr_db"`
	Segments      []struct {
		Text  string  `json:"text"`
		Start float64 `json:"start_s"`
		End   float64 `json:"end_s"`
	} `json:"segments"`
}

// sentenceAt names the negative sentence being spoken at time t (the one that ended most
// recently, since a detection lands after the words), or "?" when unknown.
func (e entry) sentenceAt(t float64) string {
	best := "?"
	for _, seg := range e.Segments {
		if seg.Start <= t+0.2 {
			best = seg.Text
		}
	}
	return best
}

// detectionSlack is how long after the expected end of the wake word a detection still
// counts: the sliding window and the model's own latency add a few hundred milliseconds.
const detectionSlack = 1.5

type setting struct {
	cutoff float64
	vad    bool
}

type result struct {
	posHit, posTotal  int
	embHit, embTotal  int
	nmHit, nmTotal    int // near misses aimed at this model
	nmAllHit, nmAll   int // every near miss, whatever its target
	falseAccepts      int
	negHours          float64
	noiseAccepts      int
	noiseHours        float64
	voiceMiss         map[string]int
	voiceTotal        map[string]int
	noisyMiss, noisyN int
	cleanMiss, cleanN int
	lateHit           int
	falseAcceptLog    []string
}

func newResult() *result {
	return &result{voiceMiss: map[string]int{}, voiceTotal: map[string]int{}}
}

func main() {
	corpus := flag.String("corpus", "tools/corpus/corpus", "corpus directory with manifest.json")
	models := flag.String("models", "testdata/models/v2", "directory with <word>.tflite and <word>.json")
	word := flag.String("word", "", "evaluate only this model (default: alexa, hey_jarvis, hey_mycroft, okay_nabu)")
	cutoffs := flag.String("cutoffs", "0.5,0.8,0.9,0.97,0.99", "comma separated cutoffs to sweep, besides the manifest one")
	preroll := flag.Float64("preroll", 3.5, "seconds of near-silence fed before each file")
	jobs := flag.Int("jobs", runtime.NumCPU(), "parallel files")
	verbose := flag.Bool("verbose", false, "list every false accept at the manifest cutoff with the sentence being spoken")
	flag.Parse()

	if err := run(*corpus, *models, *word, *cutoffs, *preroll, *jobs, *verbose); err != nil {
		fmt.Fprintln(os.Stderr, "wakeword-eval:", err)
		os.Exit(2)
	}
}

func run(corpus, models, only, cutoffList string, preroll float64, jobs int, verbose bool) error {
	raw, err := os.ReadFile(filepath.Join(corpus, "manifest.json"))
	if err != nil {
		return err
	}
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return fmt.Errorf("manifest: %w", err)
	}
	words := []string{"alexa", "hey_jarvis", "hey_mycroft", "okay_nabu"}
	if only != "" {
		words = []string{only}
	}
	vadModel, err := os.ReadFile(filepath.Join(models, "vad.tflite"))
	if err != nil {
		return err
	}
	vadConfig, err := os.ReadFile(filepath.Join(models, "vad.json"))
	if err != nil {
		return err
	}

	// Load every file once.
	fmt.Fprintf(os.Stderr, "loading %d files\n", len(m.Entries))
	audio := make([][]int16, len(m.Entries))
	if err := parallel(jobs, len(m.Entries), func(i int) error {
		pcm, err := wav.ReadFile(filepath.Join(corpus, m.Entries[i].File))
		audio[i] = pcm
		return err
	}); err != nil {
		return err
	}
	pre := dither(int(preroll * wav.SampleRate))

	for _, w := range words {
		model, err := os.ReadFile(filepath.Join(models, w+".tflite"))
		if err != nil {
			return err
		}
		config, err := os.ReadFile(filepath.Join(models, w+".json"))
		if err != nil {
			return err
		}
		cfg, err := wakeword.ParseModelConfig(config)
		if err != nil {
			return err
		}
		settings, err := buildSettings(cfg.Micro.ProbabilityCutoff, cutoffList)
		if err != nil {
			return err
		}
		fmt.Printf("\n## %s (%q, manifest cutoff %.2f, window %d)\n\n", w, cfg.WakeWord,
			cfg.Micro.ProbabilityCutoff, cfg.Micro.SlidingWindowSize)
		fmt.Println("| cutoff | VAD | recall alone | recall embedded | near-miss (targeted) | near-miss (all) | false accepts/h (speech) | false accepts/h (noise+music) |")
		fmt.Println("|---|---|---|---|---|---|---|---|")
		var detail strings.Builder
		for _, s := range settings {
			opts := wakeword.Options{ProbabilityCutoff: s.cutoff}
			if s.vad {
				opts.VADModel, opts.VADConfig = vadModel, vadConfig
			}
			r, err := evaluate(m.Entries, audio, pre, w, model, config, opts, jobs)
			if err != nil {
				return fmt.Errorf("%s cutoff %.2f vad %v: %w", w, s.cutoff, s.vad, err)
			}
			mark := ""
			if s.cutoff == cfg.Micro.ProbabilityCutoff {
				mark = " (manifest)"
			}
			fmt.Printf("| %.2f%s | %s | %s | %s | %s | %s | %s | %s |\n", s.cutoff, mark, yesNo(s.vad),
				pct(r.posHit, r.posTotal), pct(r.embHit, r.embTotal), pct(r.nmHit, r.nmTotal),
				pct(r.nmAllHit, r.nmAll), perHour(r.falseAccepts, r.negHours), perHour(r.noiseAccepts, r.noiseHours))
			if s.cutoff == cfg.Micro.ProbabilityCutoff && !s.vad {
				fmt.Fprintf(&detail, "\nAt the manifest cutoff without VAD: clean clips %s, noisy clips %s",
					pct(r.cleanN-r.cleanMiss, r.cleanN), pct(r.noisyN-r.noisyMiss, r.noisyN))
				if r.lateHit > 0 {
					fmt.Fprintf(&detail, ", %d detections later than %.1f s after the word (not counted)", r.lateHit, detectionSlack)
				}
				fmt.Fprintf(&detail, ".\nMisses per voice (alone + embedded):")
				voices := make([]string, 0, len(r.voiceTotal))
				for v := range r.voiceTotal {
					voices = append(voices, v)
				}
				sort.Strings(voices)
				for _, v := range voices {
					if r.voiceMiss[v] > 0 {
						fmt.Fprintf(&detail, " %s %d/%d;", v, r.voiceMiss[v], r.voiceTotal[v])
					}
				}
				fmt.Fprintln(&detail)
				if verbose && len(r.falseAcceptLog) > 0 {
					sort.Strings(r.falseAcceptLog)
					fmt.Fprintf(&detail, "False accepts on speech:\n")
					for _, l := range r.falseAcceptLog {
						fmt.Fprintf(&detail, "  %s\n", l)
					}
				}
			}
		}
		fmt.Print(detail.String())
		fmt.Printf("Speech negatives: %.1f min; noise and music: %.1f min; near misses aimed at this model: %d clips.\n",
			hoursOf(m.Entries, "negative")*60, hoursOf(m.Entries, "noise")*60, countNM(m.Entries, w))
	}
	return nil
}

func buildSettings(manifestCutoff float64, list string) ([]setting, error) {
	cut := []float64{manifestCutoff}
	for _, f := range strings.Split(list, ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		v, err := strconv.ParseFloat(f, 64)
		if err != nil || v <= 0 || v > 1 {
			return nil, fmt.Errorf("bad cutoff %q", f)
		}
		if v != manifestCutoff {
			cut = append(cut, v)
		}
	}
	sort.Float64s(cut)
	var out []setting
	for _, c := range cut {
		out = append(out, setting{c, false}, setting{c, true})
	}
	return out, nil
}

// evaluate runs one (model, options) over the whole corpus, one detector per worker.
func evaluate(
	entries []entry, audio [][]int16, pre []int16, word string, model, config []byte,
	opts wakeword.Options, jobs int,
) (*result, error) {
	var mu sync.Mutex
	r := newResult()
	pool := sync.Pool{New: func() any {
		d, err := wakeword.NewDetector(model, config, opts)
		if err != nil {
			return err
		}
		return d
	}}
	err := parallel(jobs, len(entries), func(i int) error {
		got := pool.Get()
		if err, ok := got.(error); ok {
			return err
		}
		det := got.(*wakeword.Detector)
		defer pool.Put(det)
		times, err := detectTimes(det, pre, audio[i])
		if err != nil {
			return fmt.Errorf("%s: %w", entries[i].File, err)
		}
		e := entries[i]
		mu.Lock()
		defer mu.Unlock()
		switch e.Subset {
		case "positive", "embedded":
			if e.Word != word {
				return nil
			}
			hit, late := inWindow(times, e.ExpectedStart, e.ExpectedEnd+detectionSlack)
			r.lateHit += late
			r.voiceTotal[e.Voice]++
			if !hit {
				r.voiceMiss[e.Voice]++
			}
			if e.Noise != "" {
				r.noisyN++
				if !hit {
					r.noisyMiss++
				}
			} else {
				r.cleanN++
				if !hit {
					r.cleanMiss++
				}
			}
			if e.Subset == "positive" {
				r.posTotal++
				if hit {
					r.posHit++
				}
			} else {
				r.embTotal++
				if hit {
					r.embHit++
				}
			}
		case "near_miss":
			r.nmAll++
			if len(times) > 0 {
				r.nmAllHit++
			}
			if e.Word == word {
				r.nmTotal++
				if len(times) > 0 {
					r.nmHit++
				}
			}
		case "negative":
			r.falseAccepts += len(times)
			r.negHours += e.Duration / 3600
			for _, t := range times {
				r.falseAcceptLog = append(r.falseAcceptLog, fmt.Sprintf("%s at %.1f s: %q", e.File, t, e.sentenceAt(t)))
			}
		case "noise":
			r.noiseAccepts += len(times)
			r.noiseHours += e.Duration / 3600
		}
		return nil
	})
	return r, err
}

// detectTimes feeds the preroll then the file and returns the detection times in seconds,
// relative to the start of the file. Detections blocked by the VAD do not count.
func detectTimes(det *wakeword.Detector, pre, pcm []int16) ([]float64, error) {
	if err := det.Reset(); err != nil {
		return nil, err
	}
	var times []float64
	feed := func(p []int16) error {
		const chunk = 1600
		for off := 0; off < len(p); off += chunk {
			evs, err := det.Feed(p[off:min(off+chunk, len(p))])
			if err != nil {
				return err
			}
			for _, ev := range evs {
				if !ev.BlockedByVAD {
					times = append(times, float64(ev.Sample-len(pre))/wav.SampleRate)
				}
			}
		}
		return nil
	}
	if err := feed(pre); err != nil {
		return nil, err
	}
	if err := feed(pcm); err != nil {
		return nil, err
	}
	return times, nil
}

func inWindow(times []float64, from, to float64) (hit bool, late int) {
	for _, t := range times {
		if t >= from && t <= to {
			hit = true
		} else if t > to {
			late++
		}
	}
	return hit, late
}

// dither returns n samples of low-level noise (up to 2 LSB), a stand-in for an idle
// microphone: digital silence would be an unrealistic warm-up signal.
func dither(n int) []int16 {
	rng := rand.New(rand.NewPCG(7, 11))
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(rng.IntN(5) - 2)
	}
	return out
}

func parallel(jobs, n int, fn func(i int) error) error {
	if jobs < 1 {
		jobs = 1
	}
	var wg sync.WaitGroup
	idx := make(chan int)
	errs := make(chan error, jobs)
	for range jobs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range idx {
				if err := fn(i); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	var first error
loop:
	for i := range n {
		select {
		case idx <- i:
		case first = <-errs:
			break loop
		}
	}
	close(idx)
	wg.Wait()
	close(errs)
	for err := range errs {
		first = errors.Join(first, err)
	}
	return first
}

func pct(hit, total int) string {
	if total == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.1f%% (%d/%d)", 100*float64(hit)/float64(total), hit, total)
}

func perHour(n int, hours float64) string {
	if hours == 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.2f (%d)", float64(n)/hours, n)
}

func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func hoursOf(entries []entry, subset string) float64 {
	var s float64
	for _, e := range entries {
		if e.Subset == subset {
			s += e.Duration
		}
	}
	return s / 3600
}

func countNM(entries []entry, word string) int {
	n := 0
	for _, e := range entries {
		if e.Subset == "near_miss" && e.Word == word {
			n++
		}
	}
	return n
}
