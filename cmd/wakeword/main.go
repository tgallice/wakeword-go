// Command wakeword runs a microWakeWord v2 model over audio.
//
//	wakeword detect --model okay_nabu.tflite [--config okay_nabu.json] [--cutoff 0.97]
//	                [--window 5] [--vad vad.tflite] [--vad-config vad.json] [--verbose]
//	                (file.wav | --raw | --features golden.json)
//	wakeword features (file.wav | --raw)
//
// Audio must be 16 kHz mono 16-bit PCM: a WAV file, or raw samples on stdin with --raw
// (for instance `arecord -f S16_LE -r 16000 -c 1 -t raw | wakeword detect --raw ...`).
// `--features` replays the int8 frames of a testdata/frontend JSON file instead of audio.
//
// Exit codes: 0 success, 1 usage error, 2 runtime error.
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/tgallice/wakeword-go/frontend"
	"github.com/tgallice/wakeword-go/internal/wav"
	"github.com/tgallice/wakeword-go/wakeword"
)

const (
	exitOK      = 0
	exitUsage   = 1
	exitRuntime = 2
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

// warn prints a diagnostic; a failure to write to stderr has nowhere to be reported.
func warn(stderr io.Writer, format string, args ...any) {
	_, _ = fmt.Fprintf(stderr, format+"\n", args...)
}

// fail reports a runtime error, without doubling the package prefix of wakeword errors.
func fail(stderr io.Writer, err error) int {
	if strings.HasPrefix(err.Error(), "wakeword: ") {
		warn(stderr, "%v", err)
	} else {
		warn(stderr, "wakeword: %v", err)
	}
	return exitRuntime
}

// outWriter buffers stdout and keeps the first write error, checked once at the end.
type outWriter struct {
	w   *bufio.Writer
	err error
}

func newOutWriter(w io.Writer) *outWriter { return &outWriter{w: bufio.NewWriter(w)} }

func (o *outWriter) printf(format string, args ...any) {
	if o.err == nil {
		_, o.err = fmt.Fprintf(o.w, format, args...)
	}
}

func (o *outWriter) flush() error {
	if o.err != nil {
		return o.err
	}
	return o.w.Flush()
}

func usage(stderr io.Writer) int {
	warn(stderr, "%s", strings.TrimSpace(`
usage: wakeword detect --model MODEL.tflite [options] (FILE.wav | --raw | --features GOLDEN.json)
       wakeword features (FILE.wav | --raw)

detect options:
  --config PATH      model manifest (default: the .json next to the model)
  --cutoff P         probability cutoff in (0, 1], overrides the manifest
  --window N         sliding window size, overrides the manifest
  --vad PATH         VAD model gating the detections
  --vad-config PATH  VAD manifest (default: the .json next to the VAD model)
  --verbose          also print detections blocked by the VAD
  --probabilities    print the raw model output after every invocation (debugging)

Audio is 16 kHz mono 16-bit PCM; --raw reads raw samples from stdin.`))
	return exitUsage
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		return usage(stderr)
	}
	switch args[0] {
	case "detect":
		return runDetect(args[1:], stdin, stdout, stderr)
	case "features":
		return runFeatures(args[1:], stdin, stdout, stderr)
	case "-h", "--help", "help":
		usage(stderr)
		return exitOK
	default:
		warn(stderr, "wakeword: unknown command %q", args[0])
		return usage(stderr)
	}
}

// audioSource resolves the positional WAV file or --raw stdin into a sample reader.
func audioSource(path string, raw bool, stdin io.Reader) (func(yield func([]int16) error) error, error) {
	if raw == (path != "") {
		return nil, errors.New("give exactly one of a WAV file or --raw")
	}
	if raw {
		return func(yield func([]int16) error) error {
			br := bufio.NewReaderSize(stdin, 1<<16)
			buf := make([]byte, 3200) // 100 ms
			for {
				n, err := io.ReadFull(br, buf)
				if n > 0 {
					if yerr := yield(wav.DecodePCM16(buf[:n])); yerr != nil {
						return yerr
					}
				}
				if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
					return nil
				}
				if err != nil {
					return err
				}
			}
		}, nil
	}
	pcm, err := wav.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return func(yield func([]int16) error) error {
		const chunk = 1600 // 100 ms
		for off := 0; off < len(pcm); off += chunk {
			if err := yield(pcm[off:min(off+chunk, len(pcm))]); err != nil {
				return err
			}
		}
		return nil
	}, nil
}

func defaultConfigPath(model string) string {
	return strings.TrimSuffix(model, ".tflite") + ".json"
}

func runDetect(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("detect", flag.ContinueOnError)
	fs.SetOutput(stderr)
	model := fs.String("model", "", "model file")
	config := fs.String("config", "", "model manifest")
	cutoff := fs.Float64("cutoff", 0, "probability cutoff override")
	window := fs.Int("window", 0, "sliding window size override")
	vad := fs.String("vad", "", "VAD model file")
	vadConfig := fs.String("vad-config", "", "VAD manifest")
	raw := fs.Bool("raw", false, "raw PCM on stdin")
	features := fs.String("features", "", "replay a testdata/frontend JSON file")
	verbose := fs.Bool("verbose", false, "print detections blocked by the VAD")
	probabilities := fs.Bool("probabilities", false, "print the raw model output after every invocation")
	if err := fs.Parse(args); err != nil {
		return exitUsage
	}
	if *model == "" {
		warn(stderr, "wakeword: --model is required")
		return usage(stderr)
	}
	if *config == "" {
		*config = defaultConfigPath(*model)
	}
	if *vad != "" && *vadConfig == "" {
		*vadConfig = defaultConfigPath(*vad)
	}
	path := fs.Arg(0)
	if fs.NArg() > 1 {
		warn(stderr, "wakeword: too many arguments")
		return usage(stderr)
	}
	inputs := 0
	for _, set := range []bool{path != "", *raw, *features != ""} {
		if set {
			inputs++
		}
	}
	if inputs != 1 {
		warn(stderr, "wakeword: give exactly one of a WAV file, --raw or --features")
		return usage(stderr)
	}

	var onProb func(frame int, p uint8)
	out := newOutWriter(stdout)
	if *probabilities {
		onProb = func(frame int, p uint8) {
			out.printf("t=%.3fs p=%.3f\n", float64(frame*frontend.DefaultConfig().WindowStepMS)/1000, float64(p)/255)
		}
	}
	det, err := loadDetector(*model, *config, *vad, *vadConfig, *cutoff, *window, onProb)
	if err != nil {
		return fail(stderr, err)
	}
	report := func(ev wakeword.Event) {
		if ev.BlockedByVAD && !*verbose {
			return
		}
		what := "detected"
		if ev.BlockedByVAD {
			what = "blocked by VAD"
		}
		out.printf("t=%.3fs %s %q avg=%.3f max=%.3f\n", float64(ev.Sample)/wav.SampleRate,
			what, ev.WakeWord, float64(ev.AverageProbability)/255, float64(ev.MaxProbability)/255)
		if err := out.flush(); err != nil { // events are rare, deliver them as they happen
			warn(stderr, "wakeword: %v", err)
		}
	}

	if *features != "" {
		err = replayFeatures(det, *features, report)
	} else {
		src, serr := audioSource(path, *raw, stdin)
		if serr != nil {
			err = serr
		} else {
			err = src(func(pcm []int16) error {
				evs, err := det.Feed(pcm)
				for _, ev := range evs {
					report(ev)
				}
				return err
			})
		}
	}
	if err == nil {
		err = out.flush()
	}
	if err != nil {
		return fail(stderr, err)
	}
	return exitOK
}

func loadDetector(model, config, vad, vadConfig string, cutoff float64, window int,
	onProb func(frame int, p uint8),
) (*wakeword.Detector, error) {
	modelBytes, err := os.ReadFile(model)
	if err != nil {
		return nil, err
	}
	configBytes, err := os.ReadFile(config)
	if err != nil {
		return nil, err
	}
	opts := wakeword.Options{ProbabilityCutoff: cutoff, SlidingWindowSize: window, OnProbability: onProb}
	if vad != "" {
		if opts.VADModel, err = os.ReadFile(vad); err != nil {
			return nil, err
		}
		if opts.VADConfig, err = os.ReadFile(vadConfig); err != nil {
			return nil, err
		}
	}
	return wakeword.NewDetector(modelBytes, configBytes, opts)
}

// goldenFeatures is the subset of a testdata/frontend JSON file needed for a replay.
type goldenFeatures struct {
	Signal string `json:"signal"`
	Chunks []struct {
		FeaturesInt8 []int8 `json:"features_int8"`
	} `json:"chunks"`
}

func replayFeatures(det *wakeword.Detector, path string, report func(wakeword.Event)) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var g goldenFeatures
	if err := json.Unmarshal(data, &g); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	for i, c := range g.Chunks {
		if c.FeaturesInt8 == nil {
			continue
		}
		ev, ok, err := det.FeedFeatures(c.FeaturesInt8)
		if err != nil {
			return fmt.Errorf("%s: chunk %d: %w", path, i, err)
		}
		if ok {
			report(ev)
		}
	}
	return nil
}

func runFeatures(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("features", flag.ContinueOnError)
	fs.SetOutput(stderr)
	raw := fs.Bool("raw", false, "raw PCM on stdin")
	if err := fs.Parse(args); err != nil {
		return exitUsage
	}
	if fs.NArg() > 1 {
		warn(stderr, "wakeword: too many arguments")
		return usage(stderr)
	}
	src, err := audioSource(fs.Arg(0), *raw, stdin)
	if err != nil {
		warn(stderr, "wakeword: %v", err)
		if fs.Arg(0) == "" && !*raw {
			return usage(stderr)
		}
		return exitRuntime
	}
	fe := frontend.New()
	int8s := make([]int8, fe.NumChannels())
	out := newOutWriter(stdout)
	frame := 0
	err = src(func(pcm []int16) error {
		for len(pcm) > 0 {
			features, n := fe.Process(pcm)
			pcm = pcm[n:]
			if features == nil {
				continue
			}
			frontend.QuantizeFeatures(int8s, features)
			t := float64(fe.WindowSize()+frame*fe.WindowStep()) / wav.SampleRate
			out.printf("%d t=%.3fs u16:", frame, t)
			for _, v := range features {
				out.printf(" %d", v)
			}
			out.printf(" | int8:")
			for _, v := range int8s {
				out.printf(" %d", v)
			}
			out.printf("\n")
			frame++
		}
		return out.err
	})
	if err == nil {
		err = out.flush()
	}
	if err != nil {
		return fail(stderr, err)
	}
	return exitOK
}
