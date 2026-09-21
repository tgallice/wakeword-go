package kernels

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/tgallice/wakeword-go/internal/parity"
	"github.com/tgallice/wakeword-go/runtime"
	"github.com/tgallice/wakeword-go/tflite"
)

// phase4Ops are the two operators that complete the graph: the sigmoid and the final
// requantization to uint8.
var phase4Ops = map[tflite.BuiltinOperator]bool{
	tflite.BuiltinOperatorLOGISTIC: true,
	tflite.BuiltinOperatorQUANTIZE: true,
}

// TestPhase4OperatorsAgainstTraces checks LOGISTIC and QUANTIZE against every trace.
func TestPhase4OperatorsAgainstTraces(t *testing.T) {
	checkOperatorsAgainstTraces(t, phase4Ops)
}

// traceKey identifies a trace by its position in a run.
type traceKey struct{ sequence, step int }

// traceIndex maps (sequence, step) to the traces of an oracle file.
func traceIndex(f *parity.File) map[traceKey]*parity.Trace {
	idx := make(map[traceKey]*parity.Trace, len(f.Traces))
	for i := range f.Traces {
		tr := &f.Traces[i]
		idx[traceKey{tr.Sequence, tr.Step}] = tr
	}
	return idx
}

// runSequence feeds every step of a sequence through it and compares the uint8 output at
// every step, plus every dynamic tensor whenever the oracle recorded a trace for that step.
// It returns the number of invocations performed.
func runSequence(t *testing.T, o oracle, it *runtime.Interpreter, si int, traces map[traceKey]*parity.Trace) int {
	t.Helper()
	seq := &o.file.Sequences[si]
	for k, step := range seq.Steps {
		if err := it.Input(0).SetBytes(step.Input); err != nil {
			t.Fatalf("%s sequence %s step %d: %v", o.name, seq.Name, k, err)
		}
		if err := it.Invoke(); err != nil {
			t.Fatalf("%s sequence %s step %d: %v", o.name, seq.Name, k, err)
		}
		got := int(it.Output(0).Uint8()[0])
		if got != step.Output {
			t.Fatalf("%s sequence %s (%d) step %d: expected output %d, got %d", o.name, seq.Name, si, k, step.Output, got)
		}
		tr, ok := traces[traceKey{si, k}]
		if !ok {
			continue
		}
		for idx, want := range tr.Tensors {
			tensor := it.Tensor(0, idx)
			if tensor.Info.DType == runtime.Resource {
				continue
			}
			if !bytes.Equal(tensor.Data, want.Data) {
				t.Fatalf("%s sequence %s (%d) step %d: tensor %d (%s) differs from the trace after Invoke",
					o.name, seq.Name, si, k, idx, want.Name)
			}
		}
	}
	return len(seq.Steps)
}

// TestEndToEndParity replays every sequence of every oracle file through a full Invoke and
// compares the uint8 output at every step by exact equality. Each sequence starts from a fresh
// interpreter, as the oracle did (sequence_start: fresh_interpreter). Whenever the oracle
// recorded a trace for a step, every dynamic tensor is compared too.
func TestEndToEndParity(t *testing.T) {
	total := 0
	for _, o := range loadOracles(t) {
		t.Run(o.name, func(t *testing.T) {
			traces := traceIndex(o.file)
			invocations := 0
			for si := range o.file.Sequences {
				it, err := o.model.NewInterpreter()
				if err != nil {
					t.Fatal(err)
				}
				invocations += runSequence(t, o, it, si, traces)
			}
			t.Logf("%s: %d sequences, %d invocations, no mismatch", o.name, len(o.file.Sequences), invocations)
			total += invocations
		})
	}
	t.Logf("%d invocations across all models", total)
}

// TestEndToEndParityWithReset replays the sequences of one model through a single interpreter
// with Reset between sequences: Reset must be equivalent to a fresh interpreter.
func TestEndToEndParityWithReset(t *testing.T) {
	for _, o := range loadOracles(t) {
		if o.name != "hey_jarvis" {
			continue
		}
		traces := traceIndex(o.file)
		it, err := o.model.NewInterpreter()
		if err != nil {
			t.Fatal(err)
		}
		for si := range o.file.Sequences {
			if si > 0 {
				if err := it.Reset(); err != nil {
					t.Fatal(err)
				}
			}
			runSequence(t, o, it, si, traces)
		}
		return
	}
	t.Fatal("hey_jarvis oracle not found")
}

// TestInvokeDoesNotAllocate checks that a full Invoke allocates nothing on every model, after
// the first call has run the init subgraph.
func TestInvokeDoesNotAllocate(t *testing.T) {
	for _, o := range loadOracles(t) {
		t.Run(o.name, func(t *testing.T) {
			it, err := o.model.NewInterpreter()
			if err != nil {
				t.Fatal(err)
			}
			step := o.file.Sequences[len(o.file.Sequences)-1].Steps[0]
			if err := it.Input(0).SetBytes(step.Input); err != nil {
				t.Fatal(err)
			}
			if err := it.Invoke(); err != nil {
				t.Fatal(err)
			}
			allocs := testing.AllocsPerRun(20, func() {
				if err := it.Invoke(); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Errorf("%v allocations per Invoke", allocs)
			}
			allocs = testing.AllocsPerRun(5, func() {
				if err := it.Reset(); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Errorf("%v allocations per Reset", allocs)
			}
		})
	}
}

// BenchmarkInvoke measures a full streaming invocation (three 10 ms frames) per model.
func BenchmarkInvoke(b *testing.B) {
	for _, name := range []string{"vad", "hey_jarvis", "alexa", "hey_mycroft", "okay_nabu"} {
		buf, err := os.ReadFile(filepath.Join(modelsDir, name+".tflite"))
		if err != nil {
			b.Fatal(err)
		}
		m, err := runtime.Load(buf)
		if err != nil {
			b.Fatal(err)
		}
		it, err := m.NewInterpreter()
		if err != nil {
			b.Fatal(err)
		}
		input := make([]int8, it.Input(0).Info.NumElements)
		for i := range input {
			input[i] = int8(i*7 - 128)
		}
		if err := it.Input(0).SetInt8(input); err != nil {
			b.Fatal(err)
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if err := it.Invoke(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
