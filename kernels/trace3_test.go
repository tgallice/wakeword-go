package kernels

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tgallice/wakeword-go/internal/parity"
	"github.com/tgallice/wakeword-go/runtime"
	"github.com/tgallice/wakeword-go/tflite"
)

// phase3Ops are the int8 arithmetic operators of Phase 3; the trace tests cover every instance
// of them in every trace.
var phase3Ops = map[tflite.BuiltinOperator]bool{
	tflite.BuiltinOperatorCONV_2D:           true,
	tflite.BuiltinOperatorDEPTHWISE_CONV_2D: true,
	tflite.BuiltinOperatorFULLY_CONNECTED:   true,
}

// compareOutputs checks every non-resource output of op against the trace byte for byte and
// reports the first differing element of the first differing tensor.
func compareOutputs(t *testing.T, it *runtime.Interpreter, op *runtime.Operator, tr *parity.Trace, where func(what string, idx int) string) {
	t.Helper()
	for k, idx := range op.Outputs {
		out := it.Out(op, k)
		if out.Info.DType == runtime.Resource {
			continue
		}
		want, ok := tr.Tensors[idx]
		if !ok {
			t.Fatalf("%s: output missing from trace", where("tensor", idx))
		}
		if len(out.Data) != len(want.Data) {
			t.Errorf("%s: %d bytes, oracle has %d", where("tensor", idx), len(out.Data), len(want.Data))
			continue
		}
		for i := range out.Data {
			if out.Data[i] != want.Data[i] {
				t.Errorf("%s: element %d differs: expected %d, got %d", where("tensor", idx), i, int8(want.Data[i]), int8(out.Data[i]))
				break
			}
		}
	}
}

// checkOperatorsAgainstTraces runs every operator of the given kinds, in every trace of every
// model, in isolation, and compares its output byte for byte with the oracle. It reports the
// number of instances checked per operator kind.
func checkOperatorsAgainstTraces(t *testing.T, ops map[tflite.BuiltinOperator]bool) {
	t.Helper()
	counts := map[string]int{}
	for _, o := range loadOracles(t) {
		t.Run(o.name, func(t *testing.T) {
			it, err := o.model.NewInterpreter()
			if err != nil {
				t.Fatal(err)
			}
			main := o.model.Main()
			checked, expected := 0, 0
			for _, op := range main.Operators {
				if ops[op.Code] {
					expected++
				}
			}
			for ti := range o.file.Traces {
				tr := &o.file.Traces[ti]
				for oi := range main.Operators {
					op := &main.Operators[oi]
					if !ops[op.Code] {
						continue
					}
					where := func(what string, idx int) string {
						return strings.Join([]string{
							o.name, "sequence", itoa(tr.Sequence), "step", itoa(tr.Step),
							"operator", itoa(oi), op.Name, what, itoa(idx),
						}, " ")
					}
					setInputsFromTrace(t, it, op, tr)
					fill(it.Out(op, 0).Data, 0x55)
					if err := it.RunOperator(0, oi); err != nil {
						t.Fatal(where("", -1), err)
					}
					compareOutputs(t, it, op, tr, where)
					counts[op.Name]++
					checked++
				}
			}
			if checked != expected*len(o.file.Traces) {
				t.Errorf("checked %d operator instances, expected %d x %d traces", checked, expected, len(o.file.Traces))
			}
		})
	}
	for code := range ops {
		name := tflite.EnumNamesBuiltinOperator[code]
		if counts[name] == 0 {
			t.Errorf("%s: no instance checked", name)
		}
		t.Logf("%-18s %6d instances checked", name, counts[name])
	}
}

// TestPhase3OperatorsAgainstTraces checks CONV_2D, DEPTHWISE_CONV_2D and FULLY_CONNECTED.
func TestPhase3OperatorsAgainstTraces(t *testing.T) {
	checkOperatorsAgainstTraces(t, phase3Ops)
}

// TestArithmeticKernelsDoNotAllocate runs each Phase 3 operator kind of okay_nabu and checks
// the execution path allocates nothing.
func TestArithmeticKernelsDoNotAllocate(t *testing.T) {
	buf, err := os.ReadFile(filepath.Join(modelsDir, "okay_nabu.tflite"))
	if err != nil {
		t.Fatal(err)
	}
	m, err := runtime.Load(buf)
	if err != nil {
		t.Fatal(err)
	}
	it, err := m.NewInterpreter()
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for oi, op := range m.Main().Operators {
		if !phase3Ops[op.Code] || seen[op.Name] {
			continue
		}
		seen[op.Name] = true
		allocs := testing.AllocsPerRun(50, func() {
			if err := it.RunOperator(0, oi); err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Errorf("operator %d (%s): %v allocations per run", oi, op.Name, allocs)
		}
	}
	if len(seen) != len(phase3Ops) {
		t.Errorf("only %d of %d operator kinds present in okay_nabu", len(seen), len(phase3Ops))
	}
}
