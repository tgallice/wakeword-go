package kernels

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tgallice/wakeword-go/internal/parity"
	"github.com/tgallice/wakeword-go/runtime"
	"github.com/tgallice/wakeword-go/tflite"
)

const (
	parityDir = "../testdata/parity"
	modelsDir = "../testdata/models/v2"
)

// phase2Ops are the operators this phase implements; the trace tests cover every instance of
// them in every trace.
var phase2Ops = map[tflite.BuiltinOperator]bool{
	tflite.BuiltinOperatorRESHAPE:         true,
	tflite.BuiltinOperatorCONCATENATION:   true,
	tflite.BuiltinOperatorSTRIDED_SLICE:   true,
	tflite.BuiltinOperatorSPLIT_V:         true,
	tflite.BuiltinOperatorVAR_HANDLE:      true,
	tflite.BuiltinOperatorREAD_VARIABLE:   true,
	tflite.BuiltinOperatorASSIGN_VARIABLE: true,
	tflite.BuiltinOperatorCALL_ONCE:       true,
}

type oracle struct {
	name  string
	file  *parity.File
	model *runtime.Model
}

// loadOracles loads every parity file with its model.
func loadOracles(t *testing.T) []oracle {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(parityDir, "*.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no parity files in %s: %v", parityDir, err)
	}
	var out []oracle
	for _, p := range paths {
		f, err := parity.Load(p)
		if err != nil {
			t.Fatalf("parity.Load: %v", err)
		}
		buf, err := os.ReadFile(filepath.Join(modelsDir, f.Model))
		if err != nil {
			t.Fatalf("read model: %v", err)
		}
		m, err := runtime.Load(buf)
		if err != nil {
			t.Fatalf("%s: Load: %v", f.Model, err)
		}
		out = append(out, oracle{name: strings.TrimSuffix(f.Model, ".tflite"), file: f, model: m})
	}
	return out
}

// assignFor returns the ASSIGN_VARIABLE operator of the main subgraph writing variable vi.
func assignFor(m *runtime.Model, vi int) *runtime.Operator {
	for i := range m.Main().Operators {
		op := &m.Main().Operators[i]
		if op.Code == tflite.BuiltinOperatorASSIGN_VARIABLE && op.Variable == vi {
			return op
		}
	}
	return nil
}

// setInputsFromTrace loads every dynamic input of op from the trace. Constants are already in
// place and resource tensors carry no data.
func setInputsFromTrace(t *testing.T, it *runtime.Interpreter, op *runtime.Operator, tr *parity.Trace) {
	t.Helper()
	for k, idx := range op.Inputs {
		in := it.In(op, k)
		if in.Info.IsConst() || in.Info.DType == runtime.Resource {
			continue
		}
		tt, ok := tr.Tensors[idx]
		if !ok {
			t.Fatalf("operator %d (%s): input tensor %d missing from trace", op.Index, op.Name, idx)
		}
		if err := in.SetBytes(tt.Data); err != nil {
			t.Fatalf("operator %d (%s): %v", op.Index, op.Name, err)
		}
	}
}

// TestPhase2OperatorsAgainstTraces runs every copy and state operator of every trace of every
// model in isolation and compares its outputs byte for byte with the oracle.
func TestPhase2OperatorsAgainstTraces(t *testing.T) {
	counts := map[string]int{}
	for _, o := range loadOracles(t) {
		t.Run(o.name, func(t *testing.T) {
			it, err := o.model.NewInterpreter()
			if err != nil {
				t.Fatal(err)
			}
			main := o.model.Main()
			checked := 0
			for ti := range o.file.Traces {
				tr := &o.file.Traces[ti]
				prev := o.file.Previous(tr)
				for oi := range main.Operators {
					op := &main.Operators[oi]
					if !phase2Ops[op.Code] {
						continue
					}
					where := func(what string, idx int) string {
						return strings.Join([]string{
							o.name, "sequence", itoa(tr.Sequence), "step", itoa(tr.Step),
							"operator", itoa(oi), op.Name, what, itoa(idx),
						}, " ")
					}
					switch op.Code {
					case tflite.BuiltinOperatorREAD_VARIABLE:
						// The variable holds what the previous step assigned, or the init
						// constant on the first step of a sequence.
						state := it.Variable(op.Variable)
						switch {
						case tr.Step == 0:
							copy(state, o.model.Variables[op.Variable].InitConst)
						case prev != nil:
							assign := assignFor(o.model, op.Variable)
							if assign == nil {
								t.Fatalf("no ASSIGN_VARIABLE for variable %d", op.Variable)
							}
							copy(state, prev.Tensors[assign.Inputs[1]].Data)
						default:
							continue
						}
						if err := it.RunOperator(0, oi); err != nil {
							t.Fatal(where("", -1), err)
						}
						want := tr.Tensors[op.Outputs[0]].Data
						if !bytes.Equal(it.Out(op, 0).Data, want) {
							t.Errorf("%s: output differs from oracle", where("tensor", op.Outputs[0]))
						}
					case tflite.BuiltinOperatorASSIGN_VARIABLE:
						setInputsFromTrace(t, it, op, tr)
						fill(it.Variable(op.Variable), 0x55)
						if err := it.RunOperator(0, oi); err != nil {
							t.Fatal(where("", -1), err)
						}
						if !bytes.Equal(it.Variable(op.Variable), tr.Tensors[op.Inputs[1]].Data) {
							t.Errorf("%s: variable differs from the assigned tensor", where("variable", op.Variable))
						}
					default:
						setInputsFromTrace(t, it, op, tr)
						for k := range op.Outputs {
							if d := it.Out(op, k).Data; d != nil {
								fill(d, 0x55)
							}
						}
						if err := it.RunOperator(0, oi); err != nil {
							t.Fatal(where("", -1), err)
						}
						for k, idx := range op.Outputs {
							out := it.Out(op, k)
							if out.Info.DType == runtime.Resource {
								continue
							}
							want, ok := tr.Tensors[idx]
							if !ok {
								t.Fatalf("%s: output missing from trace", where("tensor", idx))
							}
							if !bytes.Equal(out.Data, want.Data) {
								t.Errorf("%s: output differs from oracle", where("tensor", idx))
							}
						}
					}
					counts[op.Name]++
					checked++
				}
			}
			// Every phase 2 operator of the main subgraph must have been checked in every trace.
			expected := 0
			for _, op := range main.Operators {
				if phase2Ops[op.Code] {
					expected++
				}
			}
			// READ_VARIABLE at a step whose predecessor is not traced is skipped; that never
			// happens with the generated traces (steps are consecutive within a sequence).
			if checked != expected*len(o.file.Traces) {
				t.Errorf("checked %d operator instances, expected %d x %d traces", checked, expected, len(o.file.Traces))
			}
		})
	}
	for _, name := range []string{"RESHAPE", "CONCATENATION", "STRIDED_SLICE", "SPLIT_V", "VAR_HANDLE", "READ_VARIABLE", "ASSIGN_VARIABLE", "CALL_ONCE"} {
		if counts[name] == 0 {
			t.Errorf("%s: no instance checked", name)
		}
		t.Logf("%-16s %6d instances checked", name, counts[name])
	}
}

// TestResetAndInvokeState checks CALL_ONCE and Reset on real models: the first Invoke runs
// the init subgraph, then stops at the first arithmetic operator; the state operators before
// it produce what the oracle recorded at step 0.
func TestResetAndInvokeState(t *testing.T) {
	for _, o := range loadOracles(t) {
		t.Run(o.name, func(t *testing.T) {
			it, err := o.model.NewInterpreter()
			if err != nil {
				t.Fatal(err)
			}
			tr := &o.file.Traces[0]
			if tr.Sequence != 0 || tr.Step != 0 {
				t.Fatalf("first trace is sequence %d step %d", tr.Sequence, tr.Step)
			}
			if err := it.Input(0).SetBytes(tr.Tensors[o.file.Input.Index].Data); err != nil {
				t.Fatal(err)
			}
			err = it.Invoke()
			if err == nil {
				t.Fatal("Invoke succeeded without arithmetic kernels")
			}
			if !strings.Contains(err.Error(), "CONV_2D") {
				t.Fatalf("Invoke stopped elsewhere than the first CONV_2D: %v", err)
			}
			if !it.Initialized() {
				t.Fatal("init subgraph did not run")
			}
			// Only the READ_VARIABLE operators scheduled before the first arithmetic operator
			// have run; in alexa and hey_mycroft some reads are interleaved with convolutions.
			for _, op := range o.model.Main().Operators {
				if op.Code == tflite.BuiltinOperatorCONV_2D {
					break
				}
				if op.Code != tflite.BuiltinOperatorREAD_VARIABLE {
					continue
				}
				if !bytes.Equal(it.Out(&op, 0).Data, tr.Tensors[op.Outputs[0]].Data) {
					t.Errorf("operator %d: READ_VARIABLE after init differs from oracle step 0", op.Index)
				}
			}
			for vi := range o.model.Variables {
				fill(it.Variable(vi), 0x11)
			}
			if err := it.Reset(); err != nil {
				t.Fatal(err)
			}
			for vi, v := range o.model.Variables {
				for i, b := range it.Variable(vi) {
					if b != 0x80 {
						t.Errorf("variable %d (%s): byte %d is %#x after Reset, want 0x80", vi, v.SharedName, i, b)
						break
					}
				}
			}
		})
	}
}

// TestCopyKernelsDoNotAllocate runs each phase 2 operator of okay_nabu (the model with every
// operator kind) and checks the execution path allocates nothing.
func TestCopyKernelsDoNotAllocate(t *testing.T) {
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
		if !phase2Ops[op.Code] || seen[op.Name] {
			continue
		}
		seen[op.Name] = true
		if err := it.RunOperator(0, oi); err != nil {
			t.Fatalf("operator %d (%s): %v", oi, op.Name, err)
		}
		allocs := testing.AllocsPerRun(100, func() {
			if err := it.RunOperator(0, oi); err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Errorf("operator %d (%s): %v allocations per run", oi, op.Name, allocs)
		}
	}
	allocs := testing.AllocsPerRun(20, func() {
		if err := it.Reset(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Errorf("Reset: %v allocations per run", allocs)
	}
	if len(seen) != len(phase2Ops) {
		t.Errorf("only %d of %d operator kinds present in okay_nabu", len(seen), len(phase2Ops))
	}
}

func fill(b []byte, v byte) {
	for i := range b {
		b[i] = v
	}
}

func itoa(i int) string {
	if i < 0 {
		return "-" + itoa(-i)
	}
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}
