package runtime

import (
	"encoding/json"
	"errors"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tgallice/wakeword-go/tflite"
)

const modelsDir = "../testdata/models"

func readModel(t *testing.T, rel string) []byte {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(modelsDir, rel))
	if err != nil {
		t.Fatalf("read model: %v", err)
	}
	return buf
}

func TestLoadV2Models(t *testing.T) {
	// Expected counts come from tools/oracle/dump_ops.py on the pinned models.
	cases := []struct {
		name      string
		operators int
		tensors   int
		variables int
		splitV    bool
	}{
		{name: "alexa", operators: 45, tensors: 70, variables: 6},
		{name: "hey_jarvis", operators: 45, tensors: 70, variables: 6},
		{name: "hey_mycroft", operators: 45, tensors: 71, variables: 6},
		{name: "okay_nabu", operators: 55, tensors: 94, variables: 6, splitV: true},
		{name: "vad", operators: 32, tensors: 53, variables: 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, err := Load(readModel(t, "v2/"+tc.name+".tflite"))
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			main := m.Main()
			if got := len(main.Operators); got != tc.operators {
				t.Errorf("operators: got %d, want %d", got, tc.operators)
			}
			if got := len(main.Tensors); got != tc.tensors {
				t.Errorf("tensors: got %d, want %d", got, tc.tensors)
			}
			if got := len(m.Variables); got != tc.variables {
				t.Errorf("variables: got %d, want %d", got, tc.variables)
			}
			if len(m.Subgraphs) != 2 || m.InitSubgraph != 1 {
				t.Errorf("subgraphs %d, init %d, want 2 and 1", len(m.Subgraphs), m.InitSubgraph)
			}
			in := main.Tensors[main.Inputs[0]]
			if !equalShape(in.Shape, []int{1, 3, 40}) || in.DType != Int8 || in.Quant == nil || in.Quant.ZeroPoints[0] != -128 {
				t.Errorf("input: shape %v dtype %s quant %+v", in.Shape, in.DType, in.Quant)
			}
			out := main.Tensors[main.Outputs[0]]
			if !equalShape(out.Shape, []int{1, 1}) || out.DType != UInt8 || out.Quant == nil || out.Quant.Scales[0] != 1.0/256 {
				t.Errorf("output: shape %v dtype %s quant %+v", out.Shape, out.DType, out.Quant)
			}
			hasSplit := false
			for _, op := range main.Operators {
				if op.Code == tflite.BuiltinOperatorSPLIT_V {
					hasSplit = true
				}
				if op.Code == tflite.BuiltinOperatorVAR_HANDLE || op.Code == tflite.BuiltinOperatorREAD_VARIABLE ||
					op.Code == tflite.BuiltinOperatorASSIGN_VARIABLE {
					if op.Variable < 0 || op.Variable >= len(m.Variables) {
						t.Errorf("operator %d (%s): variable index %d", op.Index, op.Name, op.Variable)
					}
				}
			}
			if hasSplit != tc.splitV {
				t.Errorf("SPLIT_V present: %v, want %v", hasSplit, tc.splitV)
			}
			for _, v := range m.Variables {
				if v.DType != Int8 || len(v.Shape) != 4 || v.ByteSize != v.NumElements() || v.InitConst == nil {
					t.Errorf("variable %q: dtype %s shape %v bytes %d init %v", v.SharedName, v.DType, v.Shape, v.ByteSize, v.InitConst != nil)
					continue
				}
				for _, b := range v.InitConst {
					if b != 0x80 {
						t.Errorf("variable %q: init byte %#x, want 0x80 (int8 -128)", v.SharedName, b)
						break
					}
				}
				if v.Quant == nil || v.Quant.ZeroPoints[0] != -128 {
					t.Errorf("variable %q: quantization %+v", v.SharedName, v.Quant)
				}
			}
		})
	}
}

func TestLoadRejectsV1Model(t *testing.T) {
	_, err := Load(readModel(t, "v1/okay_nabu.tflite"))
	if err == nil {
		t.Fatal("v1 model loaded, expected rejection")
	}
	msg := err.Error()
	if !strings.Contains(msg, "ADD") && !strings.Contains(msg, "MUL") {
		t.Fatalf("error does not name ADD or MUL: %v", err)
	}
	t.Logf("rejection: %v", err)
}

func TestLoadRejectsCorruptedInput(t *testing.T) {
	good := readModel(t, "v2/hey_jarvis.tflite")
	rng := rand.New(rand.NewPCG(1, 2))
	random := make([]byte, len(good))
	for i := range random {
		random[i] = byte(rng.IntN(256))
	}
	cases := map[string][]byte{
		"empty":         {},
		"short":         good[:6],
		"random":        random,
		"identifier":    good[:8],
		"truncated_16":  good[:16],
		"truncated_64":  good[:64],
		"truncated_1k":  good[:1024],
		"truncated_8k":  good[:8192],
		"truncated_-1":  good[:len(good)-1],
		"truncated_-64": good[:len(good)-64],
	}
	for i := range 8 {
		flipped := append([]byte(nil), good...)
		for range 16 {
			flipped[rng.IntN(len(flipped))] ^= byte(1 + rng.IntN(255))
		}
		cases["flipped_"+string(rune('a'+i))] = flipped
	}
	for name, buf := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Load panicked: %v", r)
				}
			}()
			m, err := Load(buf)
			if err == nil {
				// A flipped byte inside a weight buffer is undetectable and must load; anything
				// else must fail. Only sanity check the successful case.
				if !strings.HasPrefix(name, "flipped_") {
					t.Fatalf("corrupted input loaded")
				}
				if _, err := m.NewInterpreter(); err != nil {
					t.Fatalf("NewInterpreter: %v", err)
				}
				return
			}
			t.Logf("rejected: %v", err)
		})
	}
	if _, err := Load(random); !errors.Is(err, ErrNotTFLite) {
		t.Errorf("random bytes: got %v, want ErrNotTFLite", err)
	}
}

// parityDoc is the subset of testdata/parity/<model>.json the loader can be checked against.
type parityDoc struct {
	Model  string       `json:"model"`
	Input  parityTensor `json:"input"`
	Output parityTensor `json:"output"`
	// Tensors lists every non-constant tensor of the main subgraph, keyed by index.
	Tensors map[string]parityTensor `json:"tensors"`
}

type parityTensor struct {
	Index        int    `json:"index"`
	Name         string `json:"name"`
	Shape        []int  `json:"shape"`
	DType        string `json:"dtype"`
	Quantization struct {
		Scales             []float64 `json:"scales"`
		ZeroPoints         []int32   `json:"zero_points"`
		QuantizedDimension int       `json:"quantized_dimension"`
	} `json:"quantization"`
}

func TestLoadMatchesParityMetadata(t *testing.T) {
	for _, name := range []string{"alexa", "hey_jarvis", "hey_mycroft", "okay_nabu", "vad"} {
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("../testdata/parity", name+".json"))
			if err != nil {
				t.Fatalf("read parity: %v", err)
			}
			var doc parityDoc
			if err := json.Unmarshal(raw, &doc); err != nil {
				t.Fatalf("parse parity: %v", err)
			}
			m, err := Load(readModel(t, "v2/"+doc.Model))
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			main := m.Main()
			if main.Inputs[0] != doc.Input.Index || main.Outputs[0] != doc.Output.Index {
				t.Errorf("io indices: got %d/%d, want %d/%d", main.Inputs[0], main.Outputs[0], doc.Input.Index, doc.Output.Index)
			}
			checkTensor(t, &main.Tensors[doc.Input.Index], &doc.Input, true)
			checkTensor(t, &main.Tensors[doc.Output.Index], &doc.Output, true)
			dynamic := 0
			for _, ti := range main.Tensors {
				if !ti.IsConst() && ti.DType != Resource {
					dynamic++
				}
			}
			if dynamic != len(doc.Tensors) {
				t.Errorf("dynamic tensors: got %d, want %d", dynamic, len(doc.Tensors))
			}
			for key, pt := range doc.Tensors {
				var idx int
				if _, err := fmtSscan(key, &idx); err != nil {
					t.Fatalf("tensor key %q: %v", key, err)
				}
				ti := &main.Tensors[idx]
				if ti.IsConst() {
					t.Errorf("tensor %d (%s) is constant, oracle lists it as dynamic", idx, ti.Name)
				}
				checkTensor(t, ti, &pt, false)
			}
		})
	}
}

func checkTensor(t *testing.T, ti *TensorInfo, pt *parityTensor, withShape bool) {
	t.Helper()
	if ti.Name != pt.Name {
		t.Errorf("tensor %d: name %q, want %q", ti.Index, ti.Name, pt.Name)
	}
	if ti.DType.String() != strings.ToUpper(pt.DType) {
		t.Errorf("tensor %d (%s): dtype %s, want %s", ti.Index, ti.Name, ti.DType, strings.ToUpper(pt.DType))
	}
	if withShape && !equalShape(ti.Shape, pt.Shape) {
		t.Errorf("tensor %d (%s): shape %v, want %v", ti.Index, ti.Name, ti.Shape, pt.Shape)
	}
	q := pt.Quantization
	if len(q.Scales) == 0 {
		if ti.Quant != nil {
			t.Errorf("tensor %d (%s): has quantization, oracle has none", ti.Index, ti.Name)
		}
		return
	}
	if ti.Quant == nil {
		t.Errorf("tensor %d (%s): no quantization, oracle has %d scales", ti.Index, ti.Name, len(q.Scales))
		return
	}
	if len(ti.Quant.Scales) != len(q.Scales) || ti.Quant.QuantizedDimension != q.QuantizedDimension {
		t.Errorf("tensor %d (%s): %d scales dim %d, want %d scales dim %d", ti.Index, ti.Name,
			len(ti.Quant.Scales), ti.Quant.QuantizedDimension, len(q.Scales), q.QuantizedDimension)
		return
	}
	for i := range q.Scales {
		// The oracle stores float32 scales as JSON doubles; the round trip is exact.
		if float64(ti.Quant.Scales[i]) != q.Scales[i] || ti.Quant.ZeroPoints[i] != q.ZeroPoints[i] {
			t.Errorf("tensor %d (%s): scale[%d]=%v zp=%d, want %v zp=%d", ti.Index, ti.Name, i,
				ti.Quant.Scales[i], ti.Quant.ZeroPoints[i], q.Scales[i], q.ZeroPoints[i])
			return
		}
	}
}
