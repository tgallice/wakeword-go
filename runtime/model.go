package runtime

import (
	"errors"
	"fmt"
	"math"

	"github.com/tgallice/wakeword-go/tflite"
)

// Limits guarding the loader against corrupted files. They are far above anything a
// microWakeWord model needs (a few hundred tensors, a few hundred KiB).
const (
	maxSubgraphs     = 16
	maxTensors       = 1 << 16
	maxOperators     = 1 << 16
	maxDims          = 8
	maxTensorBytes   = 64 << 20
	maxTotalBytes    = 256 << 20
	tfliteModelMagic = "TFL3"
)

// ErrNotTFLite is returned when the buffer does not carry the TFLite file identifier.
var ErrNotTFLite = errors.New("not a TFLite model (missing TFL3 identifier)")

// TensorInfo is the static description of a tensor: what the file says about it.
type TensorInfo struct {
	Index int
	Name  string
	Shape []int
	DType DType
	// Quant is nil for tensors without quantization parameters (int32 shape parameters,
	// resource handles). Every int8 and uint8 tensor has one.
	Quant *Quantization
	// Const holds the constant data of the tensor (weights, biases, parameters), aliased into
	// the model buffer. It is nil for dynamic tensors, which the interpreter allocates.
	Const []byte
	// NumElements is the product of the shape dimensions.
	NumElements int
	// ByteSize is NumElements times the element size, 0 for resource tensors.
	ByteSize int
}

// IsConst reports whether the tensor has constant data in the file.
func (t *TensorInfo) IsConst() bool { return t.Const != nil }

// Operator is one node of a subgraph.
type Operator struct {
	// Subgraph is the index of the enclosing subgraph, Index the position in it.
	Subgraph int
	Index    int
	Code     tflite.BuiltinOperator
	Name     string
	Version  int
	// Inputs and Outputs are tensor indices in the enclosing subgraph. An input of -1 marks an
	// absent optional input.
	Inputs  []int
	Outputs []int
	// Options is the typed options struct for the operator (for example *Conv2DOptions), or nil
	// for operators whose options the runtime does not use.
	Options any
	// Variable is set for VAR_HANDLE, READ_VARIABLE and ASSIGN_VARIABLE: the index into
	// Model.Variables of the state they touch.
	Variable int
}

// Subgraph is one graph of the model. Index 0 is the main graph; the init graph referenced by
// CALL_ONCE contains only VAR_HANDLE and ASSIGN_VARIABLE operators.
type Subgraph struct {
	Index     int
	Name      string
	Tensors   []TensorInfo
	Operators []Operator
	Inputs    []int
	Outputs   []int
}

// Variable is a persistent state buffer shared by VAR_HANDLE, READ_VARIABLE and ASSIGN_VARIABLE
// operators across subgraphs. Its shape and dtype come from the tensors assigned to it.
type Variable struct {
	Index      int
	Container  string
	SharedName string
	Shape      []int
	DType      DType
	ByteSize   int
	Quant      *Quantization
	// InitConst is the constant assigned by the init subgraph, nil if the variable is not
	// initialized there.
	InitConst []byte
}

// Model is a loaded and validated TFLite model.
type Model struct {
	Version     uint32
	Description string
	Subgraphs   []Subgraph
	Variables   []Variable
	// InitSubgraph is the subgraph run once by CALL_ONCE, or -1 when the main graph has no
	// CALL_ONCE operator.
	InitSubgraph int

	buf []byte
}

// Main returns the main subgraph.
func (m *Model) Main() *Subgraph { return &m.Subgraphs[0] }

// Load parses and validates a .tflite file held in memory. The returned model aliases buf for
// its constant tensors; buf must not be modified afterwards.
func Load(buf []byte) (m *Model, err error) {
	// The generated FlatBuffers accessors do not verify offsets, so a corrupted file can make
	// them index out of range. Turn that into an error instead of a crash.
	defer func() {
		if r := recover(); r != nil {
			m = nil
			err = fmt.Errorf("corrupted model: %v", r)
		}
	}()
	if len(buf) < 8 || !tflite.ModelBufferHasIdentifier(buf) {
		return nil, ErrNotTFLite
	}
	root := tflite.GetRootAsModel(buf, 0)
	m = &Model{
		Version:      root.Version(),
		Description:  string(root.Description()),
		InitSubgraph: -1,
		buf:          buf,
	}
	if m.Version != 3 {
		return nil, fmt.Errorf("unsupported model version %d", m.Version)
	}
	n := root.SubgraphsLength()
	if n < 1 || n > maxSubgraphs {
		return nil, fmt.Errorf("model has %d subgraphs, expected between 1 and %d", n, maxSubgraphs)
	}
	codes, err := loadOperatorCodes(root)
	if err != nil {
		return nil, err
	}
	total := 0
	for i := range n {
		var sg tflite.SubGraph
		if !root.Subgraphs(&sg, i) {
			return nil, fmt.Errorf("subgraph %d: unreadable", i)
		}
		loaded, err := loadSubgraph(root, &sg, i, codes)
		if err != nil {
			return nil, fmt.Errorf("subgraph %d (%s): %w", i, loaded.Name, err)
		}
		for j := range loaded.Tensors {
			total += loaded.Tensors[j].ByteSize
		}
		if total > maxTotalBytes {
			return nil, fmt.Errorf("model tensors exceed %d bytes", maxTotalBytes)
		}
		m.Subgraphs = append(m.Subgraphs, loaded)
	}
	if err := m.resolveVariables(); err != nil {
		return nil, err
	}
	if err := m.validate(); err != nil {
		return nil, err
	}
	return m, nil
}

type operatorCode struct {
	code    tflite.BuiltinOperator
	name    string
	version int32
	custom  string
}

func loadOperatorCodes(root *tflite.Model) ([]operatorCode, error) {
	n := root.OperatorCodesLength()
	if n > maxOperators {
		return nil, fmt.Errorf("model has %d operator codes", n)
	}
	codes := make([]operatorCode, n)
	for i := range n {
		var oc tflite.OperatorCode
		if !root.OperatorCodes(&oc, i) {
			return nil, fmt.Errorf("operator code %d: unreadable", i)
		}
		code := resolveBuiltinCode(&oc)
		codes[i] = operatorCode{code: code, name: builtinName(code), version: oc.Version()}
		if code == tflite.BuiltinOperatorCUSTOM {
			codes[i].custom = string(oc.CustomCode())
			codes[i].name = "CUSTOM:" + codes[i].custom
		}
	}
	return codes, nil
}

func loadSubgraph(root *tflite.Model, sg *tflite.SubGraph, index int, codes []operatorCode) (Subgraph, error) {
	out := Subgraph{Index: index, Name: string(sg.Name())}
	nt := sg.TensorsLength()
	if nt > maxTensors {
		return out, fmt.Errorf("%d tensors", nt)
	}
	out.Tensors = make([]TensorInfo, nt)
	for i := range nt {
		var t tflite.Tensor
		if !sg.Tensors(&t, i) {
			return out, fmt.Errorf("tensor %d: unreadable", i)
		}
		info, err := loadTensor(root, &t, i)
		if err != nil {
			return out, fmt.Errorf("tensor %d (%s): %w", i, string(t.Name()), err)
		}
		out.Tensors[i] = info
	}
	for i := range sg.InputsLength() {
		idx := int(sg.Inputs(i))
		if idx < 0 || idx >= nt {
			return out, fmt.Errorf("subgraph input %d: tensor index %d out of range", i, idx)
		}
		out.Inputs = append(out.Inputs, idx)
	}
	for i := range sg.OutputsLength() {
		idx := int(sg.Outputs(i))
		if idx < 0 || idx >= nt {
			return out, fmt.Errorf("subgraph output %d: tensor index %d out of range", i, idx)
		}
		out.Outputs = append(out.Outputs, idx)
	}
	no := sg.OperatorsLength()
	if no > maxOperators {
		return out, fmt.Errorf("%d operators", no)
	}
	out.Operators = make([]Operator, no)
	for i := range no {
		var op tflite.Operator
		if !sg.Operators(&op, i) {
			return out, fmt.Errorf("operator %d: unreadable", i)
		}
		ci := int(op.OpcodeIndex())
		if ci < 0 || ci >= len(codes) {
			return out, fmt.Errorf("operator %d: opcode index %d out of range", i, ci)
		}
		oc := codes[ci]
		o := Operator{Subgraph: index, Index: i, Code: oc.code, Name: oc.name, Version: int(oc.version), Variable: -1}
		for k := range op.InputsLength() {
			idx := int(op.Inputs(k))
			if idx < -1 || idx >= nt {
				return out, fmt.Errorf("operator %d (%s): input %d: tensor index %d out of range", i, o.Name, k, idx)
			}
			o.Inputs = append(o.Inputs, idx)
		}
		for k := range op.OutputsLength() {
			idx := int(op.Outputs(k))
			if idx < 0 || idx >= nt {
				return out, fmt.Errorf("operator %d (%s): output %d: tensor index %d out of range", i, o.Name, k, idx)
			}
			o.Outputs = append(o.Outputs, idx)
		}
		// Options are only decoded for supported operators; unsupported ones are reported by
		// validate with the full list rather than failing here on the first one.
		if _, ok := supportedOps[o.Code]; ok {
			opts, err := parseOptions(o.Code, &op)
			if err != nil {
				return out, fmt.Errorf("operator %d (%s): %w", i, o.Name, err)
			}
			o.Options = opts
		}
		out.Operators[i] = o
	}
	return out, nil
}

func loadTensor(root *tflite.Model, t *tflite.Tensor, index int) (TensorInfo, error) {
	info := TensorInfo{Index: index, Name: string(t.Name())}
	dt, ok := dtypeFromTFLite(t.Type())
	if !ok {
		return info, fmt.Errorf("unsupported dtype %s", t.Type().String())
	}
	info.DType = dt
	nd := t.ShapeLength()
	if nd > maxDims {
		return info, fmt.Errorf("%d dimensions", nd)
	}
	info.Shape = make([]int, nd)
	info.NumElements = 1
	for i := range nd {
		d := int(t.Shape(i))
		if d < 0 {
			return info, fmt.Errorf("negative dimension %d", d)
		}
		info.Shape[i] = d
		if d != 0 && info.NumElements > maxTensorBytes/d {
			return info, fmt.Errorf("tensor too large")
		}
		info.NumElements *= d
	}
	info.ByteSize = info.NumElements * dt.Size()
	if info.ByteSize > maxTensorBytes {
		return info, fmt.Errorf("tensor too large (%d bytes)", info.ByteSize)
	}
	var q tflite.QuantizationParameters
	if t.Quantization(&q) != nil && q.ScaleLength() > 0 {
		quant := &Quantization{QuantizedDimension: int(q.QuantizedDimension())}
		if q.ZeroPointLength() != q.ScaleLength() {
			return info, fmt.Errorf("%d scales but %d zero points", q.ScaleLength(), q.ZeroPointLength())
		}
		if q.DetailsType() != tflite.QuantizationDetailsNONE {
			return info, fmt.Errorf("unsupported quantization details %s", q.DetailsType().String())
		}
		quant.Scales = make([]float32, q.ScaleLength())
		quant.ZeroPoints = make([]int32, q.ScaleLength())
		for i := range quant.Scales {
			s := q.Scale(i)
			if !(s > 0) || math.IsInf(float64(s), 0) {
				return info, fmt.Errorf("invalid scale %v", s)
			}
			zp := q.ZeroPoint(i)
			if zp < math.MinInt32 || zp > math.MaxInt32 {
				return info, fmt.Errorf("zero point %d out of int32 range", zp)
			}
			quant.Scales[i] = s
			quant.ZeroPoints[i] = int32(zp)
		}
		info.Quant = quant
	}
	bi := int(t.Buffer())
	if bi < 0 || bi >= root.BuffersLength() {
		return info, fmt.Errorf("buffer index %d out of range", bi)
	}
	var b tflite.Buffer
	if !root.Buffers(&b, bi) {
		return info, fmt.Errorf("buffer %d: unreadable", bi)
	}
	if b.Offset() != 0 || b.Size() != 0 {
		return info, fmt.Errorf("buffer %d: out-of-line buffer data is not supported", bi)
	}
	if b.DataLength() > 0 {
		data := b.DataBytes()
		if len(data) != info.ByteSize {
			return info, fmt.Errorf("buffer %d has %d bytes, tensor needs %d", bi, len(data), info.ByteSize)
		}
		info.Const = data
	}
	return info, nil
}

// resolveVariables collects the state variables referenced by VAR_HANDLE operators in all
// subgraphs and links READ_VARIABLE and ASSIGN_VARIABLE operators to them through the resource
// tensor they consume.
func (m *Model) resolveVariables() error {
	byKey := map[[2]string]int{}
	for si := range m.Subgraphs {
		sg := &m.Subgraphs[si]
		// resource tensor index -> variable index, within this subgraph
		handles := map[int]int{}
		for oi := range sg.Operators {
			op := &sg.Operators[oi]
			switch op.Code {
			case tflite.BuiltinOperatorVAR_HANDLE:
				opts, _ := op.Options.(*VarHandleOptions)
				if opts == nil || len(op.Outputs) != 1 {
					return fmt.Errorf("subgraph %d: operator %d (VAR_HANDLE): malformed", si, oi)
				}
				if sg.Tensors[op.Outputs[0]].DType != Resource {
					return fmt.Errorf("subgraph %d: operator %d (VAR_HANDLE): output is not a resource tensor", si, oi)
				}
				key := [2]string{opts.Container, opts.SharedName}
				vi, ok := byKey[key]
				if !ok {
					vi = len(m.Variables)
					byKey[key] = vi
					m.Variables = append(m.Variables, Variable{Index: vi, Container: opts.Container, SharedName: opts.SharedName})
				}
				handles[op.Outputs[0]] = vi
				op.Variable = vi
			case tflite.BuiltinOperatorREAD_VARIABLE, tflite.BuiltinOperatorASSIGN_VARIABLE:
				if len(op.Inputs) < 1 {
					return fmt.Errorf("subgraph %d: operator %d (%s): no inputs", si, oi, op.Name)
				}
				vi, ok := handles[op.Inputs[0]]
				if !ok {
					return fmt.Errorf("subgraph %d: operator %d (%s): input 0 is not a VAR_HANDLE output", si, oi, op.Name)
				}
				op.Variable = vi
				var ti int
				switch op.Code {
				case tflite.BuiltinOperatorREAD_VARIABLE:
					if len(op.Outputs) != 1 {
						return fmt.Errorf("subgraph %d: operator %d (READ_VARIABLE): expected 1 output", si, oi)
					}
					ti = op.Outputs[0]
				default:
					if len(op.Inputs) != 2 {
						return fmt.Errorf("subgraph %d: operator %d (ASSIGN_VARIABLE): expected 2 inputs", si, oi)
					}
					ti = op.Inputs[1]
				}
				if err := m.Variables[vi].bind(&sg.Tensors[ti]); err != nil {
					return fmt.Errorf("subgraph %d: operator %d (%s): %w", si, oi, op.Name, err)
				}
				if op.Code == tflite.BuiltinOperatorASSIGN_VARIABLE && si != 0 && sg.Tensors[ti].IsConst() {
					m.Variables[vi].InitConst = sg.Tensors[ti].Const
				}
			}
		}
	}
	return nil
}

// bind records the shape, dtype and quantization of a variable from a tensor read from it or
// assigned to it, and checks consistency with previous bindings.
func (v *Variable) bind(t *TensorInfo) error {
	if t.DType == Resource {
		return fmt.Errorf("variable %q bound to a resource tensor", v.SharedName)
	}
	if v.DType == 0 {
		v.DType = t.DType
		v.Shape = append([]int(nil), t.Shape...)
		v.ByteSize = t.ByteSize
		v.Quant = t.Quant
		return nil
	}
	if v.DType != t.DType || v.ByteSize != t.ByteSize || !equalShape(v.Shape, t.Shape) {
		return fmt.Errorf("variable %q: tensor %s has shape %v %s, variable is %v %s",
			v.SharedName, t.Name, t.Shape, t.DType, v.Shape, v.DType)
	}
	if !v.Quant.Equal(t.Quant) {
		return fmt.Errorf("variable %q: tensor %s quantization differs from the variable's", v.SharedName, t.Name)
	}
	return nil
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// NumElements returns the number of elements of the variable's buffer.
func (v *Variable) NumElements() int {
	n := 1
	for _, d := range v.Shape {
		n *= d
	}
	return n
}
