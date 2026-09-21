package runtime

import (
	"errors"
	"fmt"
	"unsafe"
)

// ErrNoKernel is returned by Invoke and RunOperator when an operator has no registered
// kernel. Every supported operator gets one once the kernels package is imported.
var ErrNoKernel = errors.New("runtime: no kernel registered")

// Tensor is a tensor with storage: either a constant aliased from the model or a buffer owned
// by the interpreter. Resource tensors have no Data.
type Tensor struct {
	Info *TensorInfo
	Data []byte
}

// Int8 returns the data as int8 elements without copying. It panics on a non-int8 tensor.
func (t *Tensor) Int8() []int8 {
	t.mustBe(Int8)
	if len(t.Data) == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&t.Data[0])), len(t.Data))
}

// Uint8 returns the data as uint8 elements. It panics on a non-uint8 tensor.
func (t *Tensor) Uint8() []uint8 {
	t.mustBe(UInt8)
	return t.Data
}

// Int32 returns the data as little-endian int32 elements without copying. It panics on a
// non-int32 tensor.
func (t *Tensor) Int32() []int32 {
	t.mustBe(Int32)
	if len(t.Data) == 0 {
		return nil
	}
	return unsafe.Slice((*int32)(unsafe.Pointer(&t.Data[0])), len(t.Data)/4)
}

// SetInt8 copies values into an int8 tensor. It fails if the length does not match.
func (t *Tensor) SetInt8(values []int8) error {
	if t.Info.DType != Int8 {
		return fmt.Errorf("tensor %s is %s, not INT8", t.Info.Name, t.Info.DType)
	}
	if len(values) != t.Info.NumElements {
		return fmt.Errorf("tensor %s has %d elements, got %d", t.Info.Name, t.Info.NumElements, len(values))
	}
	copy(t.Int8(), values)
	return nil
}

// SetBytes copies raw bytes into a dynamic tensor. It fails if the length does not match or
// the tensor is a constant.
func (t *Tensor) SetBytes(b []byte) error {
	if t.Info.IsConst() {
		return fmt.Errorf("tensor %s is constant", t.Info.Name)
	}
	if len(b) != len(t.Data) {
		return fmt.Errorf("tensor %s has %d bytes, got %d", t.Info.Name, len(t.Data), len(b))
	}
	copy(t.Data, b)
	return nil
}

func (t *Tensor) mustBe(d DType) {
	if t.Info.DType != d {
		panic(fmt.Sprintf("runtime: tensor %s is %s, not %s", t.Info.Name, t.Info.DType, d))
	}
}

// Interpreter executes a Model. All buffers are allocated by NewInterpreter; Invoke does not
// allocate. An interpreter is not safe for concurrent use.
type Interpreter struct {
	model *Model
	// tensors[subgraph][index]
	tensors [][]Tensor
	// variables[i] is the state buffer of model.Variables[i]
	variables [][]byte
	// initialized reports whether the init subgraph has run (CALL_ONCE semantics).
	initialized bool
}

// NewInterpreter allocates every dynamic tensor of every subgraph and every state variable.
// Constant tensors alias the model buffer. The init subgraph is not run here: Invoke runs it
// once on first call (CALL_ONCE), Reset re-runs it.
func (m *Model) NewInterpreter() (*Interpreter, error) {
	it := &Interpreter{model: m, tensors: make([][]Tensor, len(m.Subgraphs))}
	for si := range m.Subgraphs {
		sg := &m.Subgraphs[si]
		it.tensors[si] = make([]Tensor, len(sg.Tensors))
		for ti := range sg.Tensors {
			info := &sg.Tensors[ti]
			t := Tensor{Info: info}
			switch {
			case info.DType == Resource:
			case info.IsConst():
				t.Data = info.Const
			default:
				t.Data = make([]byte, info.ByteSize)
			}
			it.tensors[si][ti] = t
		}
	}
	it.variables = make([][]byte, len(m.Variables))
	for vi := range m.Variables {
		it.variables[vi] = make([]byte, m.Variables[vi].ByteSize)
	}
	return it, nil
}

// Model returns the model the interpreter executes.
func (it *Interpreter) Model() *Model { return it.model }

// Input returns the i-th input tensor of the main subgraph.
func (it *Interpreter) Input(i int) *Tensor {
	return &it.tensors[0][it.model.Main().Inputs[i]]
}

// Output returns the i-th output tensor of the main subgraph.
func (it *Interpreter) Output(i int) *Tensor {
	return &it.tensors[0][it.model.Main().Outputs[i]]
}

// Tensor returns a tensor of a subgraph by index.
func (it *Interpreter) Tensor(subgraph, index int) *Tensor {
	return &it.tensors[subgraph][index]
}

// In returns the k-th input tensor of an operator, in the operator's subgraph.
func (it *Interpreter) In(op *Operator, k int) *Tensor {
	return &it.tensors[op.Subgraph][op.Inputs[k]]
}

// Out returns the k-th output tensor of an operator, in the operator's subgraph.
func (it *Interpreter) Out(op *Operator, k int) *Tensor {
	return &it.tensors[op.Subgraph][op.Outputs[k]]
}

// Variable returns the state buffer of a variable. Writing into the returned slice changes
// the interpreter state.
func (it *Interpreter) Variable(i int) []byte { return it.variables[i] }

// Initialized reports whether the init subgraph has run since creation or the last Reset.
func (it *Interpreter) Initialized() bool { return it.initialized }

// Invoke runs the main subgraph once. The init subgraph runs first if it has not run yet
// (CALL_ONCE). Any operator without a registered kernel makes Invoke fail with ErrNoKernel,
// naming the operator and its index.
func (it *Interpreter) Invoke() error {
	return it.runSubgraph(0)
}

// Reset puts the state variables back into their initial state by re-running the init
// subgraph, as a fresh interpreter would on its first Invoke. Models without an init subgraph
// have their variables zeroed.
func (it *Interpreter) Reset() error {
	it.initialized = false
	if it.model.InitSubgraph < 0 {
		for _, v := range it.variables {
			clear(v)
		}
		it.initialized = true
		return nil
	}
	return it.runInit()
}

// RunOperator executes one operator against the current tensor contents, without running
// anything else. It exists for tests that check a single kernel against oracle traces.
func (it *Interpreter) RunOperator(subgraph, index int) error {
	op := &it.model.Subgraphs[subgraph].Operators[index]
	return it.run(op)
}

func (it *Interpreter) runSubgraph(si int) error {
	ops := it.model.Subgraphs[si].Operators
	for oi := range ops {
		if err := it.run(&ops[oi]); err != nil {
			return err
		}
	}
	return nil
}

func (it *Interpreter) run(op *Operator) error {
	k := supportedOps[op.Code].kernel
	if k == nil {
		return fmt.Errorf("%w for subgraph %d operator %d (%s)", ErrNoKernel, op.Subgraph, op.Index, op.Name)
	}
	if err := k(it, op); err != nil {
		return fmt.Errorf("subgraph %d operator %d (%s): %w", op.Subgraph, op.Index, op.Name, err)
	}
	return nil
}

// runInit executes the init subgraph and marks the interpreter initialized.
func (it *Interpreter) runInit() error {
	if err := it.runSubgraph(it.model.InitSubgraph); err != nil {
		return fmt.Errorf("init subgraph: %w", err)
	}
	it.initialized = true
	return nil
}
