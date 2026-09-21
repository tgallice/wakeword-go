package runtime

import (
	"errors"
	"fmt"
	"unsafe"
)

// ErrNotImplemented is returned by Invoke and Reset while the kernels of later phases are not
// registered yet.
var ErrNotImplemented = errors.New("runtime: not implemented")

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

// Variable returns the state buffer of a variable.
func (it *Interpreter) Variable(i int) []byte { return it.variables[i] }

// Invoke runs the main subgraph once. Until every operator of the model has a registered
// kernel it fails with ErrNotImplemented, naming the first operator without one.
func (it *Interpreter) Invoke() error {
	for si := range it.model.Subgraphs {
		for oi := range it.model.Subgraphs[si].Operators {
			op := &it.model.Subgraphs[si].Operators[oi]
			if supportedOps[op.Code].kernel == nil {
				return fmt.Errorf("%w: no kernel for subgraph %d operator %d (%s)", ErrNotImplemented, si, oi, op.Name)
			}
		}
	}
	return fmt.Errorf("%w: Invoke scheduling arrives with the kernels", ErrNotImplemented)
}

// Reset puts the state variables back into their initial state, as after the first Invoke.
func (it *Interpreter) Reset() error {
	it.initialized = false
	return fmt.Errorf("%w: Reset arrives with the variable kernels", ErrNotImplemented)
}
