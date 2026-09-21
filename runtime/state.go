package runtime

import (
	"fmt"

	"github.com/tgallice/wakeword-go/tflite"
)

// The state operators are interpreter mechanics rather than arithmetic: they move bytes between
// state variables and tensors, and CALL_ONCE drives the init subgraph. They live here because
// they need the interpreter internals; the kernels package holds every other operator.
func init() {
	RegisterKernel(tflite.BuiltinOperatorVAR_HANDLE, varHandle)
	RegisterKernel(tflite.BuiltinOperatorREAD_VARIABLE, readVariable)
	RegisterKernel(tflite.BuiltinOperatorASSIGN_VARIABLE, assignVariable)
	RegisterKernel(tflite.BuiltinOperatorCALL_ONCE, callOnce)
}

// varHandle is a no-op at execution time: the resource it produces was resolved to a variable
// index when the model was loaded (Operator.Variable).
func varHandle(*Interpreter, *Operator) error { return nil }

// readVariable copies the variable state into the output tensor
// (tflite-micro kernels/read_variable.cc).
func readVariable(it *Interpreter, op *Operator) error {
	out := it.Out(op, 0)
	state := it.variables[op.Variable]
	if len(out.Data) != len(state) {
		return fmt.Errorf("variable %q holds %d bytes, output tensor %s has %d",
			it.model.Variables[op.Variable].SharedName, len(state), out.Info.Name, len(out.Data))
	}
	copy(out.Data, state)
	return nil
}

// assignVariable copies the value tensor into the variable state
// (tflite-micro kernels/assign_variable.cc).
func assignVariable(it *Interpreter, op *Operator) error {
	in := it.In(op, 1)
	state := it.variables[op.Variable]
	if len(in.Data) != len(state) {
		return fmt.Errorf("variable %q holds %d bytes, value tensor %s has %d",
			it.model.Variables[op.Variable].SharedName, len(state), in.Info.Name, len(in.Data))
	}
	copy(state, in.Data)
	return nil
}

// callOnce runs the init subgraph the first time it executes and nothing afterwards
// (tflite-micro kernels/call_once.cc).
func callOnce(it *Interpreter, op *Operator) error {
	if it.initialized {
		return nil
	}
	o := op.Options.(*CallOnceOptions)
	if o.InitSubgraphIndex != it.model.InitSubgraph {
		return fmt.Errorf("init subgraph %d does not match the model's %d", o.InitSubgraphIndex, it.model.InitSubgraph)
	}
	return it.runInit()
}
