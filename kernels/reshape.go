package kernels

import (
	"fmt"

	"github.com/tgallice/wakeword-go/runtime"
)

// reshape copies the input into the output. The target shape was checked against the output
// tensor at load time, so at execution the operator is a plain copy
// (tensorflow/lite/micro/kernels/reshape.cc).
func reshape(it *runtime.Interpreter, op *runtime.Operator) error {
	in, out := it.In(op, 0), it.Out(op, 0)
	if len(in.Data) != len(out.Data) {
		return fmt.Errorf("input has %d bytes, output %d", len(in.Data), len(out.Data))
	}
	copy(out.Data, in.Data)
	return nil
}
