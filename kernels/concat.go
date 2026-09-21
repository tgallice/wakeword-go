package kernels

import (
	"fmt"

	"github.com/tgallice/wakeword-go/runtime"
)

// concatenation joins the inputs along the axis of the options
// (tensorflow/lite/kernels/internal/reference/concatenation.h, Concatenation).
//
// The reference walks the outer dimensions (before the axis) and, for each outer index, copies
// from every input the contiguous block made of the axis dimension and everything after it.
// Inputs and output share their quantization parameters (checked at load time), so no
// requantization happens.
func concatenation(it *runtime.Interpreter, op *runtime.Operator) error {
	out := it.Out(op, 0)
	rank := len(out.Info.Shape)
	axis, ok := resolveAxis(op.Options.(*runtime.ConcatenationOptions).Axis, rank)
	if !ok {
		return fmt.Errorf("axis out of range for rank %d", rank)
	}
	esz := elementSize(out)
	outer := product(out.Info.Shape, 0, axis)
	outCopy := product(out.Info.Shape, axis, rank) * esz
	if outer*outCopy != len(out.Data) {
		return fmt.Errorf("output shape %v does not match %d bytes", out.Info.Shape, len(out.Data))
	}
	pos := 0
	for o := range outer {
		for k := range op.Inputs {
			in := it.In(op, k)
			n := product(in.Info.Shape, axis, rank) * esz
			src := in.Data[o*n : o*n+n]
			pos += copy(out.Data[pos:], src)
		}
	}
	if pos != len(out.Data) {
		return fmt.Errorf("copied %d bytes into a %d-byte output", pos, len(out.Data))
	}
	return nil
}
