package kernels

import (
	"fmt"

	"github.com/tgallice/wakeword-go/runtime"
)

// splitV splits the input along an axis into slices of the sizes given by size_splits
// (tensorflow/lite/micro/kernels/split_v.cc, SplitImpl).
//
// One size may be -1, meaning the remainder; the reference resolves it at prepare time and
// this kernel does the same on every call, which costs a few integer operations. The output
// tensors must have exactly the resolved sizes along the axis.
func splitV(it *runtime.Interpreter, op *runtime.Operator) error {
	in := it.In(op, 0)
	sizes := it.In(op, 1).Int32()
	axisT := it.In(op, 2).Int32()
	rank := len(in.Info.Shape)
	axis, ok := resolveAxis(int(axisT[0]), rank)
	if !ok {
		return fmt.Errorf("axis %d out of range for rank %d", axisT[0], rank)
	}
	if len(sizes) != len(op.Outputs) || len(sizes) > maxDims*4 {
		return fmt.Errorf("%d split sizes for %d outputs", len(sizes), len(op.Outputs))
	}
	var resolved [maxDims * 4]int
	remainder := -1
	total := 0
	for i, s := range sizes {
		switch {
		case s == -1 && remainder < 0:
			remainder = i
		case s < 0:
			return fmt.Errorf("invalid split sizes %v", sizes)
		default:
			resolved[i] = int(s)
			total += int(s)
		}
	}
	if remainder >= 0 {
		resolved[remainder] = in.Info.Shape[axis] - total
		if resolved[remainder] < 0 {
			return fmt.Errorf("split sizes %v exceed axis size %d", sizes, in.Info.Shape[axis])
		}
	} else if total != in.Info.Shape[axis] {
		return fmt.Errorf("split sizes %v sum to %d, axis has %d", sizes, total, in.Info.Shape[axis])
	}
	esz := elementSize(in)
	outer := product(in.Info.Shape, 0, axis)
	inner := product(in.Info.Shape, axis+1, rank) * esz
	inStep := in.Info.Shape[axis] * inner
	for k := range op.Outputs {
		out := it.Out(op, k)
		want := resolved[k] * inner * outer
		if len(out.Data) != want || out.Info.Shape[axis] != resolved[k] {
			return fmt.Errorf("output %d has shape %v, split size is %d", k, out.Info.Shape, resolved[k])
		}
	}
	for o := range outer {
		off := o * inStep
		for k := range op.Outputs {
			out := it.Out(op, k)
			n := resolved[k] * inner
			copy(out.Data[o*n:o*n+n], in.Data[off:off+n])
			off += n
		}
	}
	return nil
}
