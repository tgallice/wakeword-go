package kernels

import (
	"fmt"
	"math"

	"github.com/tgallice/wakeword-go/runtime"
)

// stridedSlice extracts a strided window of the input
// (tensorflow/lite/kernels/internal/reference/strided_slice.h and strided_slice_logic.h).
//
// Per axis, the start and stop indices are resolved by StartForAxis and StopForAxis: the
// begin and end masks replace the given index by the extreme value for the stride sign,
// negative indices count from the end, and the result is clamped to the axis. The kernel then
// walks the output in row-major order, reading input[start + i*stride] on each axis. The
// ellipsis, new_axis and shrink_axis masks are rejected at load time, so every axis of the
// input maps to one axis of the output.
func stridedSlice(it *runtime.Interpreter, op *runtime.Operator) error {
	in, out := it.In(op, 0), it.Out(op, 0)
	begin, end, strides := it.In(op, 1).Int32(), it.In(op, 2).Int32(), it.In(op, 3).Int32()
	opts := op.Options.(*runtime.StridedSliceOptions)
	rank := len(in.Info.Shape)
	if rank > maxDims || len(begin) != rank || len(end) != rank || len(strides) != rank || len(out.Info.Shape) != rank {
		return fmt.Errorf("rank %d input with %d begin, %d end, %d strides, rank %d output",
			rank, len(begin), len(end), len(strides), len(out.Info.Shape))
	}
	var start, stop, stride, inStride dims
	for a := range rank {
		if strides[a] == 0 {
			return fmt.Errorf("axis %d has stride 0", a)
		}
		stride[a] = int(strides[a])
		start[a] = startForAxis(opts.BeginMask, int(begin[a]), stride[a], in.Info.Shape[a], a)
		stop[a] = stopForAxis(opts.EndMask, int(end[a]), stride[a], in.Info.Shape[a], a)
		// Output extent along this axis, as computed by the reference at prepare time.
		extent := 0
		if stride[a] > 0 && stop[a] > start[a] {
			extent = (stop[a] - start[a] + stride[a] - 1) / stride[a]
		} else if stride[a] < 0 && stop[a] < start[a] {
			extent = (start[a] - stop[a] - stride[a] - 1) / -stride[a]
		}
		if extent != out.Info.Shape[a] {
			return fmt.Errorf("axis %d: slice yields %d elements, output has %d", a, extent, out.Info.Shape[a])
		}
	}
	esz := elementSize(in)
	s := esz
	for a := rank - 1; a >= 0; a-- {
		inStride[a] = s
		s *= in.Info.Shape[a]
	}
	if len(out.Data) == 0 {
		return nil
	}
	// Row-major walk over the output with an index counter per axis.
	var idx dims
	pos := 0
	for {
		off := 0
		for a := range rank {
			off += (start[a] + idx[a]*stride[a]) * inStride[a]
		}
		copy(out.Data[pos:pos+esz], in.Data[off:off+esz])
		pos += esz
		a := rank - 1
		for ; a >= 0; a-- {
			idx[a]++
			if idx[a] < out.Info.Shape[a] {
				break
			}
			idx[a] = 0
		}
		if a < 0 {
			break
		}
	}
	if pos != len(out.Data) {
		return fmt.Errorf("wrote %d bytes into a %d-byte output", pos, len(out.Data))
	}
	return nil
}

// startForAxis mirrors StartForAxis of strided_slice_logic.h.
func startForAxis(beginMask int32, start, stride, axisSize, axis int) int {
	if axisSize == 0 {
		return 0
	}
	if beginMask&(1<<axis) != 0 {
		if stride > 0 {
			start = math.MinInt32
		} else {
			start = math.MaxInt32
		}
	}
	if start < 0 {
		start += axisSize
	}
	if stride > 0 {
		return clamp(start, 0, axisSize)
	}
	return clamp(start, -1, axisSize-1)
}

// stopForAxis mirrors StopForAxis of strided_slice_logic.h for the non-shrinking case.
func stopForAxis(endMask int32, stop, stride, axisSize, axis int) int {
	if axisSize == 0 {
		return 0
	}
	if endMask&(1<<axis) != 0 {
		if stride > 0 {
			stop = math.MaxInt32
		} else {
			stop = math.MinInt32
		}
	}
	if stop < 0 {
		stop += axisSize
	}
	if stride > 0 {
		return clamp(stop, 0, axisSize)
	}
	return clamp(stop, -1, axisSize-1)
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
