package kernels

import (
	"fmt"

	"github.com/tgallice/wakeword-go/runtime"
)

// convParams is what the CONV_2D and DEPTHWISE_CONV_2D prepare steps compute once per
// interpreter: geometry, padding, offsets, activation bounds and per-channel requantization.
type convParams struct {
	batches, inH, inW, inDepth      int
	filterH, filterW, filterInDepth int
	outH, outW, outDepth            int
	strideH, strideW                int
	dilationH, dilationW            int
	padH, padW                      int
	inputOffset, outputOffset       int32
	actMin, actMax                  int32
	multipliers, shifts             []int32
	depthMultiplier                 int
	groups, filtersPerGroup         int
}

// convGeometry fills the shape, stride, padding and output size fields shared by the two
// convolution kinds, checking the output tensor against the size the reference computes at
// prepare time (padding.h, ComputePaddingHeightWidth). filterOutDepth is the filter dimension
// holding the output channel count.
func convGeometry(p *convParams, in, filter, out *runtime.TensorInfo, pad runtime.Padding, filterOutDepth int) error {
	if len(in.Shape) != 4 || len(filter.Shape) != 4 || len(out.Shape) != 4 {
		return fmt.Errorf("expected 4-D tensors, got input %v, filter %v, output %v", in.Shape, filter.Shape, out.Shape)
	}
	p.batches, p.inH, p.inW, p.inDepth = in.Shape[0], in.Shape[1], in.Shape[2], in.Shape[3]
	p.filterH, p.filterW = filter.Shape[1], filter.Shape[2]
	p.outDepth = filterOutDepth
	outH, err := computeOutSize(pad, p.inH, p.filterH, p.strideH, p.dilationH)
	if err != nil {
		return err
	}
	outW, err := computeOutSize(pad, p.inW, p.filterW, p.strideW, p.dilationW)
	if err != nil {
		return err
	}
	want := [4]int{p.batches, outH, outW, p.outDepth}
	if got := [4]int{out.Shape[0], out.Shape[1], out.Shape[2], out.Shape[3]}; got != want {
		return fmt.Errorf("output shape %v, expected %v", out.Shape, want[:])
	}
	p.outH, p.outW = outH, outW
	p.padH = computePadding(p.strideH, p.dilationH, p.inH, p.filterH, outH)
	p.padW = computePadding(p.strideW, p.dilationW, p.inW, p.filterW, outW)
	return nil
}

// prepareConv computes the parameters of a CONV_2D operator
// (tflite-micro kernels/conv_common.cc, CalculateOpDataConv).
func prepareConv(it *runtime.Interpreter, op *runtime.Operator) (any, error) {
	in, filter, bias, out := it.In(op, 0).Info, it.In(op, 1).Info, it.In(op, 2).Info, it.Out(op, 0).Info
	o := op.Options.(*runtime.Conv2DOptions)
	p := &convParams{
		strideH: o.StrideH, strideW: o.StrideW, dilationH: o.DilationH, dilationW: o.DilationW,
		depthMultiplier: 1,
	}
	if err := convGeometry(p, in, filter, out, o.Padding, filter.Shape[0]); err != nil {
		return nil, err
	}
	p.filterInDepth = filter.Shape[3]
	if p.filterInDepth == 0 || p.inDepth%p.filterInDepth != 0 {
		return nil, fmt.Errorf("input depth %d is not a multiple of filter input depth %d", p.inDepth, p.filterInDepth)
	}
	p.groups = p.inDepth / p.filterInDepth
	if p.outDepth%p.groups != 0 {
		return nil, fmt.Errorf("output depth %d is not a multiple of the %d groups", p.outDepth, p.groups)
	}
	p.filtersPerGroup = p.outDepth / p.groups
	if filter.Quant.PerChannel() && (filter.Quant.QuantizedDimension != 0 || len(filter.Quant.Scales) != p.outDepth) {
		return nil, fmt.Errorf("filter quantization has %d scales on dimension %d, expected %d on dimension 0",
			len(filter.Quant.Scales), filter.Quant.QuantizedDimension, p.outDepth)
	}
	return p, finishConvParams(p, in, filter, bias, out, o.Activation)
}

// prepareDepthwiseConv computes the parameters of a DEPTHWISE_CONV_2D operator
// (tflite-micro kernels/depthwise_conv_common.cc, CalculateOpDataDepthwiseConv).
func prepareDepthwiseConv(it *runtime.Interpreter, op *runtime.Operator) (any, error) {
	in, filter, bias, out := it.In(op, 0).Info, it.In(op, 1).Info, it.In(op, 2).Info, it.Out(op, 0).Info
	o := op.Options.(*runtime.DepthwiseConv2DOptions)
	p := &convParams{
		strideH: o.StrideH, strideW: o.StrideW, dilationH: o.DilationH, dilationW: o.DilationW,
		depthMultiplier: o.DepthMultiplier,
	}
	if len(filter.Shape) != 4 {
		return nil, fmt.Errorf("expected a 4-D filter, got %v", filter.Shape)
	}
	if err := convGeometry(p, in, filter, out, o.Padding, filter.Shape[3]); err != nil {
		return nil, err
	}
	if p.depthMultiplier < 1 || p.outDepth != p.inDepth*p.depthMultiplier {
		return nil, fmt.Errorf("output depth %d, expected input depth %d times depth multiplier %d",
			p.outDepth, p.inDepth, p.depthMultiplier)
	}
	if filter.Quant.PerChannel() && (filter.Quant.QuantizedDimension != 3 || len(filter.Quant.Scales) != p.outDepth) {
		return nil, fmt.Errorf("filter quantization has %d scales on dimension %d, expected %d on dimension 3",
			len(filter.Quant.Scales), filter.Quant.QuantizedDimension, p.outDepth)
	}
	return p, finishConvParams(p, in, filter, bias, out, o.Activation)
}

// finishConvParams computes the quantization-dependent fields common to both convolutions.
func finishConvParams(p *convParams, in, filter, bias, out *runtime.TensorInfo, act runtime.Activation) error {
	if filter.Quant.ZeroPoints[0] != 0 {
		return fmt.Errorf("filter zero point %d, expected 0 for int8 weights", filter.Quant.ZeroPoints[0])
	}
	filterScale := func(c int) float32 {
		if filter.Quant.PerChannel() {
			return filter.Quant.Scales[c]
		}
		return filter.Quant.Scales[0]
	}
	if err := checkBias(bias, in, out, p.outDepth, filterScale); err != nil {
		return err
	}
	p.multipliers, p.shifts = perChannelMultipliers(in, filter, out, p.outDepth)
	p.inputOffset = -in.Quant.ZeroPoints[0]
	p.outputOffset = out.Quant.ZeroPoints[0]
	var err error
	p.actMin, p.actMax, err = activationRange(act, out)
	return err
}

// conv2D is the int8 per-channel convolution
// (tensorflow/lite/kernels/internal/reference/integer_ops/conv.h, ConvPerChannel).
func conv2D(it *runtime.Interpreter, op *runtime.Operator) error {
	p := it.Params(op).(*convParams)
	input, filter, bias := it.In(op, 0).Int8(), it.In(op, 1).Int8(), it.In(op, 2).Int32()
	output := it.Out(op, 0).Int8()
	for batch := 0; batch < p.batches; batch++ {
		for outY := 0; outY < p.outH; outY++ {
			inYOrigin := outY*p.strideH - p.padH
			for outX := 0; outX < p.outW; outX++ {
				inXOrigin := outX*p.strideW - p.padW
				for outChannel := 0; outChannel < p.outDepth; outChannel++ {
					group := outChannel / p.filtersPerGroup
					var acc int32
					for filterY := 0; filterY < p.filterH; filterY++ {
						inY := inYOrigin + p.dilationH*filterY
						for filterX := 0; filterX < p.filterW; filterX++ {
							inX := inXOrigin + p.dilationW*filterX
							// Zero padding by omitting the areas outside the image.
							if inX < 0 || inX >= p.inW || inY < 0 || inY >= p.inH {
								continue
							}
							inBase := ((batch*p.inH+inY)*p.inW+inX)*p.inDepth + group*p.filterInDepth
							filterBase := ((outChannel*p.filterH+filterY)*p.filterW + filterX) * p.filterInDepth
							for inChannel := 0; inChannel < p.filterInDepth; inChannel++ {
								inputVal := int32(input[inBase+inChannel])
								filterVal := int32(filter[filterBase+inChannel])
								acc += filterVal * (inputVal + p.inputOffset)
							}
						}
					}
					acc += bias[outChannel]
					acc = MultiplyByQuantizedMultiplier(acc, p.multipliers[outChannel], int(p.shifts[outChannel]))
					acc += p.outputOffset
					acc = max(acc, p.actMin)
					acc = min(acc, p.actMax)
					output[((batch*p.outH+outY)*p.outW+outX)*p.outDepth+outChannel] = int8(acc)
				}
			}
		}
	}
	return nil
}

// depthwiseConv2D is the int8 per-channel depthwise convolution
// (tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h,
// DepthwiseConvPerChannel).
func depthwiseConv2D(it *runtime.Interpreter, op *runtime.Operator) error {
	p := it.Params(op).(*convParams)
	input, filter, bias := it.In(op, 0).Int8(), it.In(op, 1).Int8(), it.In(op, 2).Int32()
	output := it.Out(op, 0).Int8()
	for batch := 0; batch < p.batches; batch++ {
		for outY := 0; outY < p.outH; outY++ {
			for outX := 0; outX < p.outW; outX++ {
				for inChannel := 0; inChannel < p.inDepth; inChannel++ {
					for m := 0; m < p.depthMultiplier; m++ {
						outputChannel := m + inChannel*p.depthMultiplier
						inXOrigin := outX*p.strideW - p.padW
						inYOrigin := outY*p.strideH - p.padH
						var acc int32
						for filterY := 0; filterY < p.filterH; filterY++ {
							for filterX := 0; filterX < p.filterW; filterX++ {
								inX := inXOrigin + p.dilationW*filterX
								inY := inYOrigin + p.dilationH*filterY
								// Zero padding by omitting the areas outside the image.
								if inX >= 0 && inX < p.inW && inY >= 0 && inY < p.inH {
									inputVal := int32(input[((batch*p.inH+inY)*p.inW+inX)*p.inDepth+inChannel])
									filterVal := int32(filter[(filterY*p.filterW+filterX)*p.outDepth+outputChannel])
									acc += filterVal * (inputVal + p.inputOffset)
								}
							}
						}
						acc += bias[outputChannel]
						acc = MultiplyByQuantizedMultiplier(acc, p.multipliers[outputChannel], int(p.shifts[outputChannel]))
						acc += p.outputOffset
						acc = max(acc, p.actMin)
						acc = min(acc, p.actMax)
						output[((batch*p.outH+outY)*p.outW+outX)*p.outDepth+outputChannel] = int8(acc)
					}
				}
			}
		}
	}
	return nil
}
