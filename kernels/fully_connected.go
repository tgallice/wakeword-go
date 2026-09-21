package kernels

import (
	"fmt"

	"github.com/tgallice/wakeword-go/runtime"
)

// fullyConnectedParams is what the FULLY_CONNECTED prepare step computes once per interpreter.
type fullyConnectedParams struct {
	batches, outputDepth, accumDepth int
	inputOffset, filterOffset        int32
	outputOffset                     int32
	multiplier                       int32
	shift                            int
	actMin, actMax                   int32
}

// prepareFullyConnected computes the parameters of a FULLY_CONNECTED operator with per-tensor
// weights (tflite-micro kernels/fully_connected_common.cc, CalculateOpDataFullyConnected, and
// kernel_util.cc GetQuantizedConvolutionMultipler). Unlike the convolutions, the effective
// scale is the float32 product of the input and filter scales, converted to double afterwards.
func prepareFullyConnected(it *runtime.Interpreter, op *runtime.Operator) (any, error) {
	in, filter, bias, out := it.In(op, 0).Info, it.In(op, 1).Info, it.In(op, 2).Info, it.Out(op, 0).Info
	o := op.Options.(*runtime.FullyConnectedOptions)
	if len(filter.Shape) < 2 || len(out.Shape) < 1 {
		return nil, fmt.Errorf("filter %v must be at least 2-D and output %v at least 1-D", filter.Shape, out.Shape)
	}
	if filter.Quant.PerChannel() {
		return nil, fmt.Errorf("per-channel weights are not supported")
	}
	p := &fullyConnectedParams{}
	p.outputDepth = out.Shape[len(out.Shape)-1]
	p.accumDepth = filter.Shape[len(filter.Shape)-1]
	if p.outputDepth == 0 {
		return nil, fmt.Errorf("output %v has no channels", out.Shape)
	}
	p.batches = out.NumElements / p.outputDepth
	if p.outputDepth > filter.Shape[len(filter.Shape)-2] {
		return nil, fmt.Errorf("output depth %d exceeds the filter's %d rows", p.outputDepth, filter.Shape[len(filter.Shape)-2])
	}
	if in.NumElements != p.batches*p.accumDepth {
		return nil, fmt.Errorf("input has %d elements, expected %d batches x %d", in.NumElements, p.batches, p.accumDepth)
	}
	if err := checkBias(bias, in, out, p.outputDepth, func(int) float32 { return filter.Quant.Scales[0] }); err != nil {
		return nil, err
	}
	inputProductScale := float64(in.Quant.Scales[0] * filter.Quant.Scales[0])
	if inputProductScale < 0 {
		return nil, fmt.Errorf("negative input * filter scale %v", inputProductScale)
	}
	realMultiplier := inputProductScale / float64(out.Quant.Scales[0])
	p.multiplier, p.shift = QuantizeMultiplier(realMultiplier)
	p.inputOffset = -in.Quant.ZeroPoints[0]
	p.filterOffset = -filter.Quant.ZeroPoints[0]
	p.outputOffset = out.Quant.ZeroPoints[0]
	var err error
	p.actMin, p.actMax, err = activationRange(o.Activation, out)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// fullyConnected is the int8 fully connected layer with per-tensor weights
// (tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h, FullyConnected).
func fullyConnected(it *runtime.Interpreter, op *runtime.Operator) error {
	p := it.Params(op).(*fullyConnectedParams)
	input, filter, bias := it.In(op, 0).Int8(), it.In(op, 1).Int8(), it.In(op, 2).Int32()
	output := it.Out(op, 0).Int8()
	for b := 0; b < p.batches; b++ {
		for outC := 0; outC < p.outputDepth; outC++ {
			var acc int32
			for d := 0; d < p.accumDepth; d++ {
				inputVal := int32(input[b*p.accumDepth+d])
				filterVal := int32(filter[outC*p.accumDepth+d])
				acc += (filterVal + p.filterOffset) * (inputVal + p.inputOffset)
			}
			acc += bias[outC]
			accScaled := MultiplyByQuantizedMultiplier(acc, p.multiplier, p.shift)
			accScaled += p.outputOffset
			accScaled = max(accScaled, p.actMin)
			accScaled = min(accScaled, p.actMax)
			output[outC+p.outputDepth*b] = int8(accScaled)
		}
	}
	return nil
}
