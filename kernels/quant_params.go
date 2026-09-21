package kernels

import (
	"fmt"
	"math"

	"github.com/tgallice/wakeword-go/runtime"
)

// Shared parameter computation of the int8 arithmetic operators, done once per interpreter by
// their prepare steps (tensorflow/lite/kernels/kernel_util.cc and
// tensorflow/lite/kernels/padding.h).

// activationRange returns the clamp bounds applied after requantization for a fused activation
// on a quantized output tensor (kernel_util.cc, CalculateActivationRangeQuantized).
func activationRange(act runtime.Activation, out *runtime.TensorInfo) (actMin, actMax int32, err error) {
	var qmin, qmax int32
	switch out.DType {
	case runtime.Int8:
		qmin, qmax = math.MinInt8, math.MaxInt8
	case runtime.UInt8:
		qmin, qmax = 0, math.MaxUint8
	default:
		return 0, 0, fmt.Errorf("activation range on %s output", out.DType)
	}
	scale, zeroPoint := out.Quant.Scales[0], out.Quant.ZeroPoints[0]
	// quantize mirrors kernel_util.cc Quantize: float division and rounding, then offset.
	quantize := func(f float32) (int32, error) {
		tmp := float32(math.Round(float64(f / scale)))
		if tmp < math.MinInt32 || tmp > math.MaxInt32 {
			return 0, fmt.Errorf("activation bound %v does not fit int32 at scale %v", f, scale)
		}
		return zeroPoint + int32(tmp), nil
	}
	switch act {
	case runtime.ActivationNone:
		return qmin, qmax, nil
	case runtime.ActivationRelu:
		lo, err := quantize(0)
		if err != nil {
			return 0, 0, err
		}
		return max(qmin, lo), qmax, nil
	case runtime.ActivationRelu6:
		lo, err := quantize(0)
		if err != nil {
			return 0, 0, err
		}
		hi, err := quantize(6)
		if err != nil {
			return 0, 0, err
		}
		return max(qmin, lo), min(qmax, hi), nil
	case runtime.ActivationReluN1To1:
		lo, err := quantize(-1)
		if err != nil {
			return 0, 0, err
		}
		hi, err := quantize(1)
		if err != nil {
			return 0, 0, err
		}
		return max(qmin, lo), min(qmax, hi), nil
	default:
		return 0, 0, fmt.Errorf("unsupported fused activation %s", act)
	}
}

// computeOutSize returns the output extent of a convolution along one axis
// (padding.h, ComputeOutSizeChecked).
func computeOutSize(pad runtime.Padding, imageSize, filterSize, stride, dilation int) (int, error) {
	if imageSize < 0 || filterSize <= 0 || stride <= 0 || dilation <= 0 {
		return 0, fmt.Errorf("invalid convolution geometry: image %d, filter %d, stride %d, dilation %d",
			imageSize, filterSize, stride, dilation)
	}
	effectiveFilter := (filterSize-1)*dilation + 1
	var value int
	switch pad {
	case runtime.PaddingSame:
		value = (imageSize + stride - 1) / stride
	case runtime.PaddingValid:
		value = (imageSize + stride - effectiveFilter) / stride
	default:
		return 0, fmt.Errorf("unknown padding %d", pad)
	}
	if value < 0 {
		return 0, fmt.Errorf("negative output size along an axis (image %d, filter %d)", imageSize, filterSize)
	}
	return value, nil
}

// computePadding returns the leading padding along one axis for a given output extent
// (padding.h, ComputePaddingWithOffsetChecked). The offset (odd total padding) is not needed by
// the reference kernels, which only read the leading padding.
func computePadding(stride, dilation, inSize, filterSize, outSize int) int {
	effectiveFilter := (filterSize-1)*dilation + 1
	total := (outSize-1)*stride + effectiveFilter - inSize
	if total < 0 {
		total = 0
	}
	return total / 2
}

// checkBias validates a bias tensor against the input and filter scales as the reference does
// (kernel_util.cc, GetQuantizedConvolutionMultipler): the bias zero point is zero and the bias
// scale is within 2 % of the output scale from input_scale * filter_scale, per channel.
// numChannels bias values are expected; filterScale gives the filter scale of channel c.
func checkBias(bias, in, out *runtime.TensorInfo, numChannels int, filterScale func(c int) float32) error {
	if bias.NumElements != numChannels {
		return fmt.Errorf("bias has %d elements, expected %d", bias.NumElements, numChannels)
	}
	if bias.Quant == nil || len(bias.Quant.Scales) < 1 {
		return fmt.Errorf("bias tensor %s has no quantization", bias.Name)
	}
	outScale := float64(out.Quant.Scales[0])
	for c := range numChannels {
		k := 0
		if len(bias.Quant.Scales) > 1 {
			k = c
		}
		if bias.Quant.ZeroPoints[k] != 0 {
			return fmt.Errorf("bias channel %d has zero point %d, expected 0", c, bias.Quant.ZeroPoints[k])
		}
		product := float64(in.Quant.Scales[0]) * float64(filterScale(c))
		diff := math.Abs(product - float64(bias.Quant.Scales[k]))
		if diff/outScale > 0.02 {
			return fmt.Errorf("bias channel %d scale %v differs from input * filter scale %v beyond 2 %% of the output scale",
				c, bias.Quant.Scales[k], product)
		}
	}
	return nil
}

// perChannelMultipliers computes the output multiplier and shift of every output channel
// (kernel_util.cc, PopulateConvolutionQuantizationParams): the effective scale is computed in
// double from the float32 scales, then quantized.
func perChannelMultipliers(in, filter, out *runtime.TensorInfo, numChannels int) (multipliers, shifts []int32) {
	multipliers = make([]int32, numChannels)
	shifts = make([]int32, numChannels)
	inScale, outScale := float64(in.Quant.Scales[0]), float64(out.Quant.Scales[0])
	perChannel := len(filter.Quant.Scales) > 1
	for c := range numChannels {
		scale := filter.Quant.Scales[0]
		if perChannel {
			scale = filter.Quant.Scales[c]
		}
		effective := inScale * float64(scale) / outScale
		m, s := QuantizeMultiplier(effective)
		multipliers[c], shifts[c] = m, int32(s)
	}
	return multipliers, shifts
}
