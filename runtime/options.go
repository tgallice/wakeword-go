package runtime

import (
	"fmt"

	flatbuffers "github.com/google/flatbuffers/go"

	"github.com/tgallice/wakeword-go/tflite"
)

// Padding is the convolution padding mode.
type Padding uint8

// Padding modes.
const (
	PaddingSame Padding = iota
	PaddingValid
)

// String returns the TFLite name of the padding mode.
func (p Padding) String() string {
	if p == PaddingSame {
		return "SAME"
	}
	return "VALID"
}

// Activation is a fused activation function.
type Activation uint8

// Fused activation functions accepted by the loader.
const (
	ActivationNone Activation = iota
	ActivationRelu
	ActivationReluN1To1
	ActivationRelu6
)

// String returns the TFLite name of the activation.
func (a Activation) String() string {
	switch a {
	case ActivationNone:
		return "NONE"
	case ActivationRelu:
		return "RELU"
	case ActivationReluN1To1:
		return "RELU_N1_TO_1"
	case ActivationRelu6:
		return "RELU6"
	default:
		return fmt.Sprintf("Activation(%d)", uint8(a))
	}
}

// Conv2DOptions are the options of CONV_2D.
type Conv2DOptions struct {
	Padding    Padding
	StrideH    int
	StrideW    int
	DilationH  int
	DilationW  int
	Activation Activation
}

// DepthwiseConv2DOptions are the options of DEPTHWISE_CONV_2D.
type DepthwiseConv2DOptions struct {
	Padding         Padding
	StrideH         int
	StrideW         int
	DilationH       int
	DilationW       int
	DepthMultiplier int
	Activation      Activation
}

// FullyConnectedOptions are the options of FULLY_CONNECTED.
type FullyConnectedOptions struct {
	Activation               Activation
	KeepNumDims              bool
	AsymmetricQuantizeInputs bool
}

// ConcatenationOptions are the options of CONCATENATION. Axis may be negative.
type ConcatenationOptions struct {
	Axis       int
	Activation Activation
}

// StridedSliceOptions are the bit masks of STRIDED_SLICE.
type StridedSliceOptions struct {
	BeginMask      int32
	EndMask        int32
	EllipsisMask   int32
	NewAxisMask    int32
	ShrinkAxisMask int32
	Offset         bool
}

// SplitVOptions are the options of SPLIT_V.
type SplitVOptions struct {
	NumSplits int
}

// CallOnceOptions are the options of CALL_ONCE.
type CallOnceOptions struct {
	InitSubgraphIndex int
}

// VarHandleOptions identify the state variable referenced by a VAR_HANDLE.
type VarHandleOptions struct {
	Container  string
	SharedName string
}

func paddingFromTFLite(p tflite.Padding) (Padding, error) {
	switch p {
	case tflite.PaddingSAME:
		return PaddingSame, nil
	case tflite.PaddingVALID:
		return PaddingValid, nil
	default:
		return 0, fmt.Errorf("unknown padding %d", int8(p))
	}
}

func activationFromTFLite(a tflite.ActivationFunctionType) (Activation, error) {
	switch a {
	case tflite.ActivationFunctionTypeNONE:
		return ActivationNone, nil
	case tflite.ActivationFunctionTypeRELU:
		return ActivationRelu, nil
	case tflite.ActivationFunctionTypeRELU_N1_TO_1:
		return ActivationReluN1To1, nil
	case tflite.ActivationFunctionTypeRELU6:
		return ActivationRelu6, nil
	default:
		return 0, fmt.Errorf("unsupported fused activation %s", a.String())
	}
}

// parseOptions decodes the builtin options of an operator into the typed struct for its code.
// Operators without relevant options (LOGISTIC, QUANTIZE, RESHAPE, READ_VARIABLE,
// ASSIGN_VARIABLE) yield nil.
func parseOptions(code tflite.BuiltinOperator, op *tflite.Operator) (any, error) {
	var table flatbuffers.Table
	has := op.BuiltinOptions(&table)
	want := expectedOptionsType(code)
	if want == tflite.BuiltinOptionsNONE {
		return nil, nil
	}
	if !has {
		return nil, fmt.Errorf("missing %s options", want.String())
	}
	if got := op.BuiltinOptionsType(); got != want {
		return nil, fmt.Errorf("options type %s, expected %s", got.String(), want.String())
	}
	switch code {
	case tflite.BuiltinOperatorCONV_2D:
		var o tflite.Conv2DOptions
		o.Init(table.Bytes, table.Pos)
		pad, err := paddingFromTFLite(o.Padding())
		if err != nil {
			return nil, err
		}
		act, err := activationFromTFLite(o.FusedActivationFunction())
		if err != nil {
			return nil, err
		}
		return &Conv2DOptions{
			Padding: pad, StrideH: int(o.StrideH()), StrideW: int(o.StrideW()),
			DilationH: int(o.DilationHFactor()), DilationW: int(o.DilationWFactor()), Activation: act,
		}, nil
	case tflite.BuiltinOperatorDEPTHWISE_CONV_2D:
		var o tflite.DepthwiseConv2DOptions
		o.Init(table.Bytes, table.Pos)
		pad, err := paddingFromTFLite(o.Padding())
		if err != nil {
			return nil, err
		}
		act, err := activationFromTFLite(o.FusedActivationFunction())
		if err != nil {
			return nil, err
		}
		return &DepthwiseConv2DOptions{
			Padding: pad, StrideH: int(o.StrideH()), StrideW: int(o.StrideW()),
			DilationH: int(o.DilationHFactor()), DilationW: int(o.DilationWFactor()),
			DepthMultiplier: int(o.DepthMultiplier()), Activation: act,
		}, nil
	case tflite.BuiltinOperatorFULLY_CONNECTED:
		var o tflite.FullyConnectedOptions
		o.Init(table.Bytes, table.Pos)
		act, err := activationFromTFLite(o.FusedActivationFunction())
		if err != nil {
			return nil, err
		}
		if wf := o.WeightsFormat(); wf != tflite.FullyConnectedOptionsWeightsFormatDEFAULT {
			return nil, fmt.Errorf("unsupported weights format %s", wf.String())
		}
		return &FullyConnectedOptions{
			Activation: act, KeepNumDims: o.KeepNumDims(), AsymmetricQuantizeInputs: o.AsymmetricQuantizeInputs(),
		}, nil
	case tflite.BuiltinOperatorCONCATENATION:
		var o tflite.ConcatenationOptions
		o.Init(table.Bytes, table.Pos)
		act, err := activationFromTFLite(o.FusedActivationFunction())
		if err != nil {
			return nil, err
		}
		return &ConcatenationOptions{Axis: int(o.Axis()), Activation: act}, nil
	case tflite.BuiltinOperatorSTRIDED_SLICE:
		var o tflite.StridedSliceOptions
		o.Init(table.Bytes, table.Pos)
		return &StridedSliceOptions{
			BeginMask: o.BeginMask(), EndMask: o.EndMask(), EllipsisMask: o.EllipsisMask(),
			NewAxisMask: o.NewAxisMask(), ShrinkAxisMask: o.ShrinkAxisMask(), Offset: o.Offset(),
		}, nil
	case tflite.BuiltinOperatorSPLIT_V:
		var o tflite.SplitVOptions
		o.Init(table.Bytes, table.Pos)
		return &SplitVOptions{NumSplits: int(o.NumSplits())}, nil
	case tflite.BuiltinOperatorCALL_ONCE:
		var o tflite.CallOnceOptions
		o.Init(table.Bytes, table.Pos)
		return &CallOnceOptions{InitSubgraphIndex: int(o.InitSubgraphIndex())}, nil
	case tflite.BuiltinOperatorVAR_HANDLE:
		var o tflite.VarHandleOptions
		o.Init(table.Bytes, table.Pos)
		return &VarHandleOptions{Container: string(o.Container()), SharedName: string(o.SharedName())}, nil
	default:
		return nil, nil
	}
}

// expectedOptionsType maps a supported operator to the builtin options union member it must
// carry, or BuiltinOptionsNONE when the loader ignores its options.
func expectedOptionsType(code tflite.BuiltinOperator) tflite.BuiltinOptions {
	switch code {
	case tflite.BuiltinOperatorCONV_2D:
		return tflite.BuiltinOptionsConv2DOptions
	case tflite.BuiltinOperatorDEPTHWISE_CONV_2D:
		return tflite.BuiltinOptionsDepthwiseConv2DOptions
	case tflite.BuiltinOperatorFULLY_CONNECTED:
		return tflite.BuiltinOptionsFullyConnectedOptions
	case tflite.BuiltinOperatorCONCATENATION:
		return tflite.BuiltinOptionsConcatenationOptions
	case tflite.BuiltinOperatorSTRIDED_SLICE:
		return tflite.BuiltinOptionsStridedSliceOptions
	case tflite.BuiltinOperatorSPLIT_V:
		return tflite.BuiltinOptionsSplitVOptions
	case tflite.BuiltinOperatorCALL_ONCE:
		return tflite.BuiltinOptionsCallOnceOptions
	case tflite.BuiltinOperatorVAR_HANDLE:
		return tflite.BuiltinOptionsVarHandleOptions
	default:
		return tflite.BuiltinOptionsNONE
	}
}
