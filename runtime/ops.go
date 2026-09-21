package runtime

import (
	"github.com/tgallice/wakeword-go/tflite"
)

// Kernel executes one operator inside an interpreter. Kernels are registered by the kernels
// package; an operator whose kernel is nil is loadable but not executable.
type Kernel func(it *Interpreter, op *Operator) error

// Prepare computes, once per interpreter, whatever an operator's kernel needs at execution
// time (quantized multipliers, padding, activation bounds). NewInterpreter calls it for every
// operator whose code has a prepare function and stores the result for Interpreter.Params.
// A prepare error makes NewInterpreter fail, naming the operator.
type Prepare func(it *Interpreter, op *Operator) (any, error)

// opSpec describes one supported builtin operator: the versions accepted at load time, the
// prepare step run at interpreter creation and the kernel used at execution time.
type opSpec struct {
	name     string
	versions []int32
	prepare  Prepare
	kernel   Kernel
}

// supportedOps is the closed operator set of docs/spec.md section 5. Versions are those observed
// in the microWakeWord v2 models; a different version is rejected because kernel semantics can
// change between versions.
var supportedOps = map[tflite.BuiltinOperator]*opSpec{
	tflite.BuiltinOperatorCONV_2D:           {name: "CONV_2D", versions: []int32{3}},
	tflite.BuiltinOperatorDEPTHWISE_CONV_2D: {name: "DEPTHWISE_CONV_2D", versions: []int32{3}},
	tflite.BuiltinOperatorFULLY_CONNECTED:   {name: "FULLY_CONNECTED", versions: []int32{4}},
	tflite.BuiltinOperatorLOGISTIC:          {name: "LOGISTIC", versions: []int32{2}},
	tflite.BuiltinOperatorQUANTIZE:          {name: "QUANTIZE", versions: []int32{1}},
	tflite.BuiltinOperatorCONCATENATION:     {name: "CONCATENATION", versions: []int32{2}},
	tflite.BuiltinOperatorSTRIDED_SLICE:     {name: "STRIDED_SLICE", versions: []int32{2}},
	tflite.BuiltinOperatorSPLIT_V:           {name: "SPLIT_V", versions: []int32{2}},
	tflite.BuiltinOperatorRESHAPE:           {name: "RESHAPE", versions: []int32{1}},
	tflite.BuiltinOperatorVAR_HANDLE:        {name: "VAR_HANDLE", versions: []int32{1}},
	tflite.BuiltinOperatorREAD_VARIABLE:     {name: "READ_VARIABLE", versions: []int32{1}},
	tflite.BuiltinOperatorASSIGN_VARIABLE:   {name: "ASSIGN_VARIABLE", versions: []int32{1}},
	tflite.BuiltinOperatorCALL_ONCE:         {name: "CALL_ONCE", versions: []int32{1}},
}

// RegisterKernel attaches the kernel for a supported operator. It panics if the operator is not
// in the supported set or already has a kernel: registration is a package initialization step,
// not a runtime decision.
func RegisterKernel(code tflite.BuiltinOperator, k Kernel) {
	spec, ok := supportedOps[code]
	if !ok {
		panic("runtime: RegisterKernel for unsupported operator " + builtinName(code))
	}
	if spec.kernel != nil {
		panic("runtime: kernel already registered for " + spec.name)
	}
	spec.kernel = k
}

// RegisterPrepare attaches the prepare step of a supported operator. Like RegisterKernel it is
// a package initialization step and panics on an unsupported or already prepared operator.
func RegisterPrepare(code tflite.BuiltinOperator, p Prepare) {
	spec, ok := supportedOps[code]
	if !ok {
		panic("runtime: RegisterPrepare for unsupported operator " + builtinName(code))
	}
	if spec.prepare != nil {
		panic("runtime: prepare already registered for " + spec.name)
	}
	spec.prepare = p
}

// builtinName returns the TFLite name of a builtin operator code, or a numeric fallback.
func builtinName(code tflite.BuiltinOperator) string {
	if s, ok := tflite.EnumNamesBuiltinOperator[code]; ok {
		return s
	}
	return code.String()
}

// resolveBuiltinCode returns the builtin code of an operator code entry, honoring the
// deprecated 8-bit field for codes below 127 as the TFLite loaders do.
func resolveBuiltinCode(oc *tflite.OperatorCode) tflite.BuiltinOperator {
	deprecated := tflite.BuiltinOperator(oc.DeprecatedBuiltinCode())
	if code := oc.BuiltinCode(); code > deprecated {
		return code
	}
	return deprecated
}
