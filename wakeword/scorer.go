package wakeword

import (
	"errors"
	"fmt"

	_ "github.com/tgallice/wakeword-go/kernels" // registers the operator kernels
	"github.com/tgallice/wakeword-go/runtime"
)

// Scorer is the model-facing side of a streaming model: a buffer of Stride frames of
// FeatureSize int8 features, scored as one uint8 probability by Invoke. The streaming logic
// writes frames into Input one at a time and calls Invoke once the buffer is full, mirroring
// how ESPHome writes into the TFLite input tensor. Tests substitute a scripted Scorer.
type Scorer interface {
	// Stride is the number of frames scored together (dimension 1 of the model input).
	Stride() int
	// FeatureSize is the number of features per frame (dimension 2 of the model input).
	FeatureSize() int
	// Input is the Stride x FeatureSize int8 buffer, frame-major, written by the caller.
	Input() []int8
	// Invoke scores the current Input and returns the probability in 0 to 255.
	Invoke() (uint8, error)
	// Reset clears the model's streaming state, as a fresh interpreter would have.
	Reset() error
}

// modelScorer scores with a runtime.Interpreter over a loaded microWakeWord model.
type modelScorer struct {
	it          *runtime.Interpreter
	in          []int8
	out         []uint8
	stride      int
	featureSize int
}

var errModelShape = errors.New("unsupported model input or output shape")

// newModelScorer loads a model and checks its input and output contract (spec section 4):
// int8 input [1, stride, features], uint8 output [1, 1].
func newModelScorer(model []byte) (*modelScorer, error) {
	m, err := runtime.Load(model)
	if err != nil {
		return nil, err
	}
	main := m.Main()
	if len(main.Inputs) != 1 || len(main.Outputs) != 1 {
		return nil, fmt.Errorf("%w: %d inputs and %d outputs, want 1 and 1", errModelShape,
			len(main.Inputs), len(main.Outputs))
	}
	in := main.Tensors[main.Inputs[0]]
	out := main.Tensors[main.Outputs[0]]
	if in.DType != runtime.Int8 || len(in.Shape) != 3 || in.Shape[0] != 1 || in.Shape[1] < 1 || in.Shape[2] < 1 {
		return nil, fmt.Errorf("%w: input %s %v, want int8 [1, stride, features]", errModelShape, in.DType, in.Shape)
	}
	if out.DType != runtime.UInt8 || out.NumElements != 1 {
		return nil, fmt.Errorf("%w: output %s %v, want uint8 [1, 1]", errModelShape, out.DType, out.Shape)
	}
	it, err := m.NewInterpreter()
	if err != nil {
		return nil, err
	}
	s := &modelScorer{
		it:          it,
		in:          it.Input(0).Int8(),
		out:         it.Output(0).Uint8(),
		stride:      in.Shape[1],
		featureSize: in.Shape[2],
	}
	return s, nil
}

// probe runs one invocation on a silent input so that an operator without a kernel is
// reported at construction rather than on the first frame, then resets the streaming state.
func (s *modelScorer) probe() error {
	for i := range s.in {
		s.in[i] = -128 // the input zero point, a silent frame
	}
	if err := s.it.Invoke(); err != nil {
		return fmt.Errorf("probe invocation: %w", err)
	}
	return s.it.Reset()
}

func (s *modelScorer) Stride() int      { return s.stride }
func (s *modelScorer) FeatureSize() int { return s.featureSize }
func (s *modelScorer) Input() []int8    { return s.in }

func (s *modelScorer) Invoke() (uint8, error) {
	if err := s.it.Invoke(); err != nil {
		return 0, err
	}
	return s.out[0], nil
}

func (s *modelScorer) Reset() error { return s.it.Reset() }
