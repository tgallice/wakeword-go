package runtime

import (
	"errors"
	"fmt"
	"testing"
)

func fmtSscan(s string, v *int) (int, error) { return fmt.Sscan(s, v) }

func TestNewInterpreterAllocates(t *testing.T) {
	m, err := Load(readModel(t, "v2/okay_nabu.tflite"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	it, err := m.NewInterpreter()
	if err != nil {
		t.Fatalf("NewInterpreter: %v", err)
	}
	in := it.Input(0)
	if len(in.Data) != 120 || len(in.Int8()) != 120 {
		t.Fatalf("input buffer: %d bytes", len(in.Data))
	}
	frames := make([]int8, 120)
	for i := range frames {
		frames[i] = int8(i - 60)
	}
	if err := in.SetInt8(frames); err != nil {
		t.Fatalf("SetInt8: %v", err)
	}
	if got := in.Int8()[119]; got != 59 {
		t.Errorf("SetInt8 did not write through: %d", got)
	}
	if err := in.SetInt8(frames[:10]); err == nil {
		t.Error("SetInt8 accepted a wrong length")
	}
	if out := it.Output(0); len(out.Uint8()) != 1 {
		t.Errorf("output buffer: %d bytes", len(out.Data))
	}
	for si, sg := range m.Subgraphs {
		for ti := range sg.Tensors {
			tt := it.Tensor(si, ti)
			switch {
			case tt.Info.DType == Resource:
				if tt.Data != nil {
					t.Errorf("resource tensor %d/%d has data", si, ti)
				}
			case tt.Info.IsConst():
				if &tt.Data[0] != &tt.Info.Const[0] {
					t.Errorf("constant tensor %d/%d was copied", si, ti)
				}
			default:
				if len(tt.Data) != tt.Info.ByteSize {
					t.Errorf("tensor %d/%d: %d bytes, want %d", si, ti, len(tt.Data), tt.Info.ByteSize)
				}
			}
		}
	}
	for vi, v := range m.Variables {
		if len(it.Variable(vi)) != v.ByteSize {
			t.Errorf("variable %d: %d bytes, want %d", vi, len(it.Variable(vi)), v.ByteSize)
		}
	}
	if err := it.Invoke(); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("Invoke: got %v, want ErrNotImplemented", err)
	}
	if err := it.Reset(); !errors.Is(err, ErrNotImplemented) {
		t.Errorf("Reset: got %v, want ErrNotImplemented", err)
	}
	// Int32 constants are readable as little-endian values (the RESHAPE target shape).
	for _, op := range m.Main().Operators {
		if op.Name == "RESHAPE" {
			shape := it.Tensor(0, op.Inputs[1]).Int32()
			if len(shape) == 4 && (shape[0] != 1 || shape[1] != 3 || shape[2] != 1 || shape[3] != 40) {
				t.Errorf("RESHAPE shape constant: %v", shape)
			}
		}
	}
}
