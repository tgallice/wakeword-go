// Package wakewordgo is the root of the wakeword-go module, a pure Go wake word detector
// for microWakeWord v2 TFLite models. It holds no code; see the subpackages:
//
//   - wakeword: the detector (frontend, model and ESPHome decision logic)
//   - frontend: the tflite-micro audio frontend, 40 features every 10 ms
//   - runtime and kernels: the minimal int8 TFLite interpreter
//   - tflite: the generated FlatBuffers bindings
package wakewordgo
