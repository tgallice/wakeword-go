package frontend

// QuantizeFeature converts one frontend feature to the int8 model input the
// way ESPHome does (components/micro_wake_word/micro_wake_word.cpp):
//
//	input = (feature * 256 + 333) / 666 - 128, clamped to [-128, 127]
//
// The divisor 666 is 25.6 * 26.0 rounded: training divides the features by
// 25.6 and the quantized model input covers 0 to 26 over 256 steps. The
// formula is an integer approximation of the model's own input scale and is
// what the models see in production, so it is reproduced as is.
func QuantizeFeature(feature uint16) int8 {
	const valueScale, valueDiv = 256, 666
	v := (int32(feature)*valueScale + valueDiv/2) / valueDiv
	v += -128
	return int8(min(max(v, -128), 127))
}

// QuantizeFeatures converts a frame of features into dst, which must be at
// least as long as src.
func QuantizeFeatures(dst []int8, src []uint16) {
	for i, v := range src {
		dst[i] = QuantizeFeature(v)
	}
}
