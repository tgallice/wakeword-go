package frontend

import "math/bits"

// mostSignificantBit32 mirrors MostSignificantBit32 in bits.h: the 1-based
// position of the highest set bit, 0 for zero.
func mostSignificantBit32(n uint32) int {
	return bits.Len32(n)
}

// mostSignificantBit64 mirrors MostSignificantBit64 in bits.h.
func mostSignificantBit64(n uint64) int {
	return bits.Len64(n)
}
