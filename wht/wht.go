// Package wht implements the 4×4 inverse Walsh-Hadamard transform used in VP9
// lossless mode (spec §8.7.1.7 / libvpx vp9_iwht4x4_16_add_c).
//
// VP9 lossless mode is active when base_q_idx == 0 and all delta-q fields are
// zero.  In that mode every block uses TX_4X4 with WHT instead of DCT.
//
// # Coefficient scaling
//
// In lossless mode the DC/AC quantizer step sizes returned by the dequant
// package are UNIT_QUANT_FACTOR = 4, not 1.  IWHT4x4 therefore right-shifts
// the row-pass input by UNIT_QUANT_SHIFT = 2 to cancel this scaling, exactly
// as libvpx does.
package wht

// IWHT4x4 performs an in-place 4×4 inverse Walsh-Hadamard transform.
//
// c must have exactly 16 elements in row-major order.  On return, c contains
// the 16 output coefficients (still int16; caller clips to [0,255] when adding
// to the prediction block).
//
// Input values are assumed to be dequantized (×4 for lossless); the function
// divides by 4 internally in the row pass.
func IWHT4x4(c []int16) {
	// Row pass: process each of the 4 rows, input >> UNIT_QUANT_SHIFT.
	for r := 0; r < 4; r++ {
		a := int32(c[r*4+0]) >> 2
		b := int32(c[r*4+3]) >> 2 // note: libvpx orders as a,c,d,b → a,b in our notation is a,d
		cc := int32(c[r*4+1]) >> 2
		d := int32(c[r*4+2]) >> 2

		a, b, cc, d = iwht4(a, b, cc, d)

		c[r*4+0] = int16(a)
		c[r*4+1] = int16(b)
		c[r*4+2] = int16(cc)
		c[r*4+3] = int16(d)
	}

	// Column pass: same input mapping as the row pass (libvpx a1,c1,d1,b1 =
	// ip[0],ip[4],ip[8],ip[12]), i.e. b comes from row 3.
	for col := 0; col < 4; col++ {
		a := int32(c[0*4+col])
		b := int32(c[3*4+col])
		cc := int32(c[1*4+col])
		d := int32(c[2*4+col])

		a, b, cc, d = iwht4(a, b, cc, d)

		c[0*4+col] = int16(a)
		c[1*4+col] = int16(b)
		c[2*4+col] = int16(cc)
		c[3*4+col] = int16(d)
	}
}

// iwht4 is one 4-point inverse WHT butterfly (spec §8.7.1.7).
// Input order: a, b, c, d → same as libvpx's (a1, b1, c1, d1) after reorder.
func iwht4(a, b, c, d int32) (int32, int32, int32, int32) {
	a1 := a + c
	d1 := d - b
	e1 := (a1 - d1) >> 1
	b = e1 - b
	c = e1 - c
	a = a1 - b
	d = d1 + c
	return a, b, c, d
}
