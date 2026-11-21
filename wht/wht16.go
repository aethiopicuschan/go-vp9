package wht

// IWHT4x4_16 is the lossless inverse Walsh-Hadamard for the high-bit-depth
// (int64) path: identical butterfly to IWHT4x4 but int64 throughout (12-bit
// lossless residuals exceed int32 during the transform).
func IWHT4x4_16(c []int64) {
	// Row pass: process each of the 4 rows, input >> UNIT_QUANT_SHIFT.
	for r := 0; r < 4; r++ {
		a := c[r*4+0] >> 2
		b := c[r*4+3] >> 2 // note: libvpx orders as a,c,d,b → a,b in our notation is a,d
		cc := c[r*4+1] >> 2
		d := c[r*4+2] >> 2

		a, b, cc, d = iwht4_64(a, b, cc, d)

		c[r*4+0] = a
		c[r*4+1] = b
		c[r*4+2] = cc
		c[r*4+3] = d
	}

	// Column pass: same input mapping as the row pass (libvpx a1,c1,d1,b1 =
	// ip[0],ip[4],ip[8],ip[12]), i.e. b comes from row 3.
	for col := 0; col < 4; col++ {
		a := c[0*4+col]
		b := c[3*4+col]
		cc := c[1*4+col]
		d := c[2*4+col]

		a, b, cc, d = iwht4_64(a, b, cc, d)

		c[0*4+col] = a
		c[1*4+col] = b
		c[2*4+col] = cc
		c[3*4+col] = d
	}
}

func iwht4_64(a, b, c, d int64) (int64, int64, int64, int64) {
	a1 := a + c
	d1 := d - b
	e1 := (a1 - d1) >> 1
	b = e1 - b
	c = e1 - c
	a = a1 - b
	d = d1 + c
	return a, b, c, d
}
