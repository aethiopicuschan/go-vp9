package transform

// High-bit-depth (10/12-bit) inverse transforms — int64 pipeline (libvpx
// tran_high_t). 12-bit dequantised coefficients and the cospi products overflow
// int32, so the array, products, and butterfly intermediates are all int64.
// Stage outputs use HIGHBD_WRAPLOW(bd). For bd=8 this reproduces the 8-bit path.
// Set BD16 to the frame bit depth before InvTransform2D16.

var BD16 = 8

var (
	cp1_64  int64 = 16364
	cp2_64  int64 = 16305
	cp3_64  int64 = 16207
	cp4_64  int64 = 16069
	cp5_64  int64 = 15893
	cp6_64  int64 = 15679
	cp7_64  int64 = 15426
	cp8_64  int64 = 15137
	cp9_64  int64 = 14811
	cp10_64 int64 = 14449
	cp11_64 int64 = 14053
	cp12_64 int64 = 13623
	cp13_64 int64 = 13160
	cp14_64 int64 = 12665
	cp15_64 int64 = 12140
	cp16_64 int64 = 11585
	cp17_64 int64 = 11003
	cp18_64 int64 = 10394
	cp19_64 int64 = 9760
	cp20_64 int64 = 9102
	cp21_64 int64 = 8423
	cp22_64 int64 = 7723
	cp23_64 int64 = 7005
	cp24_64 int64 = 6270
	cp25_64 int64 = 5520
	cp26_64 int64 = 4756
	cp27_64 int64 = 3981
	cp28_64 int64 = 3196
	cp29_64 int64 = 2404
	cp30_64 int64 = 1606
	cp31_64 int64 = 804
	sp1_64  int64 = 5283
	sp2_64  int64 = 9929
	sp3_64  int64 = 13377
	sp4_64  int64 = 15212
)

func wl16(v int64) int64  { s := uint(24 - BD16); return int64((int32(v) << s) >> s) }
func drs16(v int64) int64 { return wl16((v + (1 << 13)) >> 14) }

func InvTransform2D16(coeffs []int64, bs int, tt TxType) {
	if bs == 32 {
		tt = DCT_DCT
	}
	shift := FinalShift(bs)
	// DC-only fast path for DCT_DCT: if only coeffs[0] is non-zero, all bs²
	// outputs equal a single value computed from the two cascaded 1-D DCTs.
	if tt == DCT_DCT {
		acZero := true
		for _, v := range coeffs[1:] {
			if v != 0 {
				acZero = false
				break
			}
		}
		if acZero {
			if coeffs[0] != 0 {
				v4 := DCOnlyValue16(coeffs[0], shift)
				for i := range coeffs {
					coeffs[i] = v4
				}
			}
			return
		}
	}
	// Axis assignment verified against libvpx vp9_iht* via the oracle harness:
	// ADST_DCT applies ADST down columns + DCT along rows, DCT_ADST the reverse.
	// (go's row-pass→transpose→col-pass pipeline puts the column transform in
	// the second pass, so these booleans are the transpose of libvpx's
	// {rows,cols} struct labels.)
	rowDCT := tt == DCT_DCT || tt == ADST_DCT
	colDCT := tt == DCT_DCT || tt == DCT_ADST
	for i := 0; i < bs; i++ {
		inv1D16(coeffs[i*bs:i*bs+bs], bs, rowDCT)
	}
	transpose16(coeffs, bs)
	for i := 0; i < bs; i++ {
		inv1D16(coeffs[i*bs:i*bs+bs], bs, colDCT)
	}
	transposeAndShift16(coeffs, bs, shift)
}

func transpose16(buf []int64, bs int) {
	for r := 0; r < bs; r++ {
		for c := r + 1; c < bs; c++ {
			buf[r*bs+c], buf[c*bs+r] = buf[c*bs+r], buf[r*bs+c]
		}
	}
}

func transposeAndShift16(buf []int64, bs, n int) {
	round := int64(1) << (n - 1)
	rs := func(v int64) int64 { return (v + round) >> n }
	for r := 0; r < bs; r++ {
		buf[r*bs+r] = rs(buf[r*bs+r])
		for c := r + 1; c < bs; c++ {
			a := rs(buf[r*bs+c])
			b := rs(buf[c*bs+r])
			buf[r*bs+c] = b
			buf[c*bs+r] = a
		}
	}
}

func inv1D16(row []int64, bs int, dct bool) {
	if dct {
		switch bs {
		case 4:
			idct416(row)
		case 8:
			idct816(row)
		case 16:
			idct1616(row)
		case 32:
			idct3216(row)
		}
	} else {
		switch bs {
		case 4:
			iadst416(row)
		case 8:
			iadst816(row)
		case 16:
			iadst1616(row)
		}
	}
}

func DCOnlyValue16(dc int64, shift int) int64 {
	v := drs16(int64(dc) * cp16_64)
	v = drs16(int64(v) * cp16_64)
	round := int64(1) << (shift - 1)
	return (v + round) >> shift
}

func idct416(io []int64) {
	s0 := int64(drs16((int64(io[0]) + int64(io[2])) * cp16_64))
	s1 := int64(drs16((int64(io[0]) - int64(io[2])) * cp16_64))
	s2 := int64(drs16(int64(io[1])*cp24_64 - int64(io[3])*cp8_64))
	s3 := int64(drs16(int64(io[1])*cp8_64 + int64(io[3])*cp24_64))
	io[0] = wl16(s0 + s3)
	io[1] = wl16(s1 + s2)
	io[2] = wl16(s1 - s2)
	io[3] = wl16(s0 - s3)
}

func iadst416(io []int64) {
	s0 := sp1_64 * int64(io[0])
	s1 := sp2_64 * int64(io[0])
	s2 := sp3_64 * int64(io[1])
	s3 := sp4_64 * int64(io[2])
	s4 := sp1_64 * int64(io[2])
	s5 := sp2_64 * int64(io[3])
	s6 := sp4_64 * int64(io[3])
	s7 := int64(io[0]) - int64(io[2]) + int64(io[3])
	x0 := s0 + s3 + s5
	x1 := s1 - s4 - s6
	x2 := sp3_64 * s7
	x3 := s2
	io[0] = drs16(x0 + x3)
	io[1] = drs16(x1 + x3)
	io[2] = drs16(x2)
	io[3] = drs16(x0 + x1 - x3)
}

func idct816(io []int64) {
	// stage 1: butterflies on the 4 odd positions
	s4 := drs16(int64(io[1])*cp28_64 - int64(io[7])*cp4_64)
	s7 := drs16(int64(io[1])*cp4_64 + int64(io[7])*cp28_64)
	s5 := drs16(int64(io[5])*cp12_64 - int64(io[3])*cp20_64)
	s6 := drs16(int64(io[5])*cp20_64 + int64(io[3])*cp12_64)

	// stage 2: even half via cospi16 / cospi8,24
	e0 := drs16((int64(io[0]) + int64(io[4])) * cp16_64)
	e1 := drs16((int64(io[0]) - int64(io[4])) * cp16_64)
	e2 := drs16(int64(io[2])*cp24_64 - int64(io[6])*cp8_64)
	e3 := drs16(int64(io[2])*cp8_64 + int64(io[6])*cp24_64)
	// combine odds
	o4 := wl16(int64(s4) + int64(s5))
	o5 := wl16(int64(s4) - int64(s5))
	o6 := wl16(-int64(s6) + int64(s7))
	o7 := wl16(int64(s6) + int64(s7))

	// stage 3: even butterfly + cospi16 cross
	f0 := wl16(int64(e0) + int64(e3))
	f1 := wl16(int64(e1) + int64(e2))
	f2 := wl16(int64(e1) - int64(e2))
	f3 := wl16(int64(e0) - int64(e3))
	f5 := drs16((int64(o6) - int64(o5)) * cp16_64)
	f6 := drs16((int64(o6) + int64(o5)) * cp16_64)

	// stage 4: output
	io[0] = wl16(int64(f0) + int64(o7))
	io[1] = wl16(int64(f1) + int64(f6))
	io[2] = wl16(int64(f2) + int64(f5))
	io[3] = wl16(int64(f3) + int64(o4))
	io[4] = wl16(int64(f3) - int64(o4))
	io[5] = wl16(int64(f2) - int64(f5))
	io[6] = wl16(int64(f1) - int64(f6))
	io[7] = wl16(int64(f0) - int64(o7))
}

func iadst816(io []int64) {
	x0 := int64(io[7])
	x1 := int64(io[0])
	x2 := int64(io[5])
	x3 := int64(io[2])
	x4 := int64(io[3])
	x5 := int64(io[4])
	x6 := int64(io[1])
	x7 := int64(io[6])

	if x0|x1|x2|x3|x4|x5|x6|x7 == 0 {
		for i := 0; i < 8; i++ {
			io[i] = 0
		}
		return
	}

	// stage 1
	s0 := cp2_64*x0 + cp30_64*x1
	s1 := cp30_64*x0 - cp2_64*x1
	s2 := cp10_64*x2 + cp22_64*x3
	s3 := cp22_64*x2 - cp10_64*x3
	s4 := cp18_64*x4 + cp14_64*x5
	s5 := cp14_64*x4 - cp18_64*x5
	s6 := cp26_64*x6 + cp6_64*x7
	s7 := cp6_64*x6 - cp26_64*x7

	x0 = int64(drs16(s0 + s4))
	x1 = int64(drs16(s1 + s5))
	x2 = int64(drs16(s2 + s6))
	x3 = int64(drs16(s3 + s7))
	x4 = int64(drs16(s0 - s4))
	x5 = int64(drs16(s1 - s5))
	x6 = int64(drs16(s2 - s6))
	x7 = int64(drs16(s3 - s7))

	// stage 2
	s0 = x0
	s1 = x1
	s2 = x2
	s3 = x3
	s4 = cp8_64*x4 + cp24_64*x5
	s5 = cp24_64*x4 - cp8_64*x5
	s6 = -cp24_64*x6 + cp8_64*x7
	s7 = cp8_64*x6 + cp24_64*x7

	x0 = wl16(s0 + s2)
	x1 = wl16(s1 + s3)
	x2 = wl16(s0 - s2)
	x3 = wl16(s1 - s3)
	x4 = int64(drs16(s4 + s6))
	x5 = int64(drs16(s5 + s7))
	x6 = int64(drs16(s4 - s6))
	x7 = int64(drs16(s5 - s7))

	// stage 3
	s2 = cp16_64 * (x2 + x3)
	s3 = cp16_64 * (x2 - x3)
	s6 = cp16_64 * (x6 + x7)
	s7 = cp16_64 * (x6 - x7)

	x2 = int64(drs16(s2))
	x3 = int64(drs16(s3))
	x6 = int64(drs16(s6))
	x7 = int64(drs16(s7))

	io[0] = wl16(x0)
	io[1] = wl16(-x4)
	io[2] = wl16(x6)
	io[3] = wl16(-x2)
	io[4] = wl16(x3)
	io[5] = wl16(-x7)
	io[6] = wl16(x5)
	io[7] = wl16(-x1)
}

func idct1616(io []int64) {
	var s1, s2 [16]int64

	// stage 1
	s1[0] = int64(io[0])
	s1[1] = int64(io[8])
	s1[2] = int64(io[4])
	s1[3] = int64(io[12])
	s1[4] = int64(io[2])
	s1[5] = int64(io[10])
	s1[6] = int64(io[6])
	s1[7] = int64(io[14])
	s1[8] = int64(io[1])
	s1[9] = int64(io[9])
	s1[10] = int64(io[5])
	s1[11] = int64(io[13])
	s1[12] = int64(io[3])
	s1[13] = int64(io[11])
	s1[14] = int64(io[7])
	s1[15] = int64(io[15])

	// stage 2
	s2[0], s2[1], s2[2], s2[3] = s1[0], s1[1], s1[2], s1[3]
	s2[4], s2[5], s2[6], s2[7] = s1[4], s1[5], s1[6], s1[7]
	s2[8] = int64(drs16(s1[8]*cp30_64 - s1[15]*cp2_64))
	s2[15] = int64(drs16(s1[8]*cp2_64 + s1[15]*cp30_64))
	s2[9] = int64(drs16(s1[9]*cp14_64 - s1[14]*cp18_64))
	s2[14] = int64(drs16(s1[9]*cp18_64 + s1[14]*cp14_64))
	s2[10] = int64(drs16(s1[10]*cp22_64 - s1[13]*cp10_64))
	s2[13] = int64(drs16(s1[10]*cp10_64 + s1[13]*cp22_64))
	s2[11] = int64(drs16(s1[11]*cp6_64 - s1[12]*cp26_64))
	s2[12] = int64(drs16(s1[11]*cp26_64 + s1[12]*cp6_64))

	// stage 3
	s1[0], s1[1], s1[2], s1[3] = s2[0], s2[1], s2[2], s2[3]
	s1[4] = int64(drs16(s2[4]*cp28_64 - s2[7]*cp4_64))
	s1[7] = int64(drs16(s2[4]*cp4_64 + s2[7]*cp28_64))
	s1[5] = int64(drs16(s2[5]*cp12_64 - s2[6]*cp20_64))
	s1[6] = int64(drs16(s2[5]*cp20_64 + s2[6]*cp12_64))
	s1[8] = wl16(s2[8] + s2[9])
	s1[9] = wl16(s2[8] - s2[9])
	s1[10] = wl16(-s2[10] + s2[11])
	s1[11] = wl16(s2[10] + s2[11])
	s1[12] = wl16(s2[12] + s2[13])
	s1[13] = wl16(s2[12] - s2[13])
	s1[14] = wl16(-s2[14] + s2[15])
	s1[15] = wl16(s2[14] + s2[15])

	// stage 4
	s2[0] = int64(drs16((s1[0] + s1[1]) * cp16_64))
	s2[1] = int64(drs16((s1[0] - s1[1]) * cp16_64))
	s2[2] = int64(drs16(s1[2]*cp24_64 - s1[3]*cp8_64))
	s2[3] = int64(drs16(s1[2]*cp8_64 + s1[3]*cp24_64))
	s2[4] = wl16(s1[4] + s1[5])
	s2[5] = wl16(s1[4] - s1[5])
	s2[6] = wl16(-s1[6] + s1[7])
	s2[7] = wl16(s1[6] + s1[7])
	s2[8], s2[15] = s1[8], s1[15]
	s2[9] = int64(drs16(-s1[9]*cp8_64 + s1[14]*cp24_64))
	s2[14] = int64(drs16(s1[9]*cp24_64 + s1[14]*cp8_64))
	s2[10] = int64(drs16(-s1[10]*cp24_64 - s1[13]*cp8_64))
	s2[13] = int64(drs16(-s1[10]*cp8_64 + s1[13]*cp24_64))
	s2[11], s2[12] = s1[11], s1[12]

	// stage 5
	s1[0] = wl16(s2[0] + s2[3])
	s1[1] = wl16(s2[1] + s2[2])
	s1[2] = wl16(s2[1] - s2[2])
	s1[3] = wl16(s2[0] - s2[3])
	s1[4] = s2[4]
	s1[5] = int64(drs16((s2[6] - s2[5]) * cp16_64))
	s1[6] = int64(drs16((s2[5] + s2[6]) * cp16_64))
	s1[7] = s2[7]
	s1[8] = wl16(s2[8] + s2[11])
	s1[9] = wl16(s2[9] + s2[10])
	s1[10] = wl16(s2[9] - s2[10])
	s1[11] = wl16(s2[8] - s2[11])
	s1[12] = wl16(-s2[12] + s2[15])
	s1[13] = wl16(-s2[13] + s2[14])
	s1[14] = wl16(s2[13] + s2[14])
	s1[15] = wl16(s2[12] + s2[15])

	// stage 6
	s2[0] = wl16(s1[0] + s1[7])
	s2[1] = wl16(s1[1] + s1[6])
	s2[2] = wl16(s1[2] + s1[5])
	s2[3] = wl16(s1[3] + s1[4])
	s2[4] = wl16(s1[3] - s1[4])
	s2[5] = wl16(s1[2] - s1[5])
	s2[6] = wl16(s1[1] - s1[6])
	s2[7] = wl16(s1[0] - s1[7])
	s2[8], s2[9] = s1[8], s1[9]
	s2[10] = int64(drs16((-s1[10] + s1[13]) * cp16_64))
	s2[13] = int64(drs16((s1[10] + s1[13]) * cp16_64))
	s2[11] = int64(drs16((-s1[11] + s1[12]) * cp16_64))
	s2[12] = int64(drs16((s1[11] + s1[12]) * cp16_64))
	s2[14], s2[15] = s1[14], s1[15]

	// stage 7
	io[0] = wl16(s2[0] + s2[15])
	io[1] = wl16(s2[1] + s2[14])
	io[2] = wl16(s2[2] + s2[13])
	io[3] = wl16(s2[3] + s2[12])
	io[4] = wl16(s2[4] + s2[11])
	io[5] = wl16(s2[5] + s2[10])
	io[6] = wl16(s2[6] + s2[9])
	io[7] = wl16(s2[7] + s2[8])
	io[8] = wl16(s2[7] - s2[8])
	io[9] = wl16(s2[6] - s2[9])
	io[10] = wl16(s2[5] - s2[10])
	io[11] = wl16(s2[4] - s2[11])
	io[12] = wl16(s2[3] - s2[12])
	io[13] = wl16(s2[2] - s2[13])
	io[14] = wl16(s2[1] - s2[14])
	io[15] = wl16(s2[0] - s2[15])
}

func iadst1616(io []int64) {
	x0 := int64(io[15])
	x1 := int64(io[0])
	x2 := int64(io[13])
	x3 := int64(io[2])
	x4 := int64(io[11])
	x5 := int64(io[4])
	x6 := int64(io[9])
	x7 := int64(io[6])
	x8 := int64(io[7])
	x9 := int64(io[8])
	x10 := int64(io[5])
	x11 := int64(io[10])
	x12 := int64(io[3])
	x13 := int64(io[12])
	x14 := int64(io[1])
	x15 := int64(io[14])

	if x0|x1|x2|x3|x4|x5|x6|x7|x8|x9|x10|x11|x12|x13|x14|x15 == 0 {
		for i := 0; i < 16; i++ {
			io[i] = 0
		}
		return
	}

	// stage 1
	s0 := x0*cp1_64 + x1*cp31_64
	s1 := x0*cp31_64 - x1*cp1_64
	s2 := x2*cp5_64 + x3*cp27_64
	s3 := x2*cp27_64 - x3*cp5_64
	s4 := x4*cp9_64 + x5*cp23_64
	s5 := x4*cp23_64 - x5*cp9_64
	s6 := x6*cp13_64 + x7*cp19_64
	s7 := x6*cp19_64 - x7*cp13_64
	s8 := x8*cp17_64 + x9*cp15_64
	s9 := x8*cp15_64 - x9*cp17_64
	s10 := x10*cp21_64 + x11*cp11_64
	s11 := x10*cp11_64 - x11*cp21_64
	s12 := x12*cp25_64 + x13*cp7_64
	s13 := x12*cp7_64 - x13*cp25_64
	s14 := x14*cp29_64 + x15*cp3_64
	s15 := x14*cp3_64 - x15*cp29_64

	x0 = int64(drs16(s0 + s8))
	x1 = int64(drs16(s1 + s9))
	x2 = int64(drs16(s2 + s10))
	x3 = int64(drs16(s3 + s11))
	x4 = int64(drs16(s4 + s12))
	x5 = int64(drs16(s5 + s13))
	x6 = int64(drs16(s6 + s14))
	x7 = int64(drs16(s7 + s15))
	x8 = int64(drs16(s0 - s8))
	x9 = int64(drs16(s1 - s9))
	x10 = int64(drs16(s2 - s10))
	x11 = int64(drs16(s3 - s11))
	x12 = int64(drs16(s4 - s12))
	x13 = int64(drs16(s5 - s13))
	x14 = int64(drs16(s6 - s14))
	x15 = int64(drs16(s7 - s15))

	// stage 2
	s0 = x0
	s1 = x1
	s2 = x2
	s3 = x3
	s4 = x4
	s5 = x5
	s6 = x6
	s7 = x7
	s8 = x8*cp4_64 + x9*cp28_64
	s9 = x8*cp28_64 - x9*cp4_64
	s10 = x10*cp20_64 + x11*cp12_64
	s11 = x10*cp12_64 - x11*cp20_64
	s12 = -x12*cp28_64 + x13*cp4_64
	s13 = x12*cp4_64 + x13*cp28_64
	s14 = -x14*cp12_64 + x15*cp20_64
	s15 = x14*cp20_64 + x15*cp12_64

	x0 = wl16(s0 + s4)
	x1 = wl16(s1 + s5)
	x2 = wl16(s2 + s6)
	x3 = wl16(s3 + s7)
	x4 = wl16(s0 - s4)
	x5 = wl16(s1 - s5)
	x6 = wl16(s2 - s6)
	x7 = wl16(s3 - s7)
	x8 = int64(drs16(s8 + s12))
	x9 = int64(drs16(s9 + s13))
	x10 = int64(drs16(s10 + s14))
	x11 = int64(drs16(s11 + s15))
	x12 = int64(drs16(s8 - s12))
	x13 = int64(drs16(s9 - s13))
	x14 = int64(drs16(s10 - s14))
	x15 = int64(drs16(s11 - s15))

	// stage 3
	s0 = x0
	s1 = x1
	s2 = x2
	s3 = x3
	s4 = x4*cp8_64 + x5*cp24_64
	s5 = x4*cp24_64 - x5*cp8_64
	s6 = -x6*cp24_64 + x7*cp8_64
	s7 = x6*cp8_64 + x7*cp24_64
	s8 = x8
	s9 = x9
	s10 = x10
	s11 = x11
	s12 = x12*cp8_64 + x13*cp24_64
	s13 = x12*cp24_64 - x13*cp8_64
	s14 = -x14*cp24_64 + x15*cp8_64
	s15 = x14*cp8_64 + x15*cp24_64

	x0 = wl16(s0 + s2)
	x1 = wl16(s1 + s3)
	x2 = wl16(s0 - s2)
	x3 = wl16(s1 - s3)
	x4 = int64(drs16(s4 + s6))
	x5 = int64(drs16(s5 + s7))
	x6 = int64(drs16(s4 - s6))
	x7 = int64(drs16(s5 - s7))
	x8 = wl16(s8 + s10)
	x9 = wl16(s9 + s11)
	x10 = wl16(s8 - s10)
	x11 = wl16(s9 - s11)
	x12 = int64(drs16(s12 + s14))
	x13 = int64(drs16(s13 + s15))
	x14 = int64(drs16(s12 - s14))
	x15 = int64(drs16(s13 - s15))

	// stage 4
	s2 = -cp16_64 * (x2 + x3)
	s3 = cp16_64 * (x2 - x3)
	s6 = cp16_64 * (x6 + x7)
	s7 = cp16_64 * (-x6 + x7)
	s10 = cp16_64 * (x10 + x11)
	s11 = cp16_64 * (-x10 + x11)
	s14 = -cp16_64 * (x14 + x15)
	s15 = cp16_64 * (x14 - x15)

	x2 = int64(drs16(s2))
	x3 = int64(drs16(s3))
	x6 = int64(drs16(s6))
	x7 = int64(drs16(s7))
	x10 = int64(drs16(s10))
	x11 = int64(drs16(s11))
	x14 = int64(drs16(s14))
	x15 = int64(drs16(s15))

	io[0] = wl16(x0)
	io[1] = wl16(-x8)
	io[2] = wl16(x12)
	io[3] = wl16(-x4)
	io[4] = wl16(x6)
	io[5] = wl16(x14)
	io[6] = wl16(x10)
	io[7] = wl16(x2)
	io[8] = wl16(x3)
	io[9] = wl16(x11)
	io[10] = wl16(x15)
	io[11] = wl16(x7)
	io[12] = wl16(x5)
	io[13] = wl16(-x13)
	io[14] = wl16(x9)
	io[15] = wl16(-x1)
}

func idct3216(io []int64) {
	var s1, s2 [32]int64

	// stage 1
	s1[0] = int64(io[0])
	s1[1] = int64(io[16])
	s1[2] = int64(io[8])
	s1[3] = int64(io[24])
	s1[4] = int64(io[4])
	s1[5] = int64(io[20])
	s1[6] = int64(io[12])
	s1[7] = int64(io[28])
	s1[8] = int64(io[2])
	s1[9] = int64(io[18])
	s1[10] = int64(io[10])
	s1[11] = int64(io[26])
	s1[12] = int64(io[6])
	s1[13] = int64(io[22])
	s1[14] = int64(io[14])
	s1[15] = int64(io[30])
	s1[16] = int64(drs16(int64(io[1])*cp31_64 - int64(io[31])*cp1_64))
	s1[31] = int64(drs16(int64(io[1])*cp1_64 + int64(io[31])*cp31_64))
	s1[17] = int64(drs16(int64(io[17])*cp15_64 - int64(io[15])*cp17_64))
	s1[30] = int64(drs16(int64(io[17])*cp17_64 + int64(io[15])*cp15_64))
	s1[18] = int64(drs16(int64(io[9])*cp23_64 - int64(io[23])*cp9_64))
	s1[29] = int64(drs16(int64(io[9])*cp9_64 + int64(io[23])*cp23_64))
	s1[19] = int64(drs16(int64(io[25])*cp7_64 - int64(io[7])*cp25_64))
	s1[28] = int64(drs16(int64(io[25])*cp25_64 + int64(io[7])*cp7_64))
	s1[20] = int64(drs16(int64(io[5])*cp27_64 - int64(io[27])*cp5_64))
	s1[27] = int64(drs16(int64(io[5])*cp5_64 + int64(io[27])*cp27_64))
	s1[21] = int64(drs16(int64(io[21])*cp11_64 - int64(io[11])*cp21_64))
	s1[26] = int64(drs16(int64(io[21])*cp21_64 + int64(io[11])*cp11_64))
	s1[22] = int64(drs16(int64(io[13])*cp19_64 - int64(io[19])*cp13_64))
	s1[25] = int64(drs16(int64(io[13])*cp13_64 + int64(io[19])*cp19_64))
	s1[23] = int64(drs16(int64(io[29])*cp3_64 - int64(io[3])*cp29_64))
	s1[24] = int64(drs16(int64(io[29])*cp29_64 + int64(io[3])*cp3_64))

	// stage 2
	s2[0], s2[1], s2[2], s2[3] = s1[0], s1[1], s1[2], s1[3]
	s2[4], s2[5], s2[6], s2[7] = s1[4], s1[5], s1[6], s1[7]
	s2[8] = int64(drs16(s1[8]*cp30_64 - s1[15]*cp2_64))
	s2[15] = int64(drs16(s1[8]*cp2_64 + s1[15]*cp30_64))
	s2[9] = int64(drs16(s1[9]*cp14_64 - s1[14]*cp18_64))
	s2[14] = int64(drs16(s1[9]*cp18_64 + s1[14]*cp14_64))
	s2[10] = int64(drs16(s1[10]*cp22_64 - s1[13]*cp10_64))
	s2[13] = int64(drs16(s1[10]*cp10_64 + s1[13]*cp22_64))
	s2[11] = int64(drs16(s1[11]*cp6_64 - s1[12]*cp26_64))
	s2[12] = int64(drs16(s1[11]*cp26_64 + s1[12]*cp6_64))
	s2[16] = wl16(s1[16] + s1[17])
	s2[17] = wl16(s1[16] - s1[17])
	s2[18] = wl16(-s1[18] + s1[19])
	s2[19] = wl16(s1[18] + s1[19])
	s2[20] = wl16(s1[20] + s1[21])
	s2[21] = wl16(s1[20] - s1[21])
	s2[22] = wl16(-s1[22] + s1[23])
	s2[23] = wl16(s1[22] + s1[23])
	s2[24] = wl16(s1[24] + s1[25])
	s2[25] = wl16(s1[24] - s1[25])
	s2[26] = wl16(-s1[26] + s1[27])
	s2[27] = wl16(s1[26] + s1[27])
	s2[28] = wl16(s1[28] + s1[29])
	s2[29] = wl16(s1[28] - s1[29])
	s2[30] = wl16(-s1[30] + s1[31])
	s2[31] = wl16(s1[30] + s1[31])

	// stage 3
	s1[0], s1[1], s1[2], s1[3] = s2[0], s2[1], s2[2], s2[3]
	s1[4] = int64(drs16(s2[4]*cp28_64 - s2[7]*cp4_64))
	s1[7] = int64(drs16(s2[4]*cp4_64 + s2[7]*cp28_64))
	s1[5] = int64(drs16(s2[5]*cp12_64 - s2[6]*cp20_64))
	s1[6] = int64(drs16(s2[5]*cp20_64 + s2[6]*cp12_64))
	s1[8] = wl16(s2[8] + s2[9])
	s1[9] = wl16(s2[8] - s2[9])
	s1[10] = wl16(-s2[10] + s2[11])
	s1[11] = wl16(s2[10] + s2[11])
	s1[12] = wl16(s2[12] + s2[13])
	s1[13] = wl16(s2[12] - s2[13])
	s1[14] = wl16(-s2[14] + s2[15])
	s1[15] = wl16(s2[14] + s2[15])
	s1[16] = s2[16]
	s1[31] = s2[31]
	s1[17] = int64(drs16(-s2[17]*cp4_64 + s2[30]*cp28_64))
	s1[30] = int64(drs16(s2[17]*cp28_64 + s2[30]*cp4_64))
	s1[18] = int64(drs16(-s2[18]*cp28_64 - s2[29]*cp4_64))
	s1[29] = int64(drs16(-s2[18]*cp4_64 + s2[29]*cp28_64))
	s1[19] = s2[19]
	s1[20] = s2[20]
	s1[21] = int64(drs16(-s2[21]*cp20_64 + s2[26]*cp12_64))
	s1[26] = int64(drs16(s2[21]*cp12_64 + s2[26]*cp20_64))
	s1[22] = int64(drs16(-s2[22]*cp12_64 - s2[25]*cp20_64))
	s1[25] = int64(drs16(-s2[22]*cp20_64 + s2[25]*cp12_64))
	s1[23] = s2[23]
	s1[24] = s2[24]
	s1[27] = s2[27]
	s1[28] = s2[28]

	// stage 4
	s2[0] = int64(drs16((s1[0] + s1[1]) * cp16_64))
	s2[1] = int64(drs16((s1[0] - s1[1]) * cp16_64))
	s2[2] = int64(drs16(s1[2]*cp24_64 - s1[3]*cp8_64))
	s2[3] = int64(drs16(s1[2]*cp8_64 + s1[3]*cp24_64))
	s2[4] = wl16(s1[4] + s1[5])
	s2[5] = wl16(s1[4] - s1[5])
	s2[6] = wl16(-s1[6] + s1[7])
	s2[7] = wl16(s1[6] + s1[7])
	s2[8] = s1[8]
	s2[15] = s1[15]
	s2[9] = int64(drs16(-s1[9]*cp8_64 + s1[14]*cp24_64))
	s2[14] = int64(drs16(s1[9]*cp24_64 + s1[14]*cp8_64))
	s2[10] = int64(drs16(-s1[10]*cp24_64 - s1[13]*cp8_64))
	s2[13] = int64(drs16(-s1[10]*cp8_64 + s1[13]*cp24_64))
	s2[11] = s1[11]
	s2[12] = s1[12]
	s2[16] = wl16(s1[16] + s1[19])
	s2[17] = wl16(s1[17] + s1[18])
	s2[18] = wl16(s1[17] - s1[18])
	s2[19] = wl16(s1[16] - s1[19])
	s2[20] = wl16(-s1[20] + s1[23])
	s2[21] = wl16(-s1[21] + s1[22])
	s2[22] = wl16(s1[21] + s1[22])
	s2[23] = wl16(s1[20] + s1[23])
	s2[24] = wl16(s1[24] + s1[27])
	s2[25] = wl16(s1[25] + s1[26])
	s2[26] = wl16(s1[25] - s1[26])
	s2[27] = wl16(s1[24] - s1[27])
	s2[28] = wl16(-s1[28] + s1[31])
	s2[29] = wl16(-s1[29] + s1[30])
	s2[30] = wl16(s1[29] + s1[30])
	s2[31] = wl16(s1[28] + s1[31])

	// stage 5
	s1[0] = wl16(s2[0] + s2[3])
	s1[1] = wl16(s2[1] + s2[2])
	s1[2] = wl16(s2[1] - s2[2])
	s1[3] = wl16(s2[0] - s2[3])
	s1[4] = s2[4]
	s1[5] = int64(drs16((s2[6] - s2[5]) * cp16_64))
	s1[6] = int64(drs16((s2[5] + s2[6]) * cp16_64))
	s1[7] = s2[7]
	s1[8] = wl16(s2[8] + s2[11])
	s1[9] = wl16(s2[9] + s2[10])
	s1[10] = wl16(s2[9] - s2[10])
	s1[11] = wl16(s2[8] - s2[11])
	s1[12] = wl16(-s2[12] + s2[15])
	s1[13] = wl16(-s2[13] + s2[14])
	s1[14] = wl16(s2[13] + s2[14])
	s1[15] = wl16(s2[12] + s2[15])
	s1[16] = s2[16]
	s1[17] = s2[17]
	s1[18] = int64(drs16(-s2[18]*cp8_64 + s2[29]*cp24_64))
	s1[29] = int64(drs16(s2[18]*cp24_64 + s2[29]*cp8_64))
	s1[19] = int64(drs16(-s2[19]*cp8_64 + s2[28]*cp24_64))
	s1[28] = int64(drs16(s2[19]*cp24_64 + s2[28]*cp8_64))
	s1[20] = int64(drs16(-s2[20]*cp24_64 - s2[27]*cp8_64))
	s1[27] = int64(drs16(-s2[20]*cp8_64 + s2[27]*cp24_64))
	s1[21] = int64(drs16(-s2[21]*cp24_64 - s2[26]*cp8_64))
	s1[26] = int64(drs16(-s2[21]*cp8_64 + s2[26]*cp24_64))
	s1[22] = s2[22]
	s1[23] = s2[23]
	s1[24] = s2[24]
	s1[25] = s2[25]
	s1[30] = s2[30]
	s1[31] = s2[31]

	// stage 6
	s2[0] = wl16(s1[0] + s1[7])
	s2[1] = wl16(s1[1] + s1[6])
	s2[2] = wl16(s1[2] + s1[5])
	s2[3] = wl16(s1[3] + s1[4])
	s2[4] = wl16(s1[3] - s1[4])
	s2[5] = wl16(s1[2] - s1[5])
	s2[6] = wl16(s1[1] - s1[6])
	s2[7] = wl16(s1[0] - s1[7])
	s2[8] = s1[8]
	s2[9] = s1[9]
	s2[10] = int64(drs16((-s1[10] + s1[13]) * cp16_64))
	s2[13] = int64(drs16((s1[10] + s1[13]) * cp16_64))
	s2[11] = int64(drs16((-s1[11] + s1[12]) * cp16_64))
	s2[12] = int64(drs16((s1[11] + s1[12]) * cp16_64))
	s2[14] = s1[14]
	s2[15] = s1[15]
	s2[16] = wl16(s1[16] + s1[23])
	s2[17] = wl16(s1[17] + s1[22])
	s2[18] = wl16(s1[18] + s1[21])
	s2[19] = wl16(s1[19] + s1[20])
	s2[20] = wl16(s1[19] - s1[20])
	s2[21] = wl16(s1[18] - s1[21])
	s2[22] = wl16(s1[17] - s1[22])
	s2[23] = wl16(s1[16] - s1[23])
	s2[24] = wl16(-s1[24] + s1[31])
	s2[25] = wl16(-s1[25] + s1[30])
	s2[26] = wl16(-s1[26] + s1[29])
	s2[27] = wl16(-s1[27] + s1[28])
	s2[28] = wl16(s1[27] + s1[28])
	s2[29] = wl16(s1[26] + s1[29])
	s2[30] = wl16(s1[25] + s1[30])
	s2[31] = wl16(s1[24] + s1[31])

	// stage 7
	s1[0] = wl16(s2[0] + s2[15])
	s1[1] = wl16(s2[1] + s2[14])
	s1[2] = wl16(s2[2] + s2[13])
	s1[3] = wl16(s2[3] + s2[12])
	s1[4] = wl16(s2[4] + s2[11])
	s1[5] = wl16(s2[5] + s2[10])
	s1[6] = wl16(s2[6] + s2[9])
	s1[7] = wl16(s2[7] + s2[8])
	s1[8] = wl16(s2[7] - s2[8])
	s1[9] = wl16(s2[6] - s2[9])
	s1[10] = wl16(s2[5] - s2[10])
	s1[11] = wl16(s2[4] - s2[11])
	s1[12] = wl16(s2[3] - s2[12])
	s1[13] = wl16(s2[2] - s2[13])
	s1[14] = wl16(s2[1] - s2[14])
	s1[15] = wl16(s2[0] - s2[15])
	s1[16] = s2[16]
	s1[17] = s2[17]
	s1[18] = s2[18]
	s1[19] = s2[19]
	s1[20] = int64(drs16((-s2[20] + s2[27]) * cp16_64))
	s1[27] = int64(drs16((s2[20] + s2[27]) * cp16_64))
	s1[21] = int64(drs16((-s2[21] + s2[26]) * cp16_64))
	s1[26] = int64(drs16((s2[21] + s2[26]) * cp16_64))
	s1[22] = int64(drs16((-s2[22] + s2[25]) * cp16_64))
	s1[25] = int64(drs16((s2[22] + s2[25]) * cp16_64))
	s1[23] = int64(drs16((-s2[23] + s2[24]) * cp16_64))
	s1[24] = int64(drs16((s2[23] + s2[24]) * cp16_64))
	s1[28] = s2[28]
	s1[29] = s2[29]
	s1[30] = s2[30]
	s1[31] = s2[31]

	// final stage
	io[0] = wl16(s1[0] + s1[31])
	io[1] = wl16(s1[1] + s1[30])
	io[2] = wl16(s1[2] + s1[29])
	io[3] = wl16(s1[3] + s1[28])
	io[4] = wl16(s1[4] + s1[27])
	io[5] = wl16(s1[5] + s1[26])
	io[6] = wl16(s1[6] + s1[25])
	io[7] = wl16(s1[7] + s1[24])
	io[8] = wl16(s1[8] + s1[23])
	io[9] = wl16(s1[9] + s1[22])
	io[10] = wl16(s1[10] + s1[21])
	io[11] = wl16(s1[11] + s1[20])
	io[12] = wl16(s1[12] + s1[19])
	io[13] = wl16(s1[13] + s1[18])
	io[14] = wl16(s1[14] + s1[17])
	io[15] = wl16(s1[15] + s1[16])
	io[16] = wl16(s1[15] - s1[16])
	io[17] = wl16(s1[14] - s1[17])
	io[18] = wl16(s1[13] - s1[18])
	io[19] = wl16(s1[12] - s1[19])
	io[20] = wl16(s1[11] - s1[20])
	io[21] = wl16(s1[10] - s1[21])
	io[22] = wl16(s1[9] - s1[22])
	io[23] = wl16(s1[8] - s1[23])
	io[24] = wl16(s1[7] - s1[24])
	io[25] = wl16(s1[6] - s1[25])
	io[26] = wl16(s1[5] - s1[26])
	io[27] = wl16(s1[4] - s1[27])
	io[28] = wl16(s1[3] - s1[28])
	io[29] = wl16(s1[2] - s1[29])
	io[30] = wl16(s1[1] - s1[30])
	io[31] = wl16(s1[0] - s1[31])
}
