// Package transform implements VP9 inverse transforms (spec §8.7.1).
// Sizes 4/8/16/32; kernels DCT-II and ADST (no 32×32 ADST in VP9).
// Integer arithmetic: constants scaled by 2^14; drs = (v+(1<<13))>>14.
// 2-D pass: rows → >>4 → transpose → cols → >>4 → transpose.
package transform

// TxType selects the 1-D kernel pair (row kernel, col kernel).
type TxType int

const (
	DCT_DCT   TxType = 0
	ADST_DCT  TxType = 1
	DCT_ADST  TxType = 2
	ADST_ADST TxType = 3
)

// InvTransform2D applies the inverse 2-D transform in-place.
// coeffs is bs×bs int16, row-major. bs ∈ {4,8,16,32}; bs=32 forces DCT_DCT.
func InvTransform2D(coeffs []int16, bs int, tt TxType) {
	if bs == 32 {
		tt = DCT_DCT
	}
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
				v4 := DCOnlyValue(coeffs[0])
				for i := range coeffs {
					coeffs[i] = v4
				}
			}
			return
		}
	}
	rowDCT := tt == DCT_DCT || tt == DCT_ADST
	colDCT := tt == DCT_DCT || tt == ADST_DCT
	for i := 0; i < bs; i++ {
		inv1D(coeffs[i*bs:i*bs+bs], bs, rowDCT)
	}
	transposeAndRoundShift4(coeffs, bs)
	for i := 0; i < bs; i++ {
		inv1D(coeffs[i*bs:i*bs+bs], bs, colDCT)
	}
	transposeAndRoundShift4(coeffs, bs)
}

func inv1D(row []int16, bs int, dct bool) {
	if dct {
		switch bs {
		case 4:
			idct4(row)
		case 8:
			idct8(row)
		case 16:
			idct16(row)
		case 32:
			idct32(row)
		}
	} else {
		switch bs {
		case 4:
			iadst4(row)
		case 8:
			iadst8(row)
		case 16:
			iadst16(row)
		}
	}
}

// transposeAndRoundShift4 combines roundShift4 and transpose into one pass,
// halving the number of memory reads/writes compared to calling them separately.
func transposeAndRoundShift4(buf []int16, bs int) {
	for r := 0; r < bs; r++ {
		buf[r*bs+r] = int16((int32(buf[r*bs+r]) + 8) >> 4)
		for c := r + 1; c < bs; c++ {
			a := int16((int32(buf[r*bs+c]) + 8) >> 4)
			b := int16((int32(buf[c*bs+r]) + 8) >> 4)
			buf[r*bs+c] = b
			buf[c*bs+r] = a
		}
	}
}

// DCOnlyValue returns the uniform output value of a DCT_DCT inverse 2-D
// transform when only the DC coefficient (after dequantization) is non-zero.
// Identical for all block sizes 4×4 … 32×32.
func DCOnlyValue(dc int16) int16 {
	v1 := drs(int32(dc) * cp16)
	v2 := int16((int32(v1) + 8) >> 4)
	v3 := drs(int32(v2) * cp16)
	return int16((int32(v3) + 8) >> 4)
}

// drs = dct_const_round_shift: (v + (1<<13)) >> 14
func drs(v int32) int16 { return int16((v + (1 << 13)) >> 14) }

// cos(n·π/64)·2^14 rounded (spec Table 8-12)
const (
	cp1  int32 = 16364
	cp2  int32 = 16305
	cp3  int32 = 16207
	cp4  int32 = 16069
	cp5  int32 = 15893
	cp6  int32 = 15679
	cp7  int32 = 15426
	cp8  int32 = 15137
	cp9  int32 = 14811
	cp10 int32 = 14449
	cp11 int32 = 14053
	cp12 int32 = 13623
	cp13 int32 = 13160
	cp14 int32 = 12665
	cp15 int32 = 12140
	cp16 int32 = 11585
	cp17 int32 = 11003
	cp18 int32 = 10394
	cp19 int32 = 9760
	cp20 int32 = 9102
	cp21 int32 = 8423
	cp22 int32 = 7723
	cp23 int32 = 7005
	cp24 int32 = 6270
	cp25 int32 = 5520
	cp26 int32 = 4756
	cp27 int32 = 3981
	cp28 int32 = 3196
	cp29 int32 = 2404
	cp30 int32 = 1606
	cp31 int32 = 804
)

// sin(n·π/9)·2^14 rounded (ADST-4, spec §8.7.1.2)
const (
	sp1 int32 = 5283
	sp2 int32 = 9929
	sp3 int32 = 13377
	sp4 int32 = 15212
)

// ---- 4-point inverse DCT (spec §8.7.1.1) ----

func idct4(io []int16) {
	a := (int32(io[0]) + int32(io[2])) * cp16
	b := (int32(io[0]) - int32(io[2])) * cp16
	c := int32(io[1])*cp24 - int32(io[3])*cp8
	d := int32(io[1])*cp8 + int32(io[3])*cp24
	io[0] = drs(a + d)
	io[1] = drs(b + c)
	io[2] = drs(b - c)
	io[3] = drs(a - d)
}

// ---- 4-point inverse ADST (spec §8.7.1.2) ----

func iadst4(io []int16) {
	s0 := sp1 * int32(io[0])
	s1 := sp2 * int32(io[0])
	s2 := sp3 * int32(io[1])
	s3 := sp4 * int32(io[2])
	s4 := sp1 * int32(io[2])
	s5 := sp2 * int32(io[3])
	s6 := sp4 * int32(io[3])
	s7 := int32(io[0]) - int32(io[2]) + int32(io[3])
	x0 := s0 + s3 + s5
	x1 := s1 - s4 - s6
	x2 := sp3 * s7
	x3 := s2
	io[0] = drs(x0 + x3)
	io[1] = drs(x1 + x3)
	io[2] = drs(x2)
	io[3] = drs(x0 + x1 - x3)
}

// ---- 8-point inverse DCT (spec §8.7.1.3 / libvpx vp9_idct8_c) ----

func idct8(io []int16) {
	// stage 1: butterflies on the 4 odd positions
	s4 := drs(int32(io[1])*cp28 - int32(io[7])*cp4)
	s7 := drs(int32(io[1])*cp4 + int32(io[7])*cp28)
	s5 := drs(int32(io[5])*cp12 - int32(io[3])*cp20)
	s6 := drs(int32(io[5])*cp20 + int32(io[3])*cp12)

	// stage 2: even half via cospi16 / cospi8,24
	e0 := drs((int32(io[0]) + int32(io[4])) * cp16)
	e1 := drs((int32(io[0]) - int32(io[4])) * cp16)
	e2 := drs(int32(io[2])*cp24 - int32(io[6])*cp8)
	e3 := drs(int32(io[2])*cp8 + int32(io[6])*cp24)
	// combine odds
	o4 := int16(int32(s4) + int32(s5))
	o5 := int16(int32(s4) - int32(s5))
	o6 := int16(-int32(s6) + int32(s7))
	o7 := int16(int32(s6) + int32(s7))

	// stage 3: even butterfly + cospi16 cross
	f0 := int16(int32(e0) + int32(e3))
	f1 := int16(int32(e1) + int32(e2))
	f2 := int16(int32(e1) - int32(e2))
	f3 := int16(int32(e0) - int32(e3))
	f5 := drs((int32(o6) - int32(o5)) * cp16)
	f6 := drs((int32(o6) + int32(o5)) * cp16)

	// stage 4: output
	io[0] = int16(int32(f0) + int32(o7))
	io[1] = int16(int32(f1) + int32(f6))
	io[2] = int16(int32(f2) + int32(f5))
	io[3] = int16(int32(f3) + int32(o4))
	io[4] = int16(int32(f3) - int32(o4))
	io[5] = int16(int32(f2) - int32(f5))
	io[6] = int16(int32(f1) - int32(f6))
	io[7] = int16(int32(f0) - int32(o7))
}

// ---- 8-point inverse ADST (spec §8.7.1.4 / libvpx vp9_iadst8_c) ----

func iadst8(io []int16) {
	x0 := int32(io[7])
	x1 := int32(io[0])
	x2 := int32(io[5])
	x3 := int32(io[2])
	x4 := int32(io[3])
	x5 := int32(io[4])
	x6 := int32(io[1])
	x7 := int32(io[6])

	s0 := drs(cp2*x0 - cp30*x1)
	s1 := drs(cp30*x0 + cp2*x1)
	s2 := drs(cp10*x2 - cp22*x3)
	s3 := drs(cp22*x2 + cp10*x3)
	s4 := drs(cp18*x4 - cp14*x5)
	s5 := drs(cp14*x4 + cp18*x5)
	s6 := drs(cp26*x6 - cp6*x7)
	s7 := drs(cp6*x6 + cp26*x7)

	x0 = int32(s0) + int32(s4)
	x1 = int32(s1) + int32(s5)
	x2 = int32(s2) + int32(s6)
	x3 = int32(s3) + int32(s7)
	x4 = int32(s0) - int32(s4)
	x5 = int32(s1) - int32(s5)
	x6 = int32(s2) - int32(s6)
	x7 = int32(s3) - int32(s7)

	t0 := int16(x0 + x2)
	t1 := int16(x1 + x3)
	t2 := drs(cp16 * (x4 - x5))
	t3 := drs(cp16 * (x4 + x5))
	t4 := int16(x0 - x2)
	t5 := int16(x1 - x3)
	t6 := drs(cp16 * (x6 - x7))
	t7 := drs(cp16 * (x6 + x7))

	io[0] = int16(int32(t0) + int32(t1))
	io[1] = int16(int32(t4) + int32(t5))
	io[2] = int16(int32(t2) + int32(t3))
	io[3] = int16(int32(t6) + int32(t7))
	io[4] = int16(int32(t4) - int32(t5))
	io[5] = int16(int32(t0) - int32(t1))
	io[6] = int16(int32(t6) - int32(t7))
	io[7] = int16(int32(t2) - int32(t3))
}

// ---- 16-point inverse DCT (spec §8.7.1.5 / libvpx vp9_idct16_c) ----

func idct16(io []int16) {
	// stage 1: 8 butterflies on odd positions 1,3,5,7,9,11,13,15
	s0 := drs(int32(io[1])*cp30 - int32(io[15])*cp2)
	s1 := drs(int32(io[1])*cp2 + int32(io[15])*cp30)
	s2 := drs(int32(io[9])*cp22 - int32(io[7])*cp10)
	s3 := drs(int32(io[9])*cp10 + int32(io[7])*cp22)
	s4 := drs(int32(io[5])*cp14 - int32(io[11])*cp18)
	s5 := drs(int32(io[5])*cp18 + int32(io[11])*cp14)
	s6 := drs(int32(io[13])*cp6 - int32(io[3])*cp26)
	s7 := drs(int32(io[13])*cp26 + int32(io[3])*cp6)

	// stage 2: pair-combine odds
	t0 := int16(int32(s0) + int32(s4))
	t1 := int16(int32(s1) + int32(s5))
	t2 := int16(int32(s2) + int32(s6))
	t3 := int16(int32(s3) + int32(s7))
	t4 := int16(int32(s0) - int32(s4))
	t5 := int16(int32(s1) - int32(s5))
	t6 := int16(int32(s2) - int32(s6))
	t7 := int16(int32(s3) - int32(s7))

	// stage 3
	u0 := int16(int32(t0) + int32(t2))
	u1 := int16(int32(t1) + int32(t3))
	u2 := int16(int32(t0) - int32(t2))
	u3 := int16(int32(t1) - int32(t3))
	u4 := drs(int32(t4)*cp8 - int32(t5)*cp24)
	u5 := drs(int32(t4)*cp24 + int32(t5)*cp8)
	u6 := drs(-int32(t6)*cp8 + int32(t7)*cp24)
	u7 := drs(int32(t6)*cp24 + int32(t7)*cp8)

	// stage 4: cross-combine
	v0 := int16(int32(u0) + int32(u1))
	v1 := int16(int32(u0) - int32(u1))
	v2 := drs(int32(u2)*cp16 - int32(u3)*cp16)
	v3 := drs(int32(u2)*cp16 + int32(u3)*cp16)
	v4 := int16(int32(u4) + int32(u6))
	v5 := int16(int32(u5) + int32(u7))
	v6 := int16(int32(u4) - int32(u6))
	v7 := int16(int32(u5) - int32(u7))

	// even half: 8-point DCT on positions 0,2,4,6,8,10,12,14
	e := [8]int16{io[0], io[2], io[4], io[6], io[8], io[10], io[12], io[14]}
	idct8(e[:])

	io[0] = int16(int32(e[0]) + int32(v0))
	io[1] = int16(int32(e[1]) + int32(v4))
	io[2] = int16(int32(e[2]) + int32(v2))
	io[3] = int16(int32(e[3]) + int32(v6))
	io[4] = int16(int32(e[4]) + int32(v7))
	io[5] = int16(int32(e[5]) + int32(v3))
	io[6] = int16(int32(e[6]) + int32(v5))
	io[7] = int16(int32(e[7]) + int32(v1))
	io[8] = int16(int32(e[7]) - int32(v1))
	io[9] = int16(int32(e[6]) - int32(v5))
	io[10] = int16(int32(e[5]) - int32(v3))
	io[11] = int16(int32(e[4]) - int32(v7))
	io[12] = int16(int32(e[3]) - int32(v6))
	io[13] = int16(int32(e[2]) - int32(v2))
	io[14] = int16(int32(e[1]) - int32(v4))
	io[15] = int16(int32(e[0]) - int32(v0))
}

// ---- 16-point inverse ADST (spec §8.7.1.6 / libvpx vp9_iadst16_c) ----

func iadst16(io []int16) {
	x0 := int32(io[15])
	x1 := int32(io[0])
	x2 := int32(io[13])
	x3 := int32(io[2])
	x4 := int32(io[11])
	x5 := int32(io[4])
	x6 := int32(io[9])
	x7 := int32(io[6])
	x8 := int32(io[7])
	x9 := int32(io[8])
	x10 := int32(io[5])
	x11 := int32(io[10])
	x12 := int32(io[3])
	x13 := int32(io[12])
	x14 := int32(io[1])
	x15 := int32(io[14])

	// stage 1
	s0 := drs(cp1*x0 + cp31*x1)
	s1 := drs(cp31*x0 - cp1*x1)
	s2 := drs(cp5*x2 + cp27*x3)
	s3 := drs(cp27*x2 - cp5*x3)
	s4 := drs(cp9*x4 + cp23*x5)
	s5 := drs(cp23*x4 - cp9*x5)
	s6 := drs(cp13*x6 + cp19*x7)
	s7 := drs(cp19*x6 - cp13*x7)
	s8 := drs(cp17*x8 + cp15*x9)
	s9 := drs(cp15*x8 - cp17*x9)
	s10 := drs(cp21*x10 + cp11*x11)
	s11 := drs(cp11*x10 - cp21*x11)
	s12 := drs(cp25*x12 + cp7*x13)
	s13 := drs(cp7*x12 - cp25*x13)
	s14 := drs(cp29*x14 + cp3*x15)
	s15 := drs(cp3*x14 - cp29*x15)

	// stage 2
	x0 = int32(s0) + int32(s8)
	x1 = int32(s1) + int32(s9)
	x2 = int32(s2) + int32(s10)
	x3 = int32(s3) + int32(s11)
	x4 = int32(s4) + int32(s12)
	x5 = int32(s5) + int32(s13)
	x6 = int32(s6) + int32(s14)
	x7 = int32(s7) + int32(s15)
	x8 = int32(s0) - int32(s8)
	x9 = int32(s1) - int32(s9)
	x10 = int32(s2) - int32(s10)
	x11 = int32(s3) - int32(s11)
	x12 = int32(s4) - int32(s12)
	x13 = int32(s5) - int32(s13)
	x14 = int32(s6) - int32(s14)
	x15 = int32(s7) - int32(s15)

	// stage 3
	t0 := int16(x0 + x4)
	t1 := int16(x1 + x5)
	t2 := int16(x2 + x6)
	t3 := int16(x3 + x7)
	t4 := int16(x0 - x4)
	t5 := int16(x1 - x5)
	t6 := int16(x2 - x6)
	t7 := int16(x3 - x7)
	t8 := drs(x8*cp4 - x9*cp28)
	t9 := drs(x8*cp28 + x9*cp4)
	t10 := drs(-x10*cp28 + x11*cp4)
	t11 := drs(x10*cp4 + x11*cp28)
	t12 := drs(x12*cp20 - x13*cp12)
	t13 := drs(x12*cp12 + x13*cp20)
	t14 := drs(-x14*cp12 + x15*cp20)
	t15 := drs(x14*cp20 + x15*cp12)

	// stage 4
	u0 := int16(int32(t0) + int32(t2))
	u1 := int16(int32(t1) + int32(t3))
	u2 := int16(int32(t0) - int32(t2))
	u3 := int16(int32(t1) - int32(t3))
	u4 := int16(int32(t4) + int32(t6))
	u5 := int16(int32(t5) + int32(t7))
	u6 := int16(int32(t4) - int32(t6))
	u7 := int16(int32(t5) - int32(t7))
	u8 := int16(int32(t8) + int32(t10))
	u9 := int16(int32(t9) + int32(t11))
	u10 := int16(int32(t8) - int32(t10))
	u11 := int16(int32(t9) - int32(t11))
	u12 := int16(int32(t12) + int32(t14))
	u13 := int16(int32(t13) + int32(t15))
	u14 := int16(int32(t12) - int32(t14))
	u15 := int16(int32(t13) - int32(t15))

	// stage 5: 4 groups of 4; each group: cos16 butterfly on (u{k},u{k+1}),
	// cos8/24 butterfly on (u{k+2},u{k+3}). Output permutation per spec §8.7.1.6.
	var out [16]int16
	out[0] = drs(cp16 * (int32(u0) + int32(u1)))
	out[1] = drs(cp16 * (int32(u0) - int32(u1)))
	out[2] = drs(int32(u2)*cp8 - int32(u3)*cp24)
	out[3] = drs(int32(u2)*cp24 + int32(u3)*cp8)
	out[4] = drs(cp16 * (int32(u4) + int32(u5)))
	out[5] = drs(cp16 * (int32(u4) - int32(u5)))
	out[6] = drs(int32(u6)*cp8 - int32(u7)*cp24)
	out[7] = drs(int32(u6)*cp24 + int32(u7)*cp8)
	out[8] = drs(cp16 * (int32(u8) + int32(u9)))
	out[9] = drs(cp16 * (int32(u8) - int32(u9)))
	out[10] = drs(int32(u10)*cp8 - int32(u11)*cp24)
	out[11] = drs(int32(u10)*cp24 + int32(u11)*cp8)
	out[12] = drs(cp16 * (int32(u12) + int32(u13)))
	out[13] = drs(cp16 * (int32(u12) - int32(u13)))
	out[14] = drs(int32(u14)*cp8 - int32(u15)*cp24)
	out[15] = drs(int32(u14)*cp24 + int32(u15)*cp8)
	copy(io[:16], out[:])
}

// ---- 32-point inverse DCT (spec §8.7.1.7 / libvpx vp9_idct32_c) ----

func idct32(io []int16) {
	// stage 1: 16 butterflies on the 16 odd positions
	s0 := drs(int32(io[1])*cp31 - int32(io[31])*cp1)
	s1 := drs(int32(io[1])*cp1 + int32(io[31])*cp31)
	s2 := drs(int32(io[17])*cp15 - int32(io[15])*cp17)
	s3 := drs(int32(io[17])*cp17 + int32(io[15])*cp15)
	s4 := drs(int32(io[9])*cp23 - int32(io[23])*cp9)
	s5 := drs(int32(io[9])*cp9 + int32(io[23])*cp23)
	s6 := drs(int32(io[25])*cp7 - int32(io[7])*cp25)
	s7 := drs(int32(io[25])*cp25 + int32(io[7])*cp7)
	s8 := drs(int32(io[5])*cp27 - int32(io[27])*cp5)
	s9 := drs(int32(io[5])*cp5 + int32(io[27])*cp27)
	s10 := drs(int32(io[21])*cp11 - int32(io[11])*cp21)
	s11 := drs(int32(io[21])*cp21 + int32(io[11])*cp11)
	s12 := drs(int32(io[13])*cp19 - int32(io[19])*cp13)
	s13 := drs(int32(io[13])*cp13 + int32(io[19])*cp19)
	s14 := drs(int32(io[29])*cp3 - int32(io[3])*cp29)
	s15 := drs(int32(io[29])*cp29 + int32(io[3])*cp3)

	// stage 2
	t0 := int16(int32(s0) + int32(s8))
	t1 := int16(int32(s1) + int32(s9))
	t2 := int16(int32(s2) + int32(s10))
	t3 := int16(int32(s3) + int32(s11))
	t4 := int16(int32(s4) + int32(s12))
	t5 := int16(int32(s5) + int32(s13))
	t6 := int16(int32(s6) + int32(s14))
	t7 := int16(int32(s7) + int32(s15))
	t8 := int16(int32(s0) - int32(s8))
	t9 := int16(int32(s1) - int32(s9))
	t10 := int16(int32(s2) - int32(s10))
	t11 := int16(int32(s3) - int32(s11))
	t12 := int16(int32(s4) - int32(s12))
	t13 := int16(int32(s5) - int32(s13))
	t14 := int16(int32(s6) - int32(s14))
	t15 := int16(int32(s7) - int32(s15))

	// stage 3
	u0 := int16(int32(t0) + int32(t4))
	u1 := int16(int32(t1) + int32(t5))
	u2 := int16(int32(t2) + int32(t6))
	u3 := int16(int32(t3) + int32(t7))
	u4 := int16(int32(t0) - int32(t4))
	u5 := int16(int32(t1) - int32(t5))
	u6 := int16(int32(t2) - int32(t6))
	u7 := int16(int32(t3) - int32(t7))
	u8 := drs(int32(t8)*cp4 - int32(t9)*cp28)
	u9 := drs(int32(t8)*cp28 + int32(t9)*cp4)
	u10 := drs(-int32(t10)*cp28 + int32(t11)*cp4)
	u11 := drs(int32(t10)*cp4 + int32(t11)*cp28)
	u12 := drs(int32(t12)*cp20 - int32(t13)*cp12)
	u13 := drs(int32(t12)*cp12 + int32(t13)*cp20)
	u14 := drs(-int32(t14)*cp12 + int32(t15)*cp20)
	u15 := drs(int32(t14)*cp20 + int32(t15)*cp12)

	// stage 4
	v0 := int16(int32(u0) + int32(u2))
	v1 := int16(int32(u1) + int32(u3))
	v2 := int16(int32(u0) - int32(u2))
	v3 := int16(int32(u1) - int32(u3))
	v4 := int16(int32(u4) + int32(u6))
	v5 := int16(int32(u5) + int32(u7))
	v6 := int16(int32(u4) - int32(u6))
	v7 := int16(int32(u5) - int32(u7))
	v8 := int16(int32(u8) + int32(u10))
	v9 := int16(int32(u9) + int32(u11))
	v10 := int16(int32(u8) - int32(u10))
	v11 := int16(int32(u9) - int32(u11))
	v12 := int16(int32(u12) + int32(u14))
	v13 := int16(int32(u13) + int32(u15))
	v14 := int16(int32(u12) - int32(u14))
	v15 := int16(int32(u13) - int32(u15))

	// stage 5: cospi16 butterflies + cospi8/24
	w0 := drs(cp16 * (int32(v0) + int32(v1)))
	w1 := drs(cp16 * (int32(v0) - int32(v1)))
	w2 := drs(int32(v2)*cp8 - int32(v3)*cp24)
	w3 := drs(int32(v2)*cp24 + int32(v3)*cp8)
	w4 := drs(cp16 * (int32(v4) + int32(v5)))
	w5 := drs(cp16 * (int32(v4) - int32(v5)))
	w6 := drs(int32(v6)*cp8 - int32(v7)*cp24)
	w7 := drs(int32(v6)*cp24 + int32(v7)*cp8)
	w8 := drs(cp16 * (int32(v8) + int32(v9)))
	w9 := drs(cp16 * (int32(v8) - int32(v9)))
	w10 := drs(int32(v10)*cp8 - int32(v11)*cp24)
	w11 := drs(int32(v10)*cp24 + int32(v11)*cp8)
	w12 := drs(cp16 * (int32(v12) + int32(v13)))
	w13 := drs(cp16 * (int32(v12) - int32(v13)))
	w14 := drs(int32(v14)*cp8 - int32(v15)*cp24)
	w15 := drs(int32(v14)*cp24 + int32(v15)*cp8)

	odd := [16]int16{w0, w3, w1, w2, w4, w7, w5, w6, w8, w11, w9, w10, w12, w15, w13, w14}

	// even half: 16-point DCT on even positions
	even := [16]int16{
		io[0], io[2], io[4], io[6], io[8], io[10], io[12], io[14],
		io[16], io[18], io[20], io[22], io[24], io[26], io[28], io[30],
	}
	idct16(even[:])

	for i := 0; i < 16; i++ {
		io[i] = int16(int32(even[i]) + int32(odd[i]))
		io[31-i] = int16(int32(even[i]) - int32(odd[i]))
	}
}
