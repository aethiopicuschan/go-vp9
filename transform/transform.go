// Package transform implements VP9 inverse transforms (spec §8.7.1).
// Sizes 4/8/16/32; kernels DCT-II and ADST (no 32×32 ADST in VP9).
// Integer arithmetic: constants scaled by 2^14; drs = (v+(1<<13))>>14.
// 2-D pass (libvpx convention): rows (no shift) → transpose → cols → transpose
// with a single ROUND_POWER_OF_TWO of FinalShift(bs) = 4/5/6/6 for bs 4/8/16/32.
package transform

// TxType selects the 1-D kernel pair (row kernel, col kernel).
type TxType int

const (
	DCT_DCT   TxType = 0
	ADST_DCT  TxType = 1
	DCT_ADST  TxType = 2
	ADST_ADST TxType = 3
)

// FinalShift returns the VP9 inverse-transform output round-shift for a block
// of side bs pixels: 4 for 4×4, 5 for 8×8, 6 for 16×16 and 32×32 (libvpx
// vpx_idctNxN, ROUND_POWER_OF_TWO at the end of the column pass).
func FinalShift(bs int) int {
	switch bs {
	case 4:
		return 4
	case 8:
		return 5
	default: // 16, 32
		return 6
	}
}

// InvTransform2D applies the inverse 2-D transform in-place.
// coeffs is bs×bs int16, row-major. bs ∈ {4,8,16,32}; bs=32 forces DCT_DCT.
//
// Convention matches libvpx: the row pass writes 16-bit (wrapping) intermediates
// with NO shift; the column pass is followed by a single ROUND_POWER_OF_TWO of
// FinalShift(bs). (An earlier version shifted >>4 after each pass, scaling the
// residual 16× too small.)
func InvTransform2D(coeffs []int16, bs int, tt TxType) {
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
				v4 := DCOnlyValue(coeffs[0], shift)
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
		inv1D(coeffs[i*bs:i*bs+bs], bs, rowDCT)
	}
	transpose(coeffs, bs)
	for i := 0; i < bs; i++ {
		inv1D(coeffs[i*bs:i*bs+bs], bs, colDCT)
	}
	transposeAndShift(coeffs, bs, shift)
}

// transpose transposes a bs×bs int16 matrix in place (no rounding).
func transpose(buf []int16, bs int) {
	for r := 0; r < bs; r++ {
		for c := r + 1; c < bs; c++ {
			buf[r*bs+c], buf[c*bs+r] = buf[c*bs+r], buf[r*bs+c]
		}
	}
}

// transposeAndShift transposes and applies ROUND_POWER_OF_TWO(_, n) to every
// element (the final column-pass round-shift).
func transposeAndShift(buf []int16, bs, n int) {
	round := int32(1) << (n - 1)
	rs := func(v int16) int16 { return int16((int32(v) + round) >> n) }
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

// DCOnlyValue returns the uniform output value of a DCT_DCT inverse 2-D
// transform when only the DC coefficient (after dequantization) is non-zero.
// shift is FinalShift(bs) for the block (libvpx vpx_idctNxN_1_add: two cascaded
// cospi_16 round-shifts, then ROUND_POWER_OF_TWO(_, shift)).
func DCOnlyValue(dc int16, shift int) int16 {
	v := drs(int32(dc) * cp16)
	v = drs(int32(v) * cp16)
	round := int32(1) << (shift - 1)
	return int16((int32(v) + round) >> shift)
}

// drs = dct_const_round_shift: (v + (1<<13)) >> 14
func drs(v int32) int16 { return int16((v + (1 << 13)) >> 14) }

// wl mirrors libvpx WRAPLOW: truncate to int16 (the transform intermediates
// wrap at 16 bits in the non-high-bit-depth build) and sign-extend back.
func wl(v int32) int32 { return int32(int16(v)) }

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

// idct4 ports libvpx idct4_c: each stage-1 rotation is round-shifted (drs) and
// truncated to int16 BEFORE the stage-2 butterfly add — rounding once per term,
// not once on the summed product (the latter is off by ±1 on some inputs).
func idct4(io []int16) {
	s0 := int32(drs((int32(io[0]) + int32(io[2])) * cp16))
	s1 := int32(drs((int32(io[0]) - int32(io[2])) * cp16))
	s2 := int32(drs(int32(io[1])*cp24 - int32(io[3])*cp8))
	s3 := int32(drs(int32(io[1])*cp8 + int32(io[3])*cp24))
	io[0] = int16(wl(s0 + s3))
	io[1] = int16(wl(s1 + s2))
	io[2] = int16(wl(s1 - s2))
	io[3] = int16(wl(s0 - s3))
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

// iadst8 is the 8-point inverse ADST, a line-for-line port of libvpx
// iadst8_c (vpx_dsp/inv_txfm.c): 3 butterfly stages with WRAPLOW (wl) and
// dct_const_round_shift (drs), then the signed output permutation.
func iadst8(io []int16) {
	x0 := int32(io[7])
	x1 := int32(io[0])
	x2 := int32(io[5])
	x3 := int32(io[2])
	x4 := int32(io[3])
	x5 := int32(io[4])
	x6 := int32(io[1])
	x7 := int32(io[6])

	if x0|x1|x2|x3|x4|x5|x6|x7 == 0 {
		for i := 0; i < 8; i++ {
			io[i] = 0
		}
		return
	}

	// stage 1
	s0 := cp2*x0 + cp30*x1
	s1 := cp30*x0 - cp2*x1
	s2 := cp10*x2 + cp22*x3
	s3 := cp22*x2 - cp10*x3
	s4 := cp18*x4 + cp14*x5
	s5 := cp14*x4 - cp18*x5
	s6 := cp26*x6 + cp6*x7
	s7 := cp6*x6 - cp26*x7

	x0 = int32(drs(s0 + s4))
	x1 = int32(drs(s1 + s5))
	x2 = int32(drs(s2 + s6))
	x3 = int32(drs(s3 + s7))
	x4 = int32(drs(s0 - s4))
	x5 = int32(drs(s1 - s5))
	x6 = int32(drs(s2 - s6))
	x7 = int32(drs(s3 - s7))

	// stage 2
	s0 = x0
	s1 = x1
	s2 = x2
	s3 = x3
	s4 = cp8*x4 + cp24*x5
	s5 = cp24*x4 - cp8*x5
	s6 = -cp24*x6 + cp8*x7
	s7 = cp8*x6 + cp24*x7

	x0 = wl(s0 + s2)
	x1 = wl(s1 + s3)
	x2 = wl(s0 - s2)
	x3 = wl(s1 - s3)
	x4 = int32(drs(s4 + s6))
	x5 = int32(drs(s5 + s7))
	x6 = int32(drs(s4 - s6))
	x7 = int32(drs(s5 - s7))

	// stage 3
	s2 = cp16 * (x2 + x3)
	s3 = cp16 * (x2 - x3)
	s6 = cp16 * (x6 + x7)
	s7 = cp16 * (x6 - x7)

	x2 = int32(drs(s2))
	x3 = int32(drs(s3))
	x6 = int32(drs(s6))
	x7 = int32(drs(s7))

	io[0] = int16(x0)
	io[1] = int16(-x4)
	io[2] = int16(x6)
	io[3] = int16(-x2)
	io[4] = int16(x3)
	io[5] = int16(-x7)
	io[6] = int16(x5)
	io[7] = int16(-x1)
}

// ---- 16-point inverse DCT (spec §8.7.1.5 / libvpx vp9_idct16_c) ----

// idct16 is a line-for-line port of libvpx idct16_c (vpx_dsp/inv_txfm.c).
// All step values are kept in int16 range via wl (WRAPLOW); drs is
// dct_const_round_shift. s1/s2 mirror libvpx step1/step2.
func idct16(io []int16) {
	var s1, s2 [16]int32

	// stage 1
	s1[0] = int32(io[0])
	s1[1] = int32(io[8])
	s1[2] = int32(io[4])
	s1[3] = int32(io[12])
	s1[4] = int32(io[2])
	s1[5] = int32(io[10])
	s1[6] = int32(io[6])
	s1[7] = int32(io[14])
	s1[8] = int32(io[1])
	s1[9] = int32(io[9])
	s1[10] = int32(io[5])
	s1[11] = int32(io[13])
	s1[12] = int32(io[3])
	s1[13] = int32(io[11])
	s1[14] = int32(io[7])
	s1[15] = int32(io[15])

	// stage 2
	s2[0], s2[1], s2[2], s2[3] = s1[0], s1[1], s1[2], s1[3]
	s2[4], s2[5], s2[6], s2[7] = s1[4], s1[5], s1[6], s1[7]
	s2[8] = int32(drs(s1[8]*cp30 - s1[15]*cp2))
	s2[15] = int32(drs(s1[8]*cp2 + s1[15]*cp30))
	s2[9] = int32(drs(s1[9]*cp14 - s1[14]*cp18))
	s2[14] = int32(drs(s1[9]*cp18 + s1[14]*cp14))
	s2[10] = int32(drs(s1[10]*cp22 - s1[13]*cp10))
	s2[13] = int32(drs(s1[10]*cp10 + s1[13]*cp22))
	s2[11] = int32(drs(s1[11]*cp6 - s1[12]*cp26))
	s2[12] = int32(drs(s1[11]*cp26 + s1[12]*cp6))

	// stage 3
	s1[0], s1[1], s1[2], s1[3] = s2[0], s2[1], s2[2], s2[3]
	s1[4] = int32(drs(s2[4]*cp28 - s2[7]*cp4))
	s1[7] = int32(drs(s2[4]*cp4 + s2[7]*cp28))
	s1[5] = int32(drs(s2[5]*cp12 - s2[6]*cp20))
	s1[6] = int32(drs(s2[5]*cp20 + s2[6]*cp12))
	s1[8] = wl(s2[8] + s2[9])
	s1[9] = wl(s2[8] - s2[9])
	s1[10] = wl(-s2[10] + s2[11])
	s1[11] = wl(s2[10] + s2[11])
	s1[12] = wl(s2[12] + s2[13])
	s1[13] = wl(s2[12] - s2[13])
	s1[14] = wl(-s2[14] + s2[15])
	s1[15] = wl(s2[14] + s2[15])

	// stage 4
	s2[0] = int32(drs((s1[0] + s1[1]) * cp16))
	s2[1] = int32(drs((s1[0] - s1[1]) * cp16))
	s2[2] = int32(drs(s1[2]*cp24 - s1[3]*cp8))
	s2[3] = int32(drs(s1[2]*cp8 + s1[3]*cp24))
	s2[4] = wl(s1[4] + s1[5])
	s2[5] = wl(s1[4] - s1[5])
	s2[6] = wl(-s1[6] + s1[7])
	s2[7] = wl(s1[6] + s1[7])
	s2[8], s2[15] = s1[8], s1[15]
	s2[9] = int32(drs(-s1[9]*cp8 + s1[14]*cp24))
	s2[14] = int32(drs(s1[9]*cp24 + s1[14]*cp8))
	s2[10] = int32(drs(-s1[10]*cp24 - s1[13]*cp8))
	s2[13] = int32(drs(-s1[10]*cp8 + s1[13]*cp24))
	s2[11], s2[12] = s1[11], s1[12]

	// stage 5
	s1[0] = wl(s2[0] + s2[3])
	s1[1] = wl(s2[1] + s2[2])
	s1[2] = wl(s2[1] - s2[2])
	s1[3] = wl(s2[0] - s2[3])
	s1[4] = s2[4]
	s1[5] = int32(drs((s2[6] - s2[5]) * cp16))
	s1[6] = int32(drs((s2[5] + s2[6]) * cp16))
	s1[7] = s2[7]
	s1[8] = wl(s2[8] + s2[11])
	s1[9] = wl(s2[9] + s2[10])
	s1[10] = wl(s2[9] - s2[10])
	s1[11] = wl(s2[8] - s2[11])
	s1[12] = wl(-s2[12] + s2[15])
	s1[13] = wl(-s2[13] + s2[14])
	s1[14] = wl(s2[13] + s2[14])
	s1[15] = wl(s2[12] + s2[15])

	// stage 6
	s2[0] = wl(s1[0] + s1[7])
	s2[1] = wl(s1[1] + s1[6])
	s2[2] = wl(s1[2] + s1[5])
	s2[3] = wl(s1[3] + s1[4])
	s2[4] = wl(s1[3] - s1[4])
	s2[5] = wl(s1[2] - s1[5])
	s2[6] = wl(s1[1] - s1[6])
	s2[7] = wl(s1[0] - s1[7])
	s2[8], s2[9] = s1[8], s1[9]
	s2[10] = int32(drs((-s1[10] + s1[13]) * cp16))
	s2[13] = int32(drs((s1[10] + s1[13]) * cp16))
	s2[11] = int32(drs((-s1[11] + s1[12]) * cp16))
	s2[12] = int32(drs((s1[11] + s1[12]) * cp16))
	s2[14], s2[15] = s1[14], s1[15]

	// stage 7
	io[0] = int16(wl(s2[0] + s2[15]))
	io[1] = int16(wl(s2[1] + s2[14]))
	io[2] = int16(wl(s2[2] + s2[13]))
	io[3] = int16(wl(s2[3] + s2[12]))
	io[4] = int16(wl(s2[4] + s2[11]))
	io[5] = int16(wl(s2[5] + s2[10]))
	io[6] = int16(wl(s2[6] + s2[9]))
	io[7] = int16(wl(s2[7] + s2[8]))
	io[8] = int16(wl(s2[7] - s2[8]))
	io[9] = int16(wl(s2[6] - s2[9]))
	io[10] = int16(wl(s2[5] - s2[10]))
	io[11] = int16(wl(s2[4] - s2[11]))
	io[12] = int16(wl(s2[3] - s2[12]))
	io[13] = int16(wl(s2[2] - s2[13]))
	io[14] = int16(wl(s2[1] - s2[14]))
	io[15] = int16(wl(s2[0] - s2[15]))
}

// ---- 16-point inverse ADST (spec §8.7.1.6 / libvpx vp9_iadst16_c) ----

// iadst16 is a line-for-line port of libvpx iadst16_c (vpx_dsp/inv_txfm.c):
// 4 butterfly stages with WRAPLOW (wl) / dct_const_round_shift (drs), then the
// signed output permutation. Inputs are int16-range so int32 intermediates do
// not overflow.
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

	if x0|x1|x2|x3|x4|x5|x6|x7|x8|x9|x10|x11|x12|x13|x14|x15 == 0 {
		for i := 0; i < 16; i++ {
			io[i] = 0
		}
		return
	}

	// stage 1
	s0 := x0*cp1 + x1*cp31
	s1 := x0*cp31 - x1*cp1
	s2 := x2*cp5 + x3*cp27
	s3 := x2*cp27 - x3*cp5
	s4 := x4*cp9 + x5*cp23
	s5 := x4*cp23 - x5*cp9
	s6 := x6*cp13 + x7*cp19
	s7 := x6*cp19 - x7*cp13
	s8 := x8*cp17 + x9*cp15
	s9 := x8*cp15 - x9*cp17
	s10 := x10*cp21 + x11*cp11
	s11 := x10*cp11 - x11*cp21
	s12 := x12*cp25 + x13*cp7
	s13 := x12*cp7 - x13*cp25
	s14 := x14*cp29 + x15*cp3
	s15 := x14*cp3 - x15*cp29

	x0 = int32(drs(s0 + s8))
	x1 = int32(drs(s1 + s9))
	x2 = int32(drs(s2 + s10))
	x3 = int32(drs(s3 + s11))
	x4 = int32(drs(s4 + s12))
	x5 = int32(drs(s5 + s13))
	x6 = int32(drs(s6 + s14))
	x7 = int32(drs(s7 + s15))
	x8 = int32(drs(s0 - s8))
	x9 = int32(drs(s1 - s9))
	x10 = int32(drs(s2 - s10))
	x11 = int32(drs(s3 - s11))
	x12 = int32(drs(s4 - s12))
	x13 = int32(drs(s5 - s13))
	x14 = int32(drs(s6 - s14))
	x15 = int32(drs(s7 - s15))

	// stage 2
	s0 = x0
	s1 = x1
	s2 = x2
	s3 = x3
	s4 = x4
	s5 = x5
	s6 = x6
	s7 = x7
	s8 = x8*cp4 + x9*cp28
	s9 = x8*cp28 - x9*cp4
	s10 = x10*cp20 + x11*cp12
	s11 = x10*cp12 - x11*cp20
	s12 = -x12*cp28 + x13*cp4
	s13 = x12*cp4 + x13*cp28
	s14 = -x14*cp12 + x15*cp20
	s15 = x14*cp20 + x15*cp12

	x0 = wl(s0 + s4)
	x1 = wl(s1 + s5)
	x2 = wl(s2 + s6)
	x3 = wl(s3 + s7)
	x4 = wl(s0 - s4)
	x5 = wl(s1 - s5)
	x6 = wl(s2 - s6)
	x7 = wl(s3 - s7)
	x8 = int32(drs(s8 + s12))
	x9 = int32(drs(s9 + s13))
	x10 = int32(drs(s10 + s14))
	x11 = int32(drs(s11 + s15))
	x12 = int32(drs(s8 - s12))
	x13 = int32(drs(s9 - s13))
	x14 = int32(drs(s10 - s14))
	x15 = int32(drs(s11 - s15))

	// stage 3
	s0 = x0
	s1 = x1
	s2 = x2
	s3 = x3
	s4 = x4*cp8 + x5*cp24
	s5 = x4*cp24 - x5*cp8
	s6 = -x6*cp24 + x7*cp8
	s7 = x6*cp8 + x7*cp24
	s8 = x8
	s9 = x9
	s10 = x10
	s11 = x11
	s12 = x12*cp8 + x13*cp24
	s13 = x12*cp24 - x13*cp8
	s14 = -x14*cp24 + x15*cp8
	s15 = x14*cp8 + x15*cp24

	x0 = wl(s0 + s2)
	x1 = wl(s1 + s3)
	x2 = wl(s0 - s2)
	x3 = wl(s1 - s3)
	x4 = int32(drs(s4 + s6))
	x5 = int32(drs(s5 + s7))
	x6 = int32(drs(s4 - s6))
	x7 = int32(drs(s5 - s7))
	x8 = wl(s8 + s10)
	x9 = wl(s9 + s11)
	x10 = wl(s8 - s10)
	x11 = wl(s9 - s11)
	x12 = int32(drs(s12 + s14))
	x13 = int32(drs(s13 + s15))
	x14 = int32(drs(s12 - s14))
	x15 = int32(drs(s13 - s15))

	// stage 4
	s2 = -cp16 * (x2 + x3)
	s3 = cp16 * (x2 - x3)
	s6 = cp16 * (x6 + x7)
	s7 = cp16 * (-x6 + x7)
	s10 = cp16 * (x10 + x11)
	s11 = cp16 * (-x10 + x11)
	s14 = -cp16 * (x14 + x15)
	s15 = cp16 * (x14 - x15)

	x2 = int32(drs(s2))
	x3 = int32(drs(s3))
	x6 = int32(drs(s6))
	x7 = int32(drs(s7))
	x10 = int32(drs(s10))
	x11 = int32(drs(s11))
	x14 = int32(drs(s14))
	x15 = int32(drs(s15))

	io[0] = int16(x0)
	io[1] = int16(-x8)
	io[2] = int16(x12)
	io[3] = int16(-x4)
	io[4] = int16(x6)
	io[5] = int16(x14)
	io[6] = int16(x10)
	io[7] = int16(x2)
	io[8] = int16(x3)
	io[9] = int16(x11)
	io[10] = int16(x15)
	io[11] = int16(x7)
	io[12] = int16(x5)
	io[13] = int16(-x13)
	io[14] = int16(x9)
	io[15] = int16(-x1)
}

// ---- 32-point inverse DCT (spec §8.7.1.7 / libvpx vp9_idct32_c) ----

// idct32 is a line-for-line port of libvpx idct32_c (vpx_dsp/inv_txfm.c):
// 8 stages, every step kept in int16 range via wl (WRAPLOW); drs is
// dct_const_round_shift. s1/s2 mirror libvpx step1/step2.
func idct32(io []int16) {
	var s1, s2 [32]int32

	// stage 1
	s1[0] = int32(io[0])
	s1[1] = int32(io[16])
	s1[2] = int32(io[8])
	s1[3] = int32(io[24])
	s1[4] = int32(io[4])
	s1[5] = int32(io[20])
	s1[6] = int32(io[12])
	s1[7] = int32(io[28])
	s1[8] = int32(io[2])
	s1[9] = int32(io[18])
	s1[10] = int32(io[10])
	s1[11] = int32(io[26])
	s1[12] = int32(io[6])
	s1[13] = int32(io[22])
	s1[14] = int32(io[14])
	s1[15] = int32(io[30])
	s1[16] = int32(drs(int32(io[1])*cp31 - int32(io[31])*cp1))
	s1[31] = int32(drs(int32(io[1])*cp1 + int32(io[31])*cp31))
	s1[17] = int32(drs(int32(io[17])*cp15 - int32(io[15])*cp17))
	s1[30] = int32(drs(int32(io[17])*cp17 + int32(io[15])*cp15))
	s1[18] = int32(drs(int32(io[9])*cp23 - int32(io[23])*cp9))
	s1[29] = int32(drs(int32(io[9])*cp9 + int32(io[23])*cp23))
	s1[19] = int32(drs(int32(io[25])*cp7 - int32(io[7])*cp25))
	s1[28] = int32(drs(int32(io[25])*cp25 + int32(io[7])*cp7))
	s1[20] = int32(drs(int32(io[5])*cp27 - int32(io[27])*cp5))
	s1[27] = int32(drs(int32(io[5])*cp5 + int32(io[27])*cp27))
	s1[21] = int32(drs(int32(io[21])*cp11 - int32(io[11])*cp21))
	s1[26] = int32(drs(int32(io[21])*cp21 + int32(io[11])*cp11))
	s1[22] = int32(drs(int32(io[13])*cp19 - int32(io[19])*cp13))
	s1[25] = int32(drs(int32(io[13])*cp13 + int32(io[19])*cp19))
	s1[23] = int32(drs(int32(io[29])*cp3 - int32(io[3])*cp29))
	s1[24] = int32(drs(int32(io[29])*cp29 + int32(io[3])*cp3))

	// stage 2
	s2[0], s2[1], s2[2], s2[3] = s1[0], s1[1], s1[2], s1[3]
	s2[4], s2[5], s2[6], s2[7] = s1[4], s1[5], s1[6], s1[7]
	s2[8] = int32(drs(s1[8]*cp30 - s1[15]*cp2))
	s2[15] = int32(drs(s1[8]*cp2 + s1[15]*cp30))
	s2[9] = int32(drs(s1[9]*cp14 - s1[14]*cp18))
	s2[14] = int32(drs(s1[9]*cp18 + s1[14]*cp14))
	s2[10] = int32(drs(s1[10]*cp22 - s1[13]*cp10))
	s2[13] = int32(drs(s1[10]*cp10 + s1[13]*cp22))
	s2[11] = int32(drs(s1[11]*cp6 - s1[12]*cp26))
	s2[12] = int32(drs(s1[11]*cp26 + s1[12]*cp6))
	s2[16] = wl(s1[16] + s1[17])
	s2[17] = wl(s1[16] - s1[17])
	s2[18] = wl(-s1[18] + s1[19])
	s2[19] = wl(s1[18] + s1[19])
	s2[20] = wl(s1[20] + s1[21])
	s2[21] = wl(s1[20] - s1[21])
	s2[22] = wl(-s1[22] + s1[23])
	s2[23] = wl(s1[22] + s1[23])
	s2[24] = wl(s1[24] + s1[25])
	s2[25] = wl(s1[24] - s1[25])
	s2[26] = wl(-s1[26] + s1[27])
	s2[27] = wl(s1[26] + s1[27])
	s2[28] = wl(s1[28] + s1[29])
	s2[29] = wl(s1[28] - s1[29])
	s2[30] = wl(-s1[30] + s1[31])
	s2[31] = wl(s1[30] + s1[31])

	// stage 3
	s1[0], s1[1], s1[2], s1[3] = s2[0], s2[1], s2[2], s2[3]
	s1[4] = int32(drs(s2[4]*cp28 - s2[7]*cp4))
	s1[7] = int32(drs(s2[4]*cp4 + s2[7]*cp28))
	s1[5] = int32(drs(s2[5]*cp12 - s2[6]*cp20))
	s1[6] = int32(drs(s2[5]*cp20 + s2[6]*cp12))
	s1[8] = wl(s2[8] + s2[9])
	s1[9] = wl(s2[8] - s2[9])
	s1[10] = wl(-s2[10] + s2[11])
	s1[11] = wl(s2[10] + s2[11])
	s1[12] = wl(s2[12] + s2[13])
	s1[13] = wl(s2[12] - s2[13])
	s1[14] = wl(-s2[14] + s2[15])
	s1[15] = wl(s2[14] + s2[15])
	s1[16] = s2[16]
	s1[31] = s2[31]
	s1[17] = int32(drs(-s2[17]*cp4 + s2[30]*cp28))
	s1[30] = int32(drs(s2[17]*cp28 + s2[30]*cp4))
	s1[18] = int32(drs(-s2[18]*cp28 - s2[29]*cp4))
	s1[29] = int32(drs(-s2[18]*cp4 + s2[29]*cp28))
	s1[19] = s2[19]
	s1[20] = s2[20]
	s1[21] = int32(drs(-s2[21]*cp20 + s2[26]*cp12))
	s1[26] = int32(drs(s2[21]*cp12 + s2[26]*cp20))
	s1[22] = int32(drs(-s2[22]*cp12 - s2[25]*cp20))
	s1[25] = int32(drs(-s2[22]*cp20 + s2[25]*cp12))
	s1[23] = s2[23]
	s1[24] = s2[24]
	s1[27] = s2[27]
	s1[28] = s2[28]

	// stage 4
	s2[0] = int32(drs((s1[0] + s1[1]) * cp16))
	s2[1] = int32(drs((s1[0] - s1[1]) * cp16))
	s2[2] = int32(drs(s1[2]*cp24 - s1[3]*cp8))
	s2[3] = int32(drs(s1[2]*cp8 + s1[3]*cp24))
	s2[4] = wl(s1[4] + s1[5])
	s2[5] = wl(s1[4] - s1[5])
	s2[6] = wl(-s1[6] + s1[7])
	s2[7] = wl(s1[6] + s1[7])
	s2[8] = s1[8]
	s2[15] = s1[15]
	s2[9] = int32(drs(-s1[9]*cp8 + s1[14]*cp24))
	s2[14] = int32(drs(s1[9]*cp24 + s1[14]*cp8))
	s2[10] = int32(drs(-s1[10]*cp24 - s1[13]*cp8))
	s2[13] = int32(drs(-s1[10]*cp8 + s1[13]*cp24))
	s2[11] = s1[11]
	s2[12] = s1[12]
	s2[16] = wl(s1[16] + s1[19])
	s2[17] = wl(s1[17] + s1[18])
	s2[18] = wl(s1[17] - s1[18])
	s2[19] = wl(s1[16] - s1[19])
	s2[20] = wl(-s1[20] + s1[23])
	s2[21] = wl(-s1[21] + s1[22])
	s2[22] = wl(s1[21] + s1[22])
	s2[23] = wl(s1[20] + s1[23])
	s2[24] = wl(s1[24] + s1[27])
	s2[25] = wl(s1[25] + s1[26])
	s2[26] = wl(s1[25] - s1[26])
	s2[27] = wl(s1[24] - s1[27])
	s2[28] = wl(-s1[28] + s1[31])
	s2[29] = wl(-s1[29] + s1[30])
	s2[30] = wl(s1[29] + s1[30])
	s2[31] = wl(s1[28] + s1[31])

	// stage 5
	s1[0] = wl(s2[0] + s2[3])
	s1[1] = wl(s2[1] + s2[2])
	s1[2] = wl(s2[1] - s2[2])
	s1[3] = wl(s2[0] - s2[3])
	s1[4] = s2[4]
	s1[5] = int32(drs((s2[6] - s2[5]) * cp16))
	s1[6] = int32(drs((s2[5] + s2[6]) * cp16))
	s1[7] = s2[7]
	s1[8] = wl(s2[8] + s2[11])
	s1[9] = wl(s2[9] + s2[10])
	s1[10] = wl(s2[9] - s2[10])
	s1[11] = wl(s2[8] - s2[11])
	s1[12] = wl(-s2[12] + s2[15])
	s1[13] = wl(-s2[13] + s2[14])
	s1[14] = wl(s2[13] + s2[14])
	s1[15] = wl(s2[12] + s2[15])
	s1[16] = s2[16]
	s1[17] = s2[17]
	s1[18] = int32(drs(-s2[18]*cp8 + s2[29]*cp24))
	s1[29] = int32(drs(s2[18]*cp24 + s2[29]*cp8))
	s1[19] = int32(drs(-s2[19]*cp8 + s2[28]*cp24))
	s1[28] = int32(drs(s2[19]*cp24 + s2[28]*cp8))
	s1[20] = int32(drs(-s2[20]*cp24 - s2[27]*cp8))
	s1[27] = int32(drs(-s2[20]*cp8 + s2[27]*cp24))
	s1[21] = int32(drs(-s2[21]*cp24 - s2[26]*cp8))
	s1[26] = int32(drs(-s2[21]*cp8 + s2[26]*cp24))
	s1[22] = s2[22]
	s1[23] = s2[23]
	s1[24] = s2[24]
	s1[25] = s2[25]
	s1[30] = s2[30]
	s1[31] = s2[31]

	// stage 6
	s2[0] = wl(s1[0] + s1[7])
	s2[1] = wl(s1[1] + s1[6])
	s2[2] = wl(s1[2] + s1[5])
	s2[3] = wl(s1[3] + s1[4])
	s2[4] = wl(s1[3] - s1[4])
	s2[5] = wl(s1[2] - s1[5])
	s2[6] = wl(s1[1] - s1[6])
	s2[7] = wl(s1[0] - s1[7])
	s2[8] = s1[8]
	s2[9] = s1[9]
	s2[10] = int32(drs((-s1[10] + s1[13]) * cp16))
	s2[13] = int32(drs((s1[10] + s1[13]) * cp16))
	s2[11] = int32(drs((-s1[11] + s1[12]) * cp16))
	s2[12] = int32(drs((s1[11] + s1[12]) * cp16))
	s2[14] = s1[14]
	s2[15] = s1[15]
	s2[16] = wl(s1[16] + s1[23])
	s2[17] = wl(s1[17] + s1[22])
	s2[18] = wl(s1[18] + s1[21])
	s2[19] = wl(s1[19] + s1[20])
	s2[20] = wl(s1[19] - s1[20])
	s2[21] = wl(s1[18] - s1[21])
	s2[22] = wl(s1[17] - s1[22])
	s2[23] = wl(s1[16] - s1[23])
	s2[24] = wl(-s1[24] + s1[31])
	s2[25] = wl(-s1[25] + s1[30])
	s2[26] = wl(-s1[26] + s1[29])
	s2[27] = wl(-s1[27] + s1[28])
	s2[28] = wl(s1[27] + s1[28])
	s2[29] = wl(s1[26] + s1[29])
	s2[30] = wl(s1[25] + s1[30])
	s2[31] = wl(s1[24] + s1[31])

	// stage 7
	s1[0] = wl(s2[0] + s2[15])
	s1[1] = wl(s2[1] + s2[14])
	s1[2] = wl(s2[2] + s2[13])
	s1[3] = wl(s2[3] + s2[12])
	s1[4] = wl(s2[4] + s2[11])
	s1[5] = wl(s2[5] + s2[10])
	s1[6] = wl(s2[6] + s2[9])
	s1[7] = wl(s2[7] + s2[8])
	s1[8] = wl(s2[7] - s2[8])
	s1[9] = wl(s2[6] - s2[9])
	s1[10] = wl(s2[5] - s2[10])
	s1[11] = wl(s2[4] - s2[11])
	s1[12] = wl(s2[3] - s2[12])
	s1[13] = wl(s2[2] - s2[13])
	s1[14] = wl(s2[1] - s2[14])
	s1[15] = wl(s2[0] - s2[15])
	s1[16] = s2[16]
	s1[17] = s2[17]
	s1[18] = s2[18]
	s1[19] = s2[19]
	s1[20] = int32(drs((-s2[20] + s2[27]) * cp16))
	s1[27] = int32(drs((s2[20] + s2[27]) * cp16))
	s1[21] = int32(drs((-s2[21] + s2[26]) * cp16))
	s1[26] = int32(drs((s2[21] + s2[26]) * cp16))
	s1[22] = int32(drs((-s2[22] + s2[25]) * cp16))
	s1[25] = int32(drs((s2[22] + s2[25]) * cp16))
	s1[23] = int32(drs((-s2[23] + s2[24]) * cp16))
	s1[24] = int32(drs((s2[23] + s2[24]) * cp16))
	s1[28] = s2[28]
	s1[29] = s2[29]
	s1[30] = s2[30]
	s1[31] = s2[31]

	// final stage
	io[0] = int16(wl(s1[0] + s1[31]))
	io[1] = int16(wl(s1[1] + s1[30]))
	io[2] = int16(wl(s1[2] + s1[29]))
	io[3] = int16(wl(s1[3] + s1[28]))
	io[4] = int16(wl(s1[4] + s1[27]))
	io[5] = int16(wl(s1[5] + s1[26]))
	io[6] = int16(wl(s1[6] + s1[25]))
	io[7] = int16(wl(s1[7] + s1[24]))
	io[8] = int16(wl(s1[8] + s1[23]))
	io[9] = int16(wl(s1[9] + s1[22]))
	io[10] = int16(wl(s1[10] + s1[21]))
	io[11] = int16(wl(s1[11] + s1[20]))
	io[12] = int16(wl(s1[12] + s1[19]))
	io[13] = int16(wl(s1[13] + s1[18]))
	io[14] = int16(wl(s1[14] + s1[17]))
	io[15] = int16(wl(s1[15] + s1[16]))
	io[16] = int16(wl(s1[15] - s1[16]))
	io[17] = int16(wl(s1[14] - s1[17]))
	io[18] = int16(wl(s1[13] - s1[18]))
	io[19] = int16(wl(s1[12] - s1[19]))
	io[20] = int16(wl(s1[11] - s1[20]))
	io[21] = int16(wl(s1[10] - s1[21]))
	io[22] = int16(wl(s1[9] - s1[22]))
	io[23] = int16(wl(s1[8] - s1[23]))
	io[24] = int16(wl(s1[7] - s1[24]))
	io[25] = int16(wl(s1[6] - s1[25]))
	io[26] = int16(wl(s1[5] - s1[26]))
	io[27] = int16(wl(s1[4] - s1[27]))
	io[28] = int16(wl(s1[3] - s1[28]))
	io[29] = int16(wl(s1[2] - s1[29]))
	io[30] = int16(wl(s1[1] - s1[30]))
	io[31] = int16(wl(s1[0] - s1[31]))
}
