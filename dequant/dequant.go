// Package dequant implements VP9 dequantization (spec §8.6.2).
// Quantizer step sizes are looked up from dc_qlookup / ac_qlookup tables
// (spec Appendix A) using a per-frame base Q index plus optional per-plane
// and per-segment deltas. Profile 0/1 (8-bit) only.
package dequant

// DCQuant returns the DC quantizer step size for a given Q index (0–255).
func DCQuant(qIndex int) int16 {
	return dcQLookup[clampQIndex(qIndex)]
}

// ACQuant returns the AC quantizer step size for a given Q index (0–255).
func ACQuant(qIndex int) int16 {
	return acQLookup[clampQIndex(qIndex)]
}

// DCQuantBD / ACQuantBD return the quantizer step for a given bit depth (8/10/12)
// — libvpx vp9_dc_quant / vp9_ac_quant switch on bit_depth.
func DCQuantBD(qIndex, bitDepth int) int16 {
	q := clampQIndex(qIndex)
	switch bitDepth {
	case 10:
		return dcQLookup10[q]
	case 12:
		return dcQLookup12[q]
	default:
		return dcQLookup[q]
	}
}

func ACQuantBD(qIndex, bitDepth int) int16 {
	q := clampQIndex(qIndex)
	switch bitDepth {
	case 10:
		return acQLookup10[q]
	case 12:
		return acQLookup12[q]
	default:
		return acQLookup[q]
	}
}

// DequantBlock applies dequantization to a bs×bs int16 coefficient block
// (row-major, zigzag un-done by caller). The first coefficient (DC) is
// multiplied by dc; all others by ac. dqShift is libvpx's per-TX dq_shift —
// 1 for TX_32X32, else 0 — applied to the magnitude (|coeff|*q) >> dqShift
// before re-applying the sign (libvpx: v = (val * dqv) >> dq_shift, val ≥ 0).
func DequantBlock(coeffs []int16, dc, ac int16, dqShift int) {
	if len(coeffs) == 0 {
		return
	}
	coeffs[0] = dequant1(coeffs[0], dc, dqShift)
	for i := 1; i < len(coeffs); i++ {
		coeffs[i] = dequant1(coeffs[i], ac, dqShift)
	}
}

func dequant1(v, q int16, dqShift int) int16 {
	p := int32(v) * int32(q)
	if dqShift > 0 {
		if p < 0 {
			p = -((-p) >> uint(dqShift))
		} else {
			p >>= uint(dqShift)
		}
	}
	return clamp16(p)
}

// DequantBlock16 is DequantBlock for the high-bit-depth (10/12-bit) int32 path.
// dc/ac come from DCQuantBD/ACQuantBD; the result is NOT clamped to int16 (10/12-
// bit dequantized coefficients exceed int16 — libvpx highbd_check_range is a
// no-op; the bit-depth clamp happens in the highbd inverse transform).
func DequantBlock16(coeffs []int64, dc, ac int16, dqShift int) {
	if len(coeffs) == 0 {
		return
	}
	coeffs[0] = dequant1_16(coeffs[0], dc, dqShift)
	for i := 1; i < len(coeffs); i++ {
		coeffs[i] = dequant1_16(coeffs[i], ac, dqShift)
	}
}

func dequant1_16(v int64, q int16, dqShift int) int64 {
	p := v * int64(q)
	if dqShift > 0 {
		if p < 0 {
			p = -((-p) >> uint(dqShift))
		} else {
			p >>= uint(dqShift)
		}
	}
	return p
}

func clampQIndex(q int) int {
	if q < 0 {
		return 0
	}
	if q > 255 {
		return 255
	}
	return q
}

func clamp16(v int32) int16 {
	if v < -32768 {
		return -32768
	}
	if v > 32767 {
		return 32767
	}
	return int16(v)
}

// dc_qlookup[qIndex] — spec Appendix A, Table A-2 (8-bit profile 0/1).
var dcQLookup = [256]int16{
	4, 8, 8, 9, 10, 11, 12, 12,
	13, 14, 15, 16, 17, 18, 19, 19,
	20, 21, 22, 23, 24, 25, 26, 26,
	27, 28, 29, 30, 31, 32, 32, 33,
	34, 35, 36, 37, 38, 38, 39, 40,
	41, 42, 43, 43, 44, 45, 46, 47,
	48, 48, 49, 50, 51, 52, 53, 53,
	54, 55, 56, 57, 57, 58, 59, 60,
	61, 62, 62, 63, 64, 65, 66, 66,
	67, 68, 69, 70, 70, 71, 72, 73,
	74, 74, 75, 76, 77, 78, 78, 79,
	80, 81, 81, 82, 83, 84, 85, 85,
	87, 88, 90, 92, 93, 95, 96, 98,
	99, 101, 102, 104, 105, 107, 108, 110,
	111, 113, 114, 116, 117, 118, 120, 121,
	123, 125, 127, 129, 131, 134, 136, 138,
	140, 142, 144, 146, 148, 150, 152, 154,
	156, 158, 161, 164, 166, 169, 172, 174,
	177, 180, 182, 185, 187, 190, 192, 195,
	199, 202, 205, 208, 211, 214, 217, 220,
	223, 226, 230, 233, 237, 240, 243, 247,
	250, 253, 257, 261, 265, 269, 272, 276,
	280, 284, 288, 292, 296, 300, 304, 309,
	313, 317, 322, 326, 330, 335, 340, 344,
	349, 354, 359, 364, 369, 374, 379, 384,
	389, 395, 400, 406, 411, 417, 423, 429,
	435, 441, 447, 454, 461, 467, 475, 482,
	489, 497, 505, 513, 522, 530, 539, 549,
	559, 569, 579, 590, 602, 614, 626, 640,
	654, 668, 684, 700, 717, 736, 755, 775,
	796, 819, 843, 869, 896, 925, 955, 988,
	1022, 1058, 1098, 1139, 1184, 1232, 1282, 1336,
}

// ac_qlookup[qIndex] — spec Appendix A, Table A-2 (8-bit profile 0/1).
var acQLookup = [256]int16{
	4, 8, 9, 10, 11, 12, 13, 14,
	15, 16, 17, 18, 19, 20, 21, 22,
	23, 24, 25, 26, 27, 28, 29, 30,
	31, 32, 33, 34, 35, 36, 37, 38,
	39, 40, 41, 42, 43, 44, 45, 46,
	47, 48, 49, 50, 51, 52, 53, 54,
	55, 56, 57, 58, 59, 60, 61, 62,
	63, 64, 65, 66, 67, 68, 69, 70,
	71, 72, 73, 74, 75, 76, 77, 78,
	79, 80, 81, 82, 83, 84, 85, 86,
	87, 88, 89, 90, 91, 92, 93, 94,
	95, 96, 97, 98, 99, 100, 101, 102,
	104, 106, 108, 110, 112, 114, 116, 118,
	120, 122, 124, 126, 128, 130, 132, 134,
	136, 138, 140, 142, 144, 146, 148, 150,
	152, 155, 158, 161, 164, 167, 170, 173,
	176, 179, 182, 185, 188, 191, 194, 197,
	200, 203, 207, 211, 215, 219, 223, 227,
	231, 235, 239, 243, 247, 251, 255, 260,
	265, 270, 275, 280, 285, 290, 295, 300,
	305, 311, 317, 323, 329, 335, 341, 347,
	353, 359, 366, 373, 380, 387, 394, 401,
	408, 416, 424, 432, 440, 448, 456, 465,
	474, 483, 492, 501, 510, 520, 530, 540,
	550, 560, 571, 582, 593, 604, 615, 627,
	639, 651, 663, 676, 689, 702, 715, 729,
	743, 757, 771, 786, 801, 816, 832, 848,
	864, 881, 898, 915, 933, 951, 969, 988,
	1007, 1026, 1046, 1066, 1087, 1108, 1129, 1151,
	1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343,
	1369, 1396, 1423, 1451, 1479, 1508, 1537, 1567,
	1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
}
