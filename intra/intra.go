// Package intra implements VP9 intra prediction for all 10 modes (spec §8.5.1).
// All prediction functions operate on square blocks of size bs × bs pixels
// (bs = 4, 8, 16, or 32 — the four VP9 transform sizes).
// Caller provides above/left reference samples extended as required per mode.
package intra

import (
	"math/bits"

	"github.com/mgvs/go-vp9/modeinfo"
)

// Predict fills dst (row-major, stride stride) with the intra prediction for
// the given mode and block size bs.
//
// Reference sample requirements:
//
//	above[0..bs-1]   – the row of pixels immediately above the block.
//	left[0..bs-1]    – the column of pixels to the left of the block.
//	topLeft          – the single pixel at position (−1, −1).
//
// Extended requirements for specific modes (caller must provide):
//
//	D45:  above[0..2*bs-1]
//	D63:  above[0..bs+bs/2+1]
//	D117: above[0..bs-1], left[0..bs-1]   (standard sizes suffice)
//	D207: left[0..bs-1]                   (standard size suffices)
func Predict(mode modeinfo.IntraMode, dst []uint8, stride, bs int, above, left []uint8, topLeft uint8) {
	switch mode {
	case modeinfo.DC:
		dcPredict(dst, stride, bs, above, left)
	case modeinfo.V:
		vPredict(dst, stride, bs, above)
	case modeinfo.H:
		hPredict(dst, stride, bs, left)
	case modeinfo.TM:
		tmPredict(dst, stride, bs, above, left, topLeft)
	case modeinfo.D45:
		d45Predict(dst, stride, bs, above)
	case modeinfo.D63:
		d63Predict(dst, stride, bs, above)
	case modeinfo.D117:
		d117Predict(dst, stride, bs, above, left, topLeft)
	case modeinfo.D135:
		d135Predict(dst, stride, bs, above, left, topLeft)
	case modeinfo.D153:
		d153Predict(dst, stride, bs, above, left, topLeft)
	case modeinfo.D207:
		d207Predict(dst, stride, bs, left)
	}
}

// ---- helpers ----------------------------------------------------------------

func clamp8(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// avg2 = (a + b + 1) >> 1
func avg2(a, b uint8) uint8 { return uint8((int(a) + int(b) + 1) >> 1) }

// avg3 = (a + 2*b + c + 2) >> 2
func avg3(a, b, c uint8) uint8 { return uint8((int(a) + 2*int(b) + int(c) + 2) >> 2) }

// log2b returns ⌊log₂(n)⌋ for n > 0.
func log2b(n int) int { return bits.Len(uint(n)) - 1 }

// ---- DC prediction (§8.5.1.2) -----------------------------------------------

func dcPredict(dst []uint8, stride, bs int, above, left []uint8) {
	// above has stride (=bsW) samples; left has bs (=bsH) samples.
	nAbove := len(above)
	nLeft := len(left)
	total := nAbove + nLeft
	sum := 0
	for i := 0; i < nAbove; i++ {
		sum += int(above[i])
	}
	for i := 0; i < nLeft; i++ {
		sum += int(left[i])
	}
	avg := uint8((sum + total/2) / total)
	// Fill first row scalar, then copy to remaining rows (copy uses SIMD).
	first := dst[:stride]
	for i := range first {
		first[i] = avg
	}
	for r := 1; r < bs; r++ {
		copy(dst[r*stride:r*stride+stride], first)
	}
}

// ---- V prediction (§8.5.1.3) ------------------------------------------------

func vPredict(dst []uint8, stride, bs int, above []uint8) {
	for r := 0; r < bs; r++ {
		copy(dst[r*stride:r*stride+stride], above[:stride])
	}
}

// ---- H prediction (§8.5.1.4) ------------------------------------------------

func hPredict(dst []uint8, stride, bs int, left []uint8) {
	for r := 0; r < bs; r++ {
		v := left[r]
		for c := 0; c < stride; c++ {
			dst[r*stride+c] = v
		}
	}
}

// ---- TM prediction (§8.5.1.11) ---------------------------------------------

func tmPredict(dst []uint8, stride, bs int, above, left []uint8, topLeft uint8) {
	tl := int(topLeft)
	for r := 0; r < bs; r++ {
		lr := int(left[r])
		for c := 0; c < stride; c++ {
			dst[r*stride+c] = clamp8(int(above[c]) + lr - tl)
		}
	}
}

// ---- D45 prediction (§8.5.1.5) ---------------------------------------------
// Direction: 45° from vertical toward upper-right.
// Requires above[0..2*bs-1].

func d45Predict(dst []uint8, stride, bs int, above []uint8) {
	// above has 2*stride samples (caller provides extended above row).
	lim := 2 * stride
	for r := 0; r < bs; r++ {
		for c := 0; c < stride; c++ {
			if r+c+2 < lim {
				dst[r*stride+c] = avg3(above[r+c], above[r+c+1], above[r+c+2])
			} else {
				dst[r*stride+c] = above[lim-1]
			}
		}
	}
}

// ---- D63 prediction (§8.5.1.6) ---------------------------------------------
// Direction: 63° from vertical toward upper-right (steeper than D45).
// Requires above[0..bs+bs/2+1].

func d63Predict(dst []uint8, stride, bs int, above []uint8) {
	// above has 2*stride samples; max index accessed = (bs/2) + stride + 2.
	maxIdx := len(above) - 1
	for r := 0; r < bs; r++ {
		half := r >> 1
		for c := 0; c < stride; c++ {
			i := half + c
			i0 := i
			if i0 > maxIdx {
				i0 = maxIdx
			}
			i1 := i + 1
			if i1 > maxIdx {
				i1 = maxIdx
			}
			i2 := i + 2
			if i2 > maxIdx {
				i2 = maxIdx
			}
			if r&1 == 0 {
				dst[r*stride+c] = avg2(above[i0], above[i1])
			} else {
				dst[r*stride+c] = avg3(above[i0], above[i1], above[i2])
			}
		}
	}
}

// ---- D117 prediction (§8.5.1.7) --------------------------------------------
// Direction: 117° from vertical (63° from horizontal, upper-left).
// Requires above[0..bs-1], left[0..bs-1], topLeft.

func d117Predict(dst []uint8, stride, bs int, above, left []uint8, topLeft uint8) {
	// Row 0: avg2 of consecutive above pixels (above[-1] = topLeft).
	dst[0] = avg2(topLeft, above[0])
	for c := 1; c < stride; c++ {
		dst[c] = avg2(above[c-1], above[c])
	}
	if bs < 2 {
		return
	}
	// Row 1: left[0], topLeft/above samples.
	dst[stride] = avg3(left[0], topLeft, above[0])
	if stride > 1 {
		dst[stride+1] = avg3(topLeft, above[0], above[1])
	}
	for c := 2; c < stride; c++ {
		dst[stride+c] = avg3(above[c-2], above[c-1], above[c])
	}
	// Rows r ≥ 2.
	for r := 2; r < bs; r++ {
		row := dst[r*stride:]
		row[0] = avg3(left[r-2], left[r-1], left[r])
		if stride > 1 {
			row[1] = avg3(left[r-2], left[r-1], dst[(r-2)*stride])
		}
		for c := 2; c < stride; c++ {
			row[c] = dst[(r-2)*stride+c-2]
		}
	}
}

// ---- D135 prediction (§8.5.1.8) --------------------------------------------
// Direction: 135° from vertical (upper-left diagonal).
// Requires above[0..bs-1], left[0..bs-1], topLeft.

func d135Predict(dst []uint8, stride, bs int, above, left []uint8, topLeft uint8) {
	n := bs
	if stride < n {
		n = stride
	}
	buf := make([]uint8, 2*n+1)
	buf[n] = topLeft
	for i := 0; i < n; i++ {
		buf[n+1+i] = above[i]
		buf[n-1-i] = left[i]
	}
	for r := 0; r < bs; r++ {
		for c := 0; c < stride; c++ {
			k := n + c - r
			if k < 0 {
				k = 0
			} else if k > 2*n {
				k = 2 * n
			}
			kNext := k + 1
			if kNext > 2*n {
				kNext = 2 * n
			}
			kPrev := k - 1
			if kPrev < 0 {
				kPrev = 0
			}
			dst[r*stride+c] = avg3(buf[kNext], buf[k], buf[kPrev])
		}
	}
}

// ---- D153 prediction (§8.5.1.9) --------------------------------------------
// Direction: 153° from vertical (shallow upper-left, mostly from left column).
// Requires above[0..bs-1], left[0..bs-1], topLeft.

func d153Predict(dst []uint8, stride, bs int, above, left []uint8, topLeft uint8) {
	dst[0] = avg2(topLeft, left[0])
	for r := 1; r < bs; r++ {
		dst[r*stride] = avg2(left[r-1], left[r])
	}
	if stride < 2 {
		return
	}
	dst[1] = avg3(left[0], topLeft, above[0])
	if bs > 1 {
		dst[stride+1] = avg3(topLeft, left[0], left[1])
		for r := 2; r < bs; r++ {
			dst[r*stride+1] = avg3(left[r-2], left[r-1], left[r])
		}
	}
	for c := 2; c < stride; c++ {
		dst[c] = avg3(above[c-2], above[c-1], above[c])
	}
	for r := 1; r < bs; r++ {
		for c := 2; c < stride; c++ {
			dst[r*stride+c] = dst[(r-1)*stride+c-2]
		}
	}
}

// ---- D207 prediction (§8.5.1.10) -------------------------------------------
// Direction: 207° from vertical (lower-left, mostly from left column).
// Requires left[0..bs-1].

func d207Predict(dst []uint8, stride, bs int, left []uint8) {
	last := left[bs-1]
	for r := 0; r < bs-1; r++ {
		dst[r*stride] = avg2(left[r], left[r+1])
	}
	dst[(bs-1)*stride] = last
	if stride > 1 {
		for r := 0; r < bs-2; r++ {
			dst[r*stride+1] = avg3(left[r], left[r+1], left[r+2])
		}
		dst[(bs-2)*stride+1] = avg3(left[bs-2], last, last)
		dst[(bs-1)*stride+1] = last
	}
	for c := 2; c < stride; c++ {
		dst[c] = dst[c-2]
	}
	for r := 1; r < bs; r++ {
		for c := 2; c < stride; c++ {
			dst[r*stride+c] = dst[(r-1)*stride+c-2]
		}
	}
}
