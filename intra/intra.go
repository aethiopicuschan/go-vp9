// Package intra implements VP9 intra prediction for all 10 modes (spec §8.5.1).
// All prediction functions operate on square blocks of size bs × bs pixels
// (bs = 4, 8, 16, or 32 — the four VP9 transform sizes).
// Caller provides above/left reference samples extended as required per mode.
package intra

import (
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

// ---- DC prediction (§8.5.1.2) -----------------------------------------------

func dcPredict(dst []uint8, stride, bs int, above, left []uint8) {
	// DC averages exactly bs above + bs left samples (libvpx dc_predictor). The
	// caller's above slice is 2*bs long (extended for D45/D63), so cap at bs.
	total := 2 * bs
	sum := 0
	for i := 0; i < bs; i++ {
		sum += int(above[i])
	}
	for i := 0; i < bs; i++ {
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
