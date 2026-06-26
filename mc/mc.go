// Package mc implements VP9 motion compensation (inter prediction).
//
// Sub-pixel interpolation uses 8-tap (regular/smooth/sharp) or bilinear
// filters from VP9 spec Table B-2.  Filter indices run 0..15 in 1/16-pel
// units; callers convert 1/8-pel MV fractions by multiplying by 2.
//
// API convention:
//
//	src      – full reference plane buffer (row-major)
//	srcBase  – linear index of the integer-pixel origin of the block
//	           in src: pixel at (row r, col c) is src[srcBase + r*srcStride + c]
//	srcStride – row stride of src in bytes
//
// The caller must ensure that src[srcBase ± 3*srcStride ± 3] is valid (VP9
// reference frames carry an 8-pixel border on every side).
package mc

// Filter selects the VP9 sub-pixel interpolation filter type.
type Filter int

const (
	FilterRegular  Filter = 0 // EIGHTTAP
	FilterSmooth   Filter = 1 // EIGHTTAP_SMOOTH
	FilterSharp    Filter = 2 // EIGHTTAP_SHARP
	FilterBilinear Filter = 3
)

// filterKernels[filterType][subpelPos 0..15][tap 0..7]
// Each row sums to 128. subpelPos=0 → identity (copy).
// int16 because identity tap = 128, which overflows int8.
var filterKernels = [4][16][8]int16{
	// FilterRegular (EIGHTTAP) — VP9 spec Table B-2 / libvpx vp9_filter.h
	{
		{0, 0, 0, 128, 0, 0, 0, 0},
		{0, 1, -5, 126, 8, -3, 1, 0},
		{-1, 3, -10, 122, 18, -6, 2, 0},
		{-1, 4, -13, 118, 27, -9, 3, -1},
		{-1, 4, -16, 112, 37, -11, 4, -1},
		{-1, 5, -18, 105, 48, -14, 4, -1},
		{-1, 5, -19, 97, 58, -16, 5, -1},
		{-1, 6, -19, 88, 68, -18, 5, -1},
		{-1, 6, -19, 78, 78, -19, 6, -1},
		{-1, 5, -18, 68, 88, -19, 6, -1},
		{-1, 5, -16, 58, 97, -19, 5, -1},
		{-1, 4, -14, 48, 105, -18, 5, -1},
		{-1, 4, -11, 37, 112, -16, 4, -1},
		{-1, 3, -9, 27, 118, -13, 4, -1},
		{0, 2, -6, 18, 122, -10, 3, -1},
		{0, 1, -3, 8, 126, -5, 1, 0},
	},
	// FilterSmooth (EIGHTTAP_SMOOTH) — libvpx sub_pel_filters_8lp
	{
		{0, 0, 0, 128, 0, 0, 0, 0},
		{-3, -1, 32, 64, 38, 1, -3, 0},
		{-2, -2, 29, 63, 41, 2, -3, 0},
		{-2, -2, 26, 63, 43, 4, -4, 0},
		{-2, -3, 24, 62, 46, 5, -4, 0},
		{-2, -3, 21, 60, 49, 7, -4, 0},
		{-1, -4, 18, 59, 51, 9, -4, 0},
		{-1, -4, 16, 57, 53, 12, -4, -1},
		{-1, -4, 14, 55, 55, 14, -4, -1},
		{-1, -4, 12, 53, 57, 16, -4, -1},
		{0, -4, 9, 51, 59, 18, -4, -1},
		{0, -4, 7, 49, 60, 21, -3, -2},
		{0, -4, 5, 46, 62, 24, -3, -2},
		{0, -4, 4, 43, 63, 26, -2, -2},
		{0, -3, 2, 41, 63, 29, -2, -2},
		{0, -3, 1, 38, 64, 32, -1, -3},
	},
	// FilterSharp (EIGHTTAP_SHARP)
	{
		{0, 0, 0, 128, 0, 0, 0, 0},
		{-1, 3, -7, 127, 8, -3, 1, 0},
		{-2, 5, -13, 125, 17, -6, 3, -1},
		{-3, 7, -17, 121, 27, -10, 5, -2},
		{-4, 9, -20, 115, 37, -13, 6, -2},
		{-4, 10, -23, 108, 48, -16, 8, -3},
		{-4, 10, -24, 100, 59, -19, 9, -3},
		{-4, 11, -24, 90, 70, -21, 10, -4},
		{-4, 11, -23, 80, 80, -23, 11, -4},
		{-4, 10, -21, 70, 90, -24, 11, -4},
		{-3, 9, -19, 59, 100, -24, 10, -4},
		{-3, 8, -16, 48, 108, -23, 10, -4},
		{-2, 6, -13, 37, 115, -20, 9, -4},
		{-2, 5, -10, 27, 121, -17, 7, -3},
		{-1, 3, -6, 17, 125, -13, 5, -2},
		{0, 1, -3, 8, 127, -7, 3, -1},
	},
	// FilterBilinear — same centre addressing as 8-tap (taps at [3] and [4]).
	{
		{0, 0, 0, 128, 0, 0, 0, 0},
		{0, 0, 0, 120, 8, 0, 0, 0},
		{0, 0, 0, 112, 16, 0, 0, 0},
		{0, 0, 0, 104, 24, 0, 0, 0},
		{0, 0, 0, 96, 32, 0, 0, 0},
		{0, 0, 0, 88, 40, 0, 0, 0},
		{0, 0, 0, 80, 48, 0, 0, 0},
		{0, 0, 0, 72, 56, 0, 0, 0},
		{0, 0, 0, 64, 64, 0, 0, 0},
		{0, 0, 0, 56, 72, 0, 0, 0},
		{0, 0, 0, 48, 80, 0, 0, 0},
		{0, 0, 0, 40, 88, 0, 0, 0},
		{0, 0, 0, 32, 96, 0, 0, 0},
		{0, 0, 0, 24, 104, 0, 0, 0},
		{0, 0, 0, 16, 112, 0, 0, 0},
		{0, 0, 0, 8, 120, 0, 0, 0},
	},
}

func clampPixel(v int) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

func roundShift7(v int) int { return (v + 64) >> 7 }

// safeGet returns src[i] clamped to valid range.
func safeGet(src []byte, i int) byte {
	if i < 0 {
		return src[0]
	}
	if i >= len(src) {
		return src[len(src)-1]
	}
	return src[i]
}

// convolveH1 applies a 1-D horizontal filter at src[base] (writing one pixel).
// base is the index of the "center-left" input pixel (tap[3]).
func convolveH1(src []byte, base int, filt [8]int16) byte {
	sum := int(filt[0])*int(safeGet(src, base-3)) +
		int(filt[1])*int(safeGet(src, base-2)) +
		int(filt[2])*int(safeGet(src, base-1)) +
		int(filt[3])*int(safeGet(src, base)) +
		int(filt[4])*int(safeGet(src, base+1)) +
		int(filt[5])*int(safeGet(src, base+2)) +
		int(filt[6])*int(safeGet(src, base+3)) +
		int(filt[7])*int(safeGet(src, base+4))
	return clampPixel(roundShift7(sum))
}

// convolveV1 applies a 1-D vertical filter at src[base] (writing one pixel).
// base is the index of the "center-top" input pixel (tap[3]).
func convolveV1(src []byte, base, stride int, filt [8]int16) byte {
	sum := int(filt[0])*int(safeGet(src, base-3*stride)) +
		int(filt[1])*int(safeGet(src, base-2*stride)) +
		int(filt[2])*int(safeGet(src, base-1*stride)) +
		int(filt[3])*int(safeGet(src, base)) +
		int(filt[4])*int(safeGet(src, base+1*stride)) +
		int(filt[5])*int(safeGet(src, base+2*stride)) +
		int(filt[6])*int(safeGet(src, base+3*stride)) +
		int(filt[7])*int(safeGet(src, base+4*stride))
	return clampPixel(roundShift7(sum))
}

// convolveH fills dst[0..w*h] applying the horizontal filter to src.
// srcBase is the index of (row 0, col 0) of the source block.
func convolveH(src []byte, srcBase, srcStride int, dst []byte, dstStride int, filt [8]int16, w, h int) {
	for r := 0; r < h; r++ {
		rowBase := srcBase + r*srcStride
		outRow := dst[r*dstStride:]
		for c := 0; c < w; c++ {
			outRow[c] = convolveH1(src, rowBase+c, filt)
		}
	}
}

// convolveV fills dst[0..w*h] applying the vertical filter to src.
// srcBase is the index of (row 0, col 0) of the source block.
func convolveV(src []byte, srcBase, srcStride int, dst []byte, dstStride int, filt [8]int16, w, h int) {
	for r := 0; r < h; r++ {
		rowBase := srcBase + r*srcStride
		outRow := dst[r*dstStride:]
		for c := 0; c < w; c++ {
			outRow[c] = convolveV1(src, rowBase+c, srcStride, filt)
		}
	}
}

// Predict performs motion compensation for a single reference frame.
//
// src is the full reference plane buffer.  srcBase is the linear index of
// the integer-pixel origin of the prediction block (top-left corner).
// srcStride is the reference plane's row stride in bytes.
// mx, my are fractional sub-pixel offsets in 1/16-pel units (0 = integer).
// filt selects the interpolation filter type.
// w, h are the block dimensions; dst receives the prediction (dstStride bytes/row).
func Predict(src []byte, srcBase, srcStride, w, h, mx, my int, filt Filter, dst []byte, dstStride int) {
	hk := filterKernels[filt][mx]
	vk := filterKernels[filt][my]

	switch {
	case mx == 0 && my == 0:
		srcLen := len(src)
		for r := 0; r < h; r++ {
			start := srcBase + r*srcStride
			end := start + w
			if end > srcLen {
				end = srcLen
			}
			if start >= srcLen {
				break
			}
			copy(dst[r*dstStride:r*dstStride+w], src[start:end])
		}

	case my == 0:
		convolveH(src, srcBase, srcStride, dst, dstStride, hk, w, h)

	case mx == 0:
		convolveV(src, srcBase, srcStride, dst, dstStride, vk, w, h)

	default:
		// 2-D: horizontal pass (h+7 rows) into a clamped 8-bit intermediate, then
		// the vertical pass. libvpx's vpx_convolve8 clips the horizontal result to
		// uint8 (clip_pixel into a uint8 temp buffer) before the vertical pass.
		rows := h + 7
		temp := make([]byte, rows*w)
		// H pass on rows [-3 .. h+3] of src → temp row 0 .. h+6.
		convolveH(src, srcBase-3*srcStride, srcStride, temp, w, hk, w, rows)
		// V pass: in temp the "origin" is at row 3, col 0 → tempBase = 3*w.
		convolveV(temp, 3*w, w, dst, dstStride, vk, w, h)
	}
}

// PredictComp performs compound motion compensation: two single-reference
// predictions are averaged sample by sample ((a + b + 1) >> 1).
func PredictComp(
	src0 []byte, srcBase0, stride0, mx0, my0 int, filt0 Filter,
	src1 []byte, srcBase1, stride1, mx1, my1 int, filt1 Filter,
	w, h int,
	dst []byte, dstStride int,
) {
	n := w * h
	tmp0 := make([]byte, n)
	tmp1 := make([]byte, n)
	Predict(src0, srcBase0, stride0, w, h, mx0, my0, filt0, tmp0, w)
	Predict(src1, srcBase1, stride1, w, h, mx1, my1, filt1, tmp1, w)
	for r := 0; r < h; r++ {
		d := dst[r*dstStride:]
		a := tmp0[r*w:]
		b := tmp1[r*w:]
		for c := 0; c < w; c++ {
			d[c] = byte((int(a[c]) + int(b[c]) + 1) >> 1)
		}
	}
}
