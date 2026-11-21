package loopfilter

import "fmt"

// FilterFrame16 is FilterFrame for the high-bit-depth uint16 plane (same edge
// selection and ordering; only the pixel type and kernel differ). Call
// SetBitDepth16(bd) first.

func FilterFrame16(buf []uint16, stride, w4, h4, sbN4, sharpness int,
	infoAt func(r4, c4 int) (level int, txSize uint8, skipLF, predH, predV bool)) {

	// 4:2:0 chroma with an odd number of 8×8 MI columns/rows: the last chroma
	// 8×8 is only half inside the MI grid (w4/h4 odd). libvpx vp9_adjust_mask
	// then (a) replaces the 16-wide filter on that column's vertical edges /
	// that row's horizontal edges by the 8-wide one ("we don't apply a wide
	// loop filter on the last uv column/row") and (b) drops the internal 4×4
	// mask of the last column (mask_uv_int), which also removes that column's
	// internal HORIZONTAL edges.
	halfCol, halfRow := -1, -1
	if sbN4 == 8 {
		if w4&1 == 1 {
			halfCol = w4 - 1
		}
		if h4&1 == 1 {
			halfRow = h4 - 1
		}
	}

	for sbR := 0; sbR < h4; sbR += sbN4 {
		for sbC := 0; sbC < w4; sbC += sbN4 {
			rEnd := sbR + sbN4
			if rEnd > h4 {
				rEnd = h4
			}
			cEnd := sbC + sbN4
			if cEnd > w4 {
				cEnd = w4
			}

			// Vertical edges of this SB.
			for c4 := sbC; c4 < cEnd; c4++ {
				if c4 == 0 {
					continue
				}
				for r4 := sbR; r4 < rEnd; r4++ {
					level, txSize, skipLF, _, predV := infoAt(r4, c4)
					if level == 0 || c4&((1<<txSize)-1) != 0 {
						continue
					}
					// A skipped inter block only sets the prediction-block boundary
					// mask (libvpx build_masks returns before the TX / internal-4×4
					// masks); internal transform edges are not filtered.
					if skipLF && !predV {
						continue
					}
					p := ParamsForLevel(level, sharpness)
					w := txWidth(txSize)
					// libvpx vp9_adjust_mask: a TX_4X4 edge on a 32-px grid line
					// (left_border) is promoted to an 8-tap filter ("at least 8 tap
					// on every 32x32"). c4%8==0 ⇒ pixel col is a multiple of 32.
					if txSize == 0 && c4%8 == 0 {
						w = 8
					}
					if c4 == halfCol && w == 16 {
						w = 8
					}
					if DumpW != nil {
						fmt.Fprintf(DumpW, "%d V %d %d %d\n", DumpPlane, r4*4, c4*4, w)
					}
					base := r4*4*stride + c4*4
					for i := 0; i < 4; i++ {
						FilterEdge4_16(buf, base+i*stride, 1, w, p)
					}
				}
			}

			// Horizontal edges of this SB.
			for r4 := sbR; r4 < rEnd; r4++ {
				if r4 == 0 {
					continue
				}
				for c4 := sbC; c4 < cEnd; c4++ {
					level, txSize, skipLF, predH, _ := infoAt(r4, c4)
					if level == 0 || r4&((1<<txSize)-1) != 0 {
						continue
					}
					if skipLF && !predH {
						continue
					}
					if c4 == halfCol && txSize == 0 && r4&1 == 1 {
						continue
					}
					p := ParamsForLevel(level, sharpness)
					w := txWidth(txSize)
					// libvpx vp9_adjust_mask above_border: TX_4X4 horizontal edge on
					// a 32-px grid line is promoted to an 8-tap filter.
					if txSize == 0 && r4%8 == 0 {
						w = 8
					}
					if r4 == halfRow && w == 16 {
						w = 8
					}
					if DumpW != nil {
						fmt.Fprintf(DumpW, "%d H %d %d %d\n", DumpPlane, r4*4, c4*4, w)
					}
					base := r4*4*stride + c4*4
					for i := 0; i < 4; i++ {
						FilterEdge4_16(buf, base+i, stride, w, p)
					}
				}
			}
		}
	}
}
