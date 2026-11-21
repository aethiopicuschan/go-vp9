package loopfilter

// FilterFrameNon420 applies the loop filter to one chroma plane of a 4:2:2
// (ssX=1, ssY=0) or 4:4:0 (ssX=0, ssY=1) frame — a port of libvpx
// vp9_filter_block_plane_non420, which handles every chroma format that has
// no dedicated mask path (4:2:0 and 4:4:4 go through FilterFrame).
//
// The plane is processed superblock by superblock (raster); within a
// superblock all vertical edges are filtered first, then all horizontal ones.
// The unit is one 8×8 block of plane pixels. Each unit takes its level,
// transform size and skip flag from a single mode-info (8×8 luma) cell: the
// top-left one of the cells it covers, i.e. only every (1<<ssY)-th row and
// (1<<ssX)-th column of the mode-info grid is sampled.
//
// infoAt(miRow, miCol) returns for that cell: the filter level, the chroma
// transform size, skip (skipped inter block), and whether the cell is in the
// first column (edgeLeft) / first row (edgeAbove) of its coding block.
//
// line filters one line of samples across an edge: q0 is the index of the
// first sample past the edge relative to the plane's (0,0) sample, step the
// distance between samples across the edge, width the filter width (4/8/16).
// The caller binds it to FilterEdge4 or FilterEdge4_16 on its plane.
func FilterFrameNon420(stride, miRows, miCols, ssX, ssY, sharpness int,
	infoAt func(miRow, miCol int) (level int, uvTx uint8, skip, edgeLeft, edgeAbove bool),
	line func(q0, step, width int, p Params)) {

	rowStep, colStep := 1<<uint(ssY), 1<<uint(ssX)
	for sbR := 0; sbR < miRows; sbR += 8 {
		for sbC := 0; sbC < miCols; sbC += 8 {
			// Per unit [mi row in SB][plane unit column]: filter level, width of
			// the filter on the unit's top edge (0 = none) and whether its
			// internal 4×4 edges are filtered.
			var lvl, topW [8][8]int
			var int4 [8][8]bool
			y0 := (sbR * 8) >> uint(ssY)
			x0 := (sbC * 8) >> uint(ssX)

			// Vertical edges (libvpx filter_selectively_vert, one row of
			// units at a time, left to right).
			for r := 0; r < 8 && sbR+r < miRows; r += rowStep {
				y := y0 + (r>>uint(ssY))*8
				for c := 0; c < 8 && sbC+c < miCols; c += colStep {
					level, tx, skip, edgeLeft, edgeAbove := infoAt(sbR+r, sbC+c)
					if level == 0 {
						continue
					}
					u := c >> uint(ssX)
					ur := r >> uint(ssY)
					skipC := skip && !edgeLeft
					skipR := skip && !edgeAbove
					// A subsampled unit in the last (odd) mode-info column/row
					// has only 4 samples inside the frame.
					borderC := ssX != 0 && sbC+c == miCols-1
					borderR := ssY != 0 && sbR+r == miRows-1
					leftW, aboveW, inner := 0, 0, false
					if tx >= 2 { // TX_16X16 / TX_32X32
						align := 1
						if tx == 3 {
							align = 3
						}
						if !skipC && u&align == 0 {
							leftW = 16
							if borderC {
								leftW = 8
							}
						}
						if !skipR && ur&align == 0 {
							aboveW = 16
							if borderR {
								aboveW = 8
							}
						}
					} else {
						// At least an 8-wide filter on every 32-sample boundary.
						if !skipC {
							leftW = 4
							if tx == 1 || u&3 == 0 {
								leftW = 8
							}
						}
						if !skipR {
							aboveW = 4
							if tx == 1 || ur&3 == 0 {
								aboveW = 8
							}
						}
						inner = !skip && tx == 0 && !borderC
					}
					lvl[r][u], topW[r][u], int4[r][u] = level, aboveW, inner

					p := ParamsForLevel(level, sharpness)
					base := y*stride + x0 + u*8
					if leftW != 0 && sbC+c != 0 {
						for i := 0; i < 8; i++ {
							line(base+i*stride, 1, leftW, p)
						}
					}
					if inner {
						for i := 0; i < 8; i++ {
							line(base+4+i*stride, 1, 4, p)
						}
					}
				}
			}

			// Horizontal edges (libvpx filter_selectively_horiz).
			for r := 0; r < 8 && sbR+r < miRows; r += rowStep {
				y := y0 + (r>>uint(ssY))*8
				borderR := ssY != 0 && sbR+r == miRows-1
				for u := 0; u < 8; u++ {
					if lvl[r][u] == 0 {
						continue
					}
					p := ParamsForLevel(lvl[r][u], sharpness)
					base := y*stride + x0 + u*8
					if topW[r][u] != 0 && sbR+r != 0 {
						for i := 0; i < 8; i++ {
							line(base+i, stride, topW[r][u], p)
						}
					}
					if int4[r][u] && !borderR {
						for i := 0; i < 8; i++ {
							line(base+4*stride+i, stride, 4, p)
						}
					}
				}
			}
		}
	}
}
