package loopfilter

import (
	"fmt"
	"io"
)

// DumpW/DumpPlane: debug edge-trace hook. When DumpW != nil, FilterFrame writes
// one line per filtered edge ("<plane> <V|H> <y> <x> <width>") to compare the
// edge-selection against libvpx's instrumented loop filter.
var (
	DumpW     io.Writer
	DumpPlane int
)

// FilterPlane applies the VP9 loop filter to one luma or chroma plane.
//
// Each 4x4 block's edge is filtered with the given Params. For simplicity
// this API takes a uniform Params for the whole plane; a full implementation
// would supply per-block params derived from filter_level, ref_delta, etc.
//
// The filter visits:
//  1. All vertical edges (column boundaries at multiples of 4, left to right).
//  2. All horizontal edges (row boundaries at multiples of 4, top to bottom).
//
// For each 4-pixel column/row along an edge the EdgeWidth function determines
// which kernel to apply (4, 8, or 16 wide) based on the block size at that
// edge (bsLog2: 1=8px, 2=16px, 3=32/64px boundary).
//
// buf   – plane pixel data, row-major.
// stride – row stride in bytes (may be > width).
// width, height – plane dimensions in pixels.
// bsLog2 – log2 of the superblock size in pixels (2=4px blocks, 3=8px…).
//
//	Controls whether the wider 8/14-tap filters are engaged.
//	Pass 2 for chroma or for simple filter mode.
func FilterPlane(buf []byte, stride, width, height, bsLog2 int, p Params) {
	bs := 1 << bsLog2 // block size in pixels (4, 8, 16, 32)

	// 1. Vertical edges: for each row r and each column boundary x (x=bs, 2bs, …)
	//    filter 4-pixel-tall strips along the column.
	for x := bs; x < width; x += bs {
		// Edge width: wider at superblock boundaries.
		w := edgeWidth(x, bsLog2)
		for r := 0; r+4 <= height; r += 4 {
			// q0 for each of the 4 rows: address buf[r+i][x] = (r+i)*stride + x.
			for i := 0; i < 4; i++ {
				q0 := (r+i)*stride + x
				// step=1: adjacent pixels within a row (vertical edge = horizontal step).
				FilterEdge4(buf, q0, 1, w, p)
			}
		}
	}

	// 2. Horizontal edges: for each row boundary y and each column c.
	for y := bs; y < height; y += bs {
		w := edgeWidth(y, bsLog2)
		for c := 0; c+4 <= width; c += 4 {
			for i := 0; i < 4; i++ {
				q0 := y*stride + c + i
				// step=stride: adjacent pixels in the same column (horizontal edge = vertical step).
				FilterEdge4(buf, q0, stride, w, p)
			}
		}
	}
}

// FilterPlaneWithLevels applies the VP9 loop filter with per-4×4-block level lookup.
// paramsAt(row4, col4) returns the Params for the block at that 4×4 grid position.
// Blocks with level=0 produce Params{} from paramsAt; FilterEdge4 no-ops when BLimit=0.
func FilterPlaneWithLevels(buf []byte, stride, width, height, bsLog2 int, paramsAt func(row4, col4 int) Params) {
	bs := 1 << bsLog2

	for x := bs; x < width; x += bs {
		w := edgeWidth(x, bsLog2)
		for r := 0; r+4 <= height; r += 4 {
			p := paramsAt(r/4, x/4)
			for i := 0; i < 4; i++ {
				FilterEdge4(buf, (r+i)*stride+x, 1, w, p)
			}
		}
	}

	for y := bs; y < height; y += bs {
		w := edgeWidth(y, bsLog2)
		for c := 0; c+4 <= width; c += 4 {
			p := paramsAt(y/4, c/4)
			for i := 0; i < 4; i++ {
				FilterEdge4(buf, y*stride+c+i, stride, w, p)
			}
		}
	}
}

// edgeWidth returns the filter kernel width (4, 8, or 16) for an edge at pixel
// position pos with a block-size log2 of bsLog2.
// Superblock (64px) and 32px boundaries use 16-tap; 8px use 8-tap; 4px use 4-tap.
func edgeWidth(pos, bsLog2 int) int {
	bs := 1 << bsLog2
	// Check if pos aligns to larger block boundaries.
	if bsLog2 >= 4 && pos%(bs*4) == 0 { // 64px
		return 16
	}
	if bsLog2 >= 3 && pos%(bs*2) == 0 { // 32px
		return 16
	}
	if bsLog2 >= 2 && pos%bs == 0 { // 8px+
		if bsLog2 >= 2 {
			return 8
		}
	}
	return 4
}

// txWidth maps a transform size to the loop-filter kernel width (libvpx caps at
// 16: TX_32X32 also uses the 16-wide filter via vp9_adjust_mask).
func txWidth(txSize uint8) int {
	switch txSize {
	case 0:
		return 4
	case 1:
		return 8
	default:
		return 16
	}
}

// FilterFrame applies the VP9 loop filter to one plane in libvpx's order:
// superblock by superblock (raster), and within each SB ALL vertical edges then
// ALL horizontal edges. sbN4 is the superblock size in 4×4 units for this plane
// (16 for luma 64-px SBs, 8 for 4:2:0 chroma). This matters at SB boundaries,
// where a later SB's vertical edge reads the previous SB's already
// horizontally-filtered pixels. infoAt(r4,c4) returns the block's loop-filter
// level and transform size (for this plane) covering 4×4 (r4,c4).
//
// An edge at the left/top of (r4,c4) is on a transform boundary iff the
// coordinate is a multiple of the block's TX size in 4×4 units (that subsumes
// prediction-block boundaries).
func FilterFrame(buf []byte, stride, w4, h4, sbN4, sharpness int,
	infoAt func(r4, c4 int) (level int, txSize uint8, skipLF, predH, predV bool)) {

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
					if DumpW != nil {
						fmt.Fprintf(DumpW, "%d V %d %d %d\n", DumpPlane, r4*4, c4*4, w)
					}
					base := r4*4*stride + c4*4
					for i := 0; i < 4; i++ {
						FilterEdge4(buf, base+i*stride, 1, w, p)
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
					p := ParamsForLevel(level, sharpness)
					w := txWidth(txSize)
					// libvpx vp9_adjust_mask above_border: TX_4X4 horizontal edge on
					// a 32-px grid line is promoted to an 8-tap filter.
					if txSize == 0 && r4%8 == 0 {
						w = 8
					}
					if DumpW != nil {
						fmt.Fprintf(DumpW, "%d H %d %d %d\n", DumpPlane, r4*4, c4*4, w)
					}
					base := r4*4*stride + c4*4
					for i := 0; i < 4; i++ {
						FilterEdge4(buf, base+i, stride, w, p)
					}
				}
			}
		}
	}
}
