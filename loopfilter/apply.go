package loopfilter

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
//           Controls whether the wider 8/14-tap filters are engaged.
//           Pass 2 for chroma or for simple filter mode.
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
