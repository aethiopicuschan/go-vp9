// Package refbuf manages VP9 reference frame buffers (spec §8.9.1).
//
// VP9 maintains 8 named reference frame slots (indices 0..7).  Each frame
// header carries a refresh_frame_flags bitmask that tells the decoder which
// slots to overwrite with the just-decoded frame.  Three of the 8 slots have
// special names used during prediction:
//
//	LAST_FRAME   = slot pointed to by ref_frame_idx[0]
//	GOLDEN_FRAME = slot pointed to by ref_frame_idx[1]
//	ALTREF_FRAME = slot pointed to by ref_frame_idx[2]
//
// The three per-frame indices are decoded from the frame header; this package
// stores them in Refs and exposes accessor methods.
//
// # Frame layout
//
// Each Frame stores one luma (Y) plane and two chroma (U, V) planes.  All
// planes have an BORDER-pixel border on every side (VP9 requires ≥ 8 pixels;
// we use 8).  The pixel at position (row, col) of an N-wide plane is at:
//
//	buf[BORDER*stride + BORDER + row*stride + col]
//
// where stride = width + 2*BORDER.
package refbuf

// BORDER is the number of pixels added on every side of each plane for
// sub-pixel filter access.  The 8-tap filter needs ±3..+4, but inter motion
// vectors point past the frame edge, and the MC reads the reference's border
// there.  libvpx's decoder carries a 32-pixel border and falls back to an
// on-demand edge-replicating copy (extend_and_predict) for reads beyond it.
// We instead carry a wide border (replicating the frame edge, identical result
// to extend_and_predict) so any MV within BORDER pixels of the edge is exact.
const BORDER = 96

// Frame is one decoded reference frame (planar, 8-bit).
type Frame struct {
	// Y, U, V are the luma and chroma planes including their borders.
	Y, U, V []byte

	// Stride is the row stride (in bytes) of each plane.
	// For Y:   stride = width (rounded up to even when SubX=1) + 2*BORDER
	// For U/V: stride = UVWidth() + 2*BORDER
	YStride, UVStride int

	// Width and Height are the true size of the visible picture in pixels
	// (libvpx y_crop_width / y_crop_height). They may be odd; the chroma
	// planes then hold UVWidth() x UVHeight() = ceil(Width/2) x ceil(Height/2)
	// samples on each subsampled axis.
	Width, Height int

	// SubX, SubY are the chroma subsampling shifts (VP9 subsampling_x/y):
	// 1,1 = 4:2:0 (what NewFrame allocates), 0,0 = 4:4:4, 1,0 = 4:2:2,
	// 0,1 = 4:4:0. The visible chroma planes are UVWidth() x UVHeight().
	SubX, SubY int
}

// NewFrame allocates a zero-initialised Frame for a 4:2:0 picture.
func NewFrame(width, height int) *Frame {
	return NewFrameSS(width, height, 1, 1)
}

// NewFrameSS allocates a zero-initialised Frame with chroma subsampling
// shifts subX, subY (each 0 or 1).
func NewFrameSS(width, height, subX, subY int) *Frame {
	// The planes are allocated for the size rounded up to even along each
	// subsampled axis; Width and Height keep the true picture size.
	allocW := width + width&subX
	allocH := height + height&subY

	yStride := allocW + 2*BORDER
	uvStride := allocW>>uint(subX) + 2*BORDER

	yRows := allocH + 2*BORDER
	uvRows := allocH>>uint(subY) + 2*BORDER

	return &Frame{
		Y:        make([]byte, yRows*yStride),
		U:        make([]byte, uvRows*uvStride),
		V:        make([]byte, uvRows*uvStride),
		YStride:  yStride,
		UVStride: uvStride,
		Width:    width,
		Height:   height,
		SubX:     subX,
		SubY:     subY,
	}
}

// UVWidth returns the visible width of the U and V planes.
func (f *Frame) UVWidth() int { return (f.Width + f.SubX) >> uint(f.SubX) }

// UVHeight returns the visible height of the U and V planes.
func (f *Frame) UVHeight() int { return (f.Height + f.SubY) >> uint(f.SubY) }

// YBase returns the linear index of pixel (0,0) in the Y plane.
func (f *Frame) YBase() int { return BORDER*f.YStride + BORDER }

// UBase returns the linear index of pixel (0,0) in the U plane.
func (f *Frame) UBase() int { return BORDER*f.UVStride + BORDER }

// VBase returns the linear index of pixel (0,0) in the V plane.
func (f *Frame) VBase() int { return BORDER*f.UVStride + BORDER }

// Buffer holds VP9's 8 named reference frame slots.
// Slot indices 0..7 correspond to the hardware frame buffer pool.
// The three active reference frames are identified by Refs[0..2].
type Buffer struct {
	slots [8]*Frame

	// Refs[0..2] are the slot indices for LAST, GOLDEN, ALTREF respectively.
	// Values are in [0,7]; default to 0 before any frame is decoded.
	Refs [3]int
}

// Get returns the frame currently stored in the given slot index (0..7).
// Returns nil if the slot has not been filled yet.
func (b *Buffer) Get(slot int) *Frame {
	if slot < 0 || slot >= len(b.slots) {
		return nil
	}
	return b.slots[slot]
}

// Last returns the LAST_FRAME reference (slot Refs[0]).
func (b *Buffer) Last() *Frame { return b.slots[b.Refs[0]] }

// Golden returns the GOLDEN_FRAME reference (slot Refs[1]).
func (b *Buffer) Golden() *Frame { return b.slots[b.Refs[1]] }

// AltRef returns the ALTREF_FRAME reference (slot Refs[2]).
func (b *Buffer) AltRef() *Frame { return b.slots[b.Refs[2]] }

// Update stores frame f into each slot whose bit is set in refreshMask
// (refreshMask bit k → slot k, per VP9 spec §8.9.1).
// After the update it also applies ExtendBorders to keep each refreshed
// frame's border consistent.
func (b *Buffer) Update(frame *Frame, refreshMask uint8) {
	for i := 0; i < 8; i++ {
		if refreshMask&(1<<uint(i)) != 0 {
			b.slots[i] = frame
		}
	}
	ExtendBorders(frame)
}

// SetRef updates one of the three active reference indices (idx=0 for LAST,
// 1 for GOLDEN, 2 for ALTREF) to point to slotIndex.
func (b *Buffer) SetRef(idx, slotIndex int) {
	if idx >= 0 && idx < 3 {
		b.Refs[idx] = slotIndex & 7
	}
}

// ExtendBorders replicates the edge pixels of each plane into the border
// region so that sub-pixel filters always have valid data to read.
func ExtendBorders(f *Frame) {
	extendPlane(f.Y, f.Width, f.Height, f.YStride)
	extendPlane(f.U, f.UVWidth(), f.UVHeight(), f.UVStride)
	extendPlane(f.V, f.UVWidth(), f.UVHeight(), f.UVStride)
}

// extendPlane fills the BORDER-pixel margins of one plane by replicating
// the nearest edge pixel.
func extendPlane(buf []byte, width, height, stride int) {
	base := BORDER*stride + BORDER
	// The plane may be allocated wider/taller than the visible width×height
	// (odd luma sizes are rounded up to even): replicate from the true edge
	// out to the end of the allocation, as libvpx extends references from
	// y_crop_width / y_crop_height.
	right := stride - BORDER - width
	bottom := len(buf)/stride - BORDER - height

	// Left and right borders for each visible row.
	for r := 0; r < height; r++ {
		rowStart := base + r*stride
		l := buf[rowStart]
		e := buf[rowStart+width-1]
		for i := 1; i <= BORDER; i++ {
			buf[rowStart-i] = l
		}
		for i := 1; i <= right; i++ {
			buf[rowStart+width-1+i] = e
		}
	}

	// Top border rows from row 0 (including the extended left/right).
	srcRow := base - BORDER
	for i := 1; i <= BORDER; i++ {
		dstRow := srcRow - i*stride
		copy(buf[dstRow:dstRow+stride], buf[srcRow:srcRow+stride])
	}

	// Bottom border rows from the last visible row.
	lastRow := base + (height-1)*stride - BORDER
	for i := 1; i <= bottom; i++ {
		dstRow := lastRow + i*stride
		copy(buf[dstRow:dstRow+stride], buf[lastRow:lastRow+stride])
	}
}
