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
// sub-pixel filter access.  VP9 requires at least 3; we use 8 to match
// libvpx's VP9_INTERP_EXTEND.
const BORDER = 8

// Frame is one decoded reference frame (4:2:0 planar, 8-bit).
type Frame struct {
	// Y, U, V are the luma and chroma planes including their borders.
	Y, U, V []byte

	// Stride is the row stride (in bytes) of each plane.
	// For Y:   stride = width  + 2*BORDER
	// For U/V: stride = width/2 + 2*BORDER  (4:2:0)
	YStride, UVStride int

	// Width and height of the visible picture in pixels.
	Width, Height int
}

// NewFrame allocates a zero-initialised Frame for a 4:2:0 picture.
func NewFrame(width, height int) *Frame {
	// Round up to even for 4:2:0 chroma sub-sampling.
	if width&1 != 0 {
		width++
	}
	if height&1 != 0 {
		height++
	}

	yStride := width + 2*BORDER
	uvStride := width/2 + 2*BORDER

	yRows := height + 2*BORDER
	uvRows := height/2 + 2*BORDER

	return &Frame{
		Y:        make([]byte, yRows*yStride),
		U:        make([]byte, uvRows*uvStride),
		V:        make([]byte, uvRows*uvStride),
		YStride:  yStride,
		UVStride: uvStride,
		Width:    width,
		Height:   height,
	}
}

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
	extendPlane(f.U, f.Width/2, f.Height/2, f.UVStride)
	extendPlane(f.V, f.Width/2, f.Height/2, f.UVStride)
}

// extendPlane fills the BORDER-pixel margins of one plane by replicating
// the nearest edge pixel.
func extendPlane(buf []byte, width, height, stride int) {
	// Top and bottom border rows are filled after left/right so that corners
	// are also covered.

	base := BORDER*stride + BORDER

	// Left and right borders for each visible row.
	for r := 0; r < height; r++ {
		rowStart := base + r*stride
		left := buf[rowStart]
		right := buf[rowStart+width-1]
		for i := 1; i <= BORDER; i++ {
			buf[rowStart-i] = left
			buf[rowStart+width-1+i] = right
		}
	}

	// Copy top border rows from row 0 (including extended left/right).
	srcRow := base - BORDER // start of visible row 0 extended line
	rowLen := width + 2*BORDER
	for i := 1; i <= BORDER; i++ {
		dstRow := srcRow - i*stride
		copy(buf[dstRow:dstRow+rowLen], buf[srcRow:srcRow+rowLen])
	}

	// Copy bottom border rows from the last visible row.
	lastRow := base + (height-1)*stride - BORDER
	for i := 1; i <= BORDER; i++ {
		dstRow := lastRow + i*stride
		copy(buf[dstRow:dstRow+rowLen], buf[lastRow:lastRow+rowLen])
	}
}
