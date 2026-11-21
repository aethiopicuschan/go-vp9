package refbuf

// Frame16 is one decoded reference frame with 16-bit samples (VP9 profile 2/3,
// 10/12-bit). It mirrors Frame but stores []uint16 planes; the 8-bit Frame is
// left untouched so its bit-exact pipeline is unaffected.
type Frame16 struct {
	Y, U, V           []uint16
	YStride, UVStride int
	Width, Height     int
	BitDepth          int // 10 or 12
	// SubX, SubY are the chroma subsampling shifts (see Frame).
	SubX, SubY int
}

// NewFrame16 allocates a zero-initialised 16-bit 4:2:0 frame.
func NewFrame16(width, height, bitDepth int) *Frame16 {
	return NewFrame16SS(width, height, bitDepth, 1, 1)
}

// NewFrame16SS allocates a zero-initialised 16-bit frame with chroma
// subsampling shifts subX, subY (each 0 or 1).
func NewFrame16SS(width, height, bitDepth, subX, subY int) *Frame16 {
	// Planes are allocated for the even-rounded size; Width and Height keep
	// the true picture size (see Frame).
	allocW := width + width&subX
	allocH := height + height&subY
	yStride := allocW + 2*BORDER
	uvStride := allocW>>uint(subX) + 2*BORDER
	yRows := allocH + 2*BORDER
	uvRows := allocH>>uint(subY) + 2*BORDER
	return &Frame16{
		Y:        make([]uint16, yRows*yStride),
		U:        make([]uint16, uvRows*uvStride),
		V:        make([]uint16, uvRows*uvStride),
		YStride:  yStride,
		UVStride: uvStride,
		Width:    width,
		Height:   height,
		BitDepth: bitDepth,
		SubX:     subX,
		SubY:     subY,
	}
}

// UVWidth returns the visible width of the U and V planes.
func (f *Frame16) UVWidth() int { return (f.Width + f.SubX) >> uint(f.SubX) }

// UVHeight returns the visible height of the U and V planes.
func (f *Frame16) UVHeight() int { return (f.Height + f.SubY) >> uint(f.SubY) }

func (f *Frame16) YBase() int { return BORDER*f.YStride + BORDER }
func (f *Frame16) UBase() int { return BORDER*f.UVStride + BORDER }
func (f *Frame16) VBase() int { return BORDER*f.UVStride + BORDER }

// ExtendBorders16 replicates each visible edge into the border (see Frame).
func ExtendBorders16(f *Frame16) {
	extendPlane16(f.Y, f.Width, f.Height, f.YStride)
	extendPlane16(f.U, f.UVWidth(), f.UVHeight(), f.UVStride)
	extendPlane16(f.V, f.UVWidth(), f.UVHeight(), f.UVStride)
}

func extendPlane16(buf []uint16, width, height, stride int) {
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

// Buffer16 holds VP9's 8 reference-frame slots for 16-bit frames.
type Buffer16 struct {
	slots [8]*Frame16
	Refs  [3]int
}

func (b *Buffer16) Get(slot int) *Frame16 {
	if slot < 0 || slot >= len(b.slots) {
		return nil
	}
	return b.slots[slot]
}
func (b *Buffer16) Last() *Frame16   { return b.slots[b.Refs[0]] }
func (b *Buffer16) Golden() *Frame16 { return b.slots[b.Refs[1]] }
func (b *Buffer16) AltRef() *Frame16 { return b.slots[b.Refs[2]] }

func (b *Buffer16) Update(frame *Frame16, refreshMask uint8) {
	for i := 0; i < 8; i++ {
		if refreshMask&(1<<uint(i)) != 0 {
			b.slots[i] = frame
		}
	}
	ExtendBorders16(frame)
}

func (b *Buffer16) SetRef(idx, slotIndex int) {
	if idx >= 0 && idx < 3 {
		b.Refs[idx] = slotIndex
	}
}
