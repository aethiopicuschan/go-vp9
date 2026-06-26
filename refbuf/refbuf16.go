package refbuf

// Frame16 is one decoded reference frame with 16-bit samples (VP9 profile 2/3,
// 10/12-bit). It mirrors Frame but stores []uint16 planes; the 8-bit Frame is
// left untouched so its bit-exact pipeline is unaffected.
type Frame16 struct {
	Y, U, V           []uint16
	YStride, UVStride int
	Width, Height     int
	BitDepth          int // 10 or 12
}

// NewFrame16 allocates a zero-initialised 16-bit 4:2:0 frame.
func NewFrame16(width, height, bitDepth int) *Frame16 {
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
	return &Frame16{
		Y:        make([]uint16, yRows*yStride),
		U:        make([]uint16, uvRows*uvStride),
		V:        make([]uint16, uvRows*uvStride),
		YStride:  yStride,
		UVStride: uvStride,
		Width:    width,
		Height:   height,
		BitDepth: bitDepth,
	}
}

func (f *Frame16) YBase() int { return BORDER*f.YStride + BORDER }
func (f *Frame16) UBase() int { return BORDER*f.UVStride + BORDER }
func (f *Frame16) VBase() int { return BORDER*f.UVStride + BORDER }

// ExtendBorders16 replicates each visible edge into the border (see Frame).
func ExtendBorders16(f *Frame16) {
	extendPlane16(f.Y, f.Width, f.Height, f.YStride)
	extendPlane16(f.U, f.Width/2, f.Height/2, f.UVStride)
	extendPlane16(f.V, f.Width/2, f.Height/2, f.UVStride)
}

func extendPlane16(buf []uint16, width, height, stride int) {
	base := BORDER*stride + BORDER
	for r := 0; r < height; r++ {
		rowStart := base + r*stride
		left := buf[rowStart]
		right := buf[rowStart+width-1]
		for i := 1; i <= BORDER; i++ {
			buf[rowStart-i] = left
			buf[rowStart+width-1+i] = right
		}
	}
	srcRow := base - BORDER
	rowLen := width + 2*BORDER
	for i := 1; i <= BORDER; i++ {
		dstRow := srcRow - i*stride
		copy(buf[dstRow:dstRow+rowLen], buf[srcRow:srcRow+rowLen])
	}
	lastRow := base + (height-1)*stride - BORDER
	for i := 1; i <= BORDER; i++ {
		dstRow := lastRow + i*stride
		copy(buf[dstRow:dstRow+rowLen], buf[lastRow:lastRow+rowLen])
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
