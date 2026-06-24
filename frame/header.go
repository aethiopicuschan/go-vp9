// Package frame parses the VP9 uncompressed frame header (spec §7.2).
// The header lives at the start of each frame payload before the compressed
// header and tile data. It carries frame type, dimensions, reference frames,
// colour space, and the offset to the compressed header.
package frame

import (
	"errors"
)

// FrameType distinguishes key frames from inter frames.
type FrameType int

const (
	FrameKey   FrameType = 0
	FrameInter FrameType = 1
)

// ColorSpace values from VP9 spec §7.2.2.
const (
	ColorspaceBT601    = 1
	ColorspaceBT709    = 2
	ColorspaceSMPTE170 = 3
	ColorspaceSMPTE240 = 4
	ColorspaceBT2020   = 5
	ColorspaceSRGB     = 7
)

// Header is the decoded VP9 uncompressed frame header.
type Header struct {
	// Size is the number of bytes consumed by this uncompressed header.
	// Tile data starts at payload[Size + CompressedHeaderSize].
	Size int

	Profile           int // 0-3
	ShowExistingFrame bool
	ShowExistingIdx   int       // if ShowExistingFrame
	FrameType         FrameType // Key or Inter
	ShowFrame         bool
	ErrorResilient    bool

	// Key-frame sync code present and valid
	KeyframeSyncOK bool

	// Colour / pixel format
	ColorSpace   int
	ColorRange   bool // false=studio, true=full
	SubsamplingX bool // chroma sub-sampling
	SubsamplingY bool
	BitDepth     int // 8, 10, 12

	// Frame dimensions (after resolution update)
	FrameWidth  int
	FrameHeight int

	// Display dimensions (may differ from frame dimensions)
	DisplayWidth   int
	DisplayHeight  int
	RenderAndFrame bool // true when render size == frame size

	// Reference frames used for this inter frame (indices 0-7 into the ref pool)
	RefFrameIdx  [3]int // last, golden, altref
	RefFrameSign [3]bool

	// Inter-frame motion/filter fields.
	AllowHighPrecisionMV bool
	InterpolationFilter  int // 0=EIGHTTAP 1=SMOOTH 2=SHARP 3=BILINEAR 4=SWITCHABLE

	// Compressed-header size (bytes), as coded in the header.
	// The compressed header immediately follows the uncompressed header.
	CompressedHeaderSize int

	// Tile log2 dimensions.
	TileColsLog2 int
	TileRowsLog2 int

	// Loop filter (spec §7.2.8)
	LoopFilterLevel     int // 0..63
	LoopFilterSharpness int // 0..7
	LFDeltaEnabled      bool
	LFRefDeltas         [4]int // ref_deltas[INTRA/LAST/GOLDEN/ALTREF], range -63..63
	LFModeDeltas        [2]int // mode_deltas[ZEROMV/other], range -63..63

	// Refresh flags (which reference buffer slots get this frame, bit mask 0..255)
	RefreshFrameFlags uint8

	// Quantization (spec §7.2.9)
	BaseQIdx   int // base quantizer index (0 = lossless when all deltas are zero)
	DeltaQYDC  int // Y-plane DC delta, range -7..7
	DeltaQUVDC int // UV-plane DC delta
	DeltaQUVAC int // UV-plane AC delta

	// Segmentation (spec §7.2.10)
	SegEnabled     bool
	SegUpdateMap   bool
	SegTempUpdate  bool
	SegAbsOrDelta  bool         // true = absolute Q; false = delta
	SegTreeProbs   [7]uint8     // valid when SegUpdateMap
	SegPredProbs   [3]uint8     // valid when SegUpdateMap && SegTempUpdate
	SegFeatEnabled [8][4]bool
	SegFeatData    [8][4]int16
}

var (
	errSyncCode  = errors.New("vp9/frame: bad key-frame sync code")
	errTruncated = errors.New("vp9/frame: truncated frame")
)

// bitReader reads bits MSB-first from a byte slice (for the uncompressed header,
// which is NOT bool-coded — it uses raw bits, not the boolean arithmetic coder).
type bitReader struct {
	buf  []byte
	pos  int // byte index
	bit  int // next bit within current byte (7 = MSB)
	done bool
}

func newBitReader(buf []byte) *bitReader {
	return &bitReader{buf: buf, bit: 7}
}

func (b *bitReader) readBit() int {
	if b.pos >= len(b.buf) {
		b.done = true
		return 0
	}
	v := int((b.buf[b.pos] >> uint(b.bit)) & 1)
	b.bit--
	if b.bit < 0 {
		b.bit = 7
		b.pos++
	}
	return v
}

func (b *bitReader) readBits(n int) uint32 {
	var x uint32
	for i := 0; i < n; i++ {
		x = x<<1 | uint32(b.readBit())
	}
	return x
}

// bytePos returns the byte offset of the next unread bit (rounded up).
func (b *bitReader) bytePos() int {
	if b.bit == 7 {
		return b.pos
	}
	return b.pos + 1
}

// Parse decodes the VP9 uncompressed frame header from the start of payload.
// prevWidth/prevHeight are the dimensions of the most recently decoded frame
// (used for inter frames that inherit dimensions via frame_size_with_refs).
// Pass 0 for both on first call; they are ignored for key frames.
// It returns the decoded Header; the caller uses Header.CompressedHeaderSize to
// locate the compressed header immediately after.
func Parse(payload []byte, prevWidth, prevHeight int) (*Header, error) {
	b := newBitReader(payload)
	h := &Header{}

	// frame_marker (2 bits, must be 0b10)
	if b.readBits(2) != 0x2 {
		return nil, errors.New("vp9/frame: bad frame marker")
	}

	// profile_low_bit / profile_high_bit
	profileLow := b.readBit()
	profileHigh := b.readBit()
	h.Profile = profileHigh<<1 | profileLow
	if h.Profile == 3 {
		b.readBit() // reserved_zero
	}

	// show_existing_frame
	if b.readBit() == 1 {
		h.ShowExistingFrame = true
		h.ShowExistingIdx = int(b.readBits(3))
		if b.done {
			return nil, errTruncated
		}
		return h, nil
	}

	h.FrameType = FrameType(b.readBit())
	h.ShowFrame = b.readBit() == 1
	h.ErrorResilient = b.readBit() == 1

	if h.FrameType == FrameKey {
		// sync_code must be 0x49 0x83 0x42
		b0 := b.readBits(8)
		b1 := b.readBits(8)
		b2 := b.readBits(8)
		if b0 != 0x49 || b1 != 0x83 || b2 != 0x42 {
			return nil, errSyncCode
		}
		h.RefreshFrameFlags = 0xFF // key frames refresh all 8 slots
		h.KeyframeSyncOK = true
		parseColorConfig(b, h)
		parseFrameSize(b, h)
		parseRenderSize(b, h)
	} else {
		// Inter frame
		intraOnly := false
		if !h.ShowFrame {
			intraOnly = b.readBit() == 1
		}
		if !h.ErrorResilient {
			b.readBits(2) // reset_frame_context
		}
		if intraOnly {
			b0 := b.readBits(8)
			b1 := b.readBits(8)
			b2 := b.readBits(8)
			if b0 != 0x49 || b1 != 0x83 || b2 != 0x42 {
				return nil, errSyncCode
			}
			h.RefreshFrameFlags = uint8(b.readBits(8))
			parseColorConfig(b, h)
			parseFrameSize(b, h)
			parseRenderSize(b, h)
		} else {
			h.RefreshFrameFlags = uint8(b.readBits(8))
			for i := 0; i < 3; i++ {
				h.RefFrameIdx[i] = int(b.readBits(3))
				h.RefFrameSign[i] = b.readBit() == 1
			}
			parseFrameSizeWithRefs(b, h, prevWidth, prevHeight)
			h.AllowHighPrecisionMV = b.readBit() == 1
			// read_interpolation_filter (§7.2.6)
			if b.readBit() == 1 {
				h.InterpolationFilter = 4 // SWITCHABLE
			} else {
				// literal_to_filter: 0→SMOOTH(1) 1→EIGHTTAP(0) 2→SHARP(2) 3→BILINEAR(3)
				h.InterpolationFilter = [4]int{1, 0, 2, 3}[b.readBits(2)]
			}
		}
	}

	if !h.ErrorResilient {
		b.readBit() // refresh_frame_context
		b.readBit() // frame_parallel_decoding_mode
	}
	b.readBits(2) // frame_context_idx (FRAME_CONTEXTS_LOG2 = 2)

	parseLoopFilterParams(b, h)
	parseQuantizationParams(b, h)
	parseSegmentationParams(b, h)
	parseTileInfo(b, h)

	h.CompressedHeaderSize = int(b.readBits(16))

	if b.done {
		return nil, errTruncated
	}
	h.Size = b.bytePos()
	return h, nil
}

// parseColorConfig reads colour_config() (spec §7.2.2).
func parseColorConfig(b *bitReader, h *Header) {
	if h.Profile >= 2 {
		h.BitDepth = 8
		if b.readBit() == 1 {
			h.BitDepth = 12
		} else {
			h.BitDepth = 10
		}
	} else {
		h.BitDepth = 8
	}

	h.ColorSpace = int(b.readBits(3))
	if h.ColorSpace != ColorspaceSRGB {
		h.ColorRange = b.readBit() == 1
		if h.Profile == 1 || h.Profile == 3 {
			h.SubsamplingX = b.readBit() == 1
			h.SubsamplingY = b.readBit() == 1
			b.readBit() // reserved_zero
		} else {
			h.SubsamplingX = true
			h.SubsamplingY = true
		}
	} else {
		h.ColorRange = true
		if h.Profile == 1 || h.Profile == 3 {
			h.SubsamplingX = false
			h.SubsamplingY = false
			b.readBit() // reserved_zero
		}
	}
}

// parseFrameSize reads frame_size() (spec §7.2.3).
func parseFrameSize(b *bitReader, h *Header) {
	h.FrameWidth = int(b.readBits(16)) + 1
	h.FrameHeight = int(b.readBits(16)) + 1
}

// parseRenderSize reads render_size() (spec §7.2.4).
// render_and_frame_size_equal_flag=1 means render dims = frame dims (no bits follow).
// render_and_frame_size_equal_flag=0 means explicit render_width/height follow.
func parseRenderSize(b *bitReader, h *Header) {
	if b.readBit() == 0 {
		h.DisplayWidth = int(b.readBits(16)) + 1
		h.DisplayHeight = int(b.readBits(16)) + 1
		h.RenderAndFrame = false
	} else {
		h.DisplayWidth = h.FrameWidth
		h.DisplayHeight = h.FrameHeight
		h.RenderAndFrame = true
	}
}

// parseFrameSizeWithRefs reads frame_size_with_refs() for inter frames (§7.2.7).
// prevWidth/prevHeight are the decoded frame dimensions to use when
// found_ref is true (inheriting from a reference frame).
func parseFrameSizeWithRefs(b *bitReader, h *Header, prevWidth, prevHeight int) {
	found := false
	for i := 0; i < 3; i++ {
		if b.readBit() == 1 {
			found = true
			break
		}
	}
	if !found {
		parseFrameSize(b, h)
		parseRenderSize(b, h)
	} else {
		// When using a reference frame's size, render_size is also inherited (not read).
		h.FrameWidth = prevWidth
		h.FrameHeight = prevHeight
		h.DisplayWidth = prevWidth
		h.DisplayHeight = prevHeight
		h.RenderAndFrame = true
	}
}

// su6 reads a 6-bit 2's-complement signed integer (VP9 spec su(6)).
func su6(b *bitReader) int {
	v := int(b.readBits(6))
	if v >= 32 {
		v -= 64
	}
	return v
}

// parseLoopFilterParams reads loop_filter_params() (spec §7.2.8).
func parseLoopFilterParams(b *bitReader, h *Header) {
	h.LoopFilterLevel = int(b.readBits(6))
	h.LoopFilterSharpness = int(b.readBits(3))
	h.LFDeltaEnabled = b.readBit() == 1
	if h.LFDeltaEnabled {
		if b.readBit() == 1 { // mode_ref_delta_update
			for i := 0; i < 4; i++ {
				if b.readBit() == 1 {
					h.LFRefDeltas[i] = su6(b)
				}
			}
			for i := 0; i < 2; i++ {
				if b.readBit() == 1 {
					h.LFModeDeltas[i] = su6(b)
				}
			}
		}
	}
}

// parseQuantizationParams reads quantization_params() (spec §7.2.9).
func parseQuantizationParams(b *bitReader, h *Header) {
	h.BaseQIdx = int(b.readBits(8))
	// readDeltaQ: 1-bit flag; if set, su(4) = 4-bit 2's complement signed int.
	readDeltaQ := func() int {
		if b.readBit() == 0 {
			return 0
		}
		v := int(b.readBits(4))
		if v >= 8 {
			v -= 16
		}
		return v
	}
	h.DeltaQYDC = readDeltaQ()
	h.DeltaQUVDC = readDeltaQ()
	h.DeltaQUVAC = readDeltaQ()
}

// parseSegmentationParams reads segmentation_params() (spec §7.2.10).
func parseSegmentationParams(b *bitReader, h *Header) {
	h.SegEnabled = b.readBit() == 1
	if !h.SegEnabled {
		return
	}
	h.SegUpdateMap = b.readBit() == 1
	if h.SegUpdateMap {
		// 7 segment tree probs; each is preceded by a 1-bit update flag.
		for i := 0; i < 7; i++ {
			if b.readBit() == 1 {
				h.SegTreeProbs[i] = uint8(b.readBits(8))
			} else {
				h.SegTreeProbs[i] = 255
			}
		}
		h.SegTempUpdate = b.readBit() == 1
		if h.SegTempUpdate {
			for i := 0; i < 3; i++ {
				if b.readBit() == 1 {
					h.SegPredProbs[i] = uint8(b.readBits(8))
				} else {
					h.SegPredProbs[i] = 255
				}
			}
		}
	}
	if b.readBit() == 0 { // segmentation_update_data
		return
	}
	h.SegAbsOrDelta = b.readBit() == 1
	featBits := [4]int{8, 6, 2, 0}
	featSigned := [4]bool{true, true, false, false}
	for s := 0; s < 8; s++ {
		for f := 0; f < 4; f++ {
			if b.readBit() == 1 {
				h.SegFeatEnabled[s][f] = true
				val := int16(b.readBits(featBits[f]))
				if featSigned[f] && b.readBit() == 1 {
					val = -val
				}
				h.SegFeatData[s][f] = val
			}
		}
	}
}

// parseTileInfo reads tile_info() (spec §7.2.11).
func parseTileInfo(b *bitReader, h *Header) {
	minLog2TileCols := calcMinLog2TileCols(h.FrameWidth)
	maxLog2TileCols := calcMaxLog2TileCols(h.FrameWidth)
	h.TileColsLog2 = minLog2TileCols
	for h.TileColsLog2 < maxLog2TileCols {
		if b.readBit() == 1 {
			h.TileColsLog2++
		} else {
			break
		}
	}
	h.TileRowsLog2 = 0
	if b.readBit() == 1 {
		h.TileRowsLog2++
		if b.readBit() == 1 {
			h.TileRowsLog2++
		}
	}
}

// calcMinLog2TileCols returns the minimum log2 tile columns for a frame width.
func calcMinLog2TileCols(width int) int {
	sb64Cols := (width + 63) >> 6 // number of 64×64 superblocks wide
	minLog2 := 0
	for (64 << uint(minLog2)) < sb64Cols {
		minLog2++
	}
	return minLog2
}

// calcMaxLog2TileCols returns the maximum log2 tile columns.
// Matches libvpx calc_max_log2_tile_cols: starts at 1, decrements at end.
func calcMaxLog2TileCols(width int) int {
	sb64Cols := (width + 63) >> 6
	maxLog2 := 1
	for (sb64Cols >> uint(maxLog2)) >= 4 {
		maxLog2++
	}
	return maxLog2 - 1
}
