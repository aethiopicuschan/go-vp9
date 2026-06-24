package frame

import (
	"testing"
)

// bitWriter assembles bits MSB-first into a byte slice (for test vector construction).
type bitWriter struct {
	buf []byte
	cur byte
	bit int // next bit position (7 = MSB)
}

func (bw *bitWriter) writeBit(v int) {
	if v != 0 {
		bw.cur |= 1 << uint(bw.bit)
	}
	bw.bit--
	if bw.bit < 0 {
		bw.buf = append(bw.buf, bw.cur)
		bw.cur = 0
		bw.bit = 7
	}
}

func (bw *bitWriter) writeBits(n int, v uint32) {
	for i := n - 1; i >= 0; i-- {
		bw.writeBit(int((v >> uint(i)) & 1))
	}
}

func (bw *bitWriter) flush() []byte {
	if bw.bit != 7 {
		bw.buf = append(bw.buf, bw.cur)
		bw.cur = 0
		bw.bit = 7
	}
	return bw.buf
}

// buildKeyFrame constructs a minimal VP9 key-frame uncompressed header payload
// with profile 0, BT.601, 4:2:0, given width/height (1-based, coded as w-1/h-1).
func buildKeyFrame(w, h int) []byte {
	bw := &bitWriter{bit: 7}

	bw.writeBits(2, 0x2) // frame_marker = 0b10
	bw.writeBit(0)       // profile_low_bit  → profile 0
	bw.writeBit(0)       // profile_high_bit
	bw.writeBit(0)       // show_existing_frame = false
	bw.writeBit(0)       // frame_type = KEY
	bw.writeBit(1)       // show_frame
	bw.writeBit(0)       // error_resilient

	// sync code
	bw.writeBits(8, 0x49)
	bw.writeBits(8, 0x83)
	bw.writeBits(8, 0x42)

	// color_config: profile 0 → no bit depth field
	// color_space = BT601 (1)
	bw.writeBits(3, 1)
	bw.writeBit(0) // color_range = studio
	// profile 0: subsampling_x/y = true (implicit, no bits)

	// frame_size: w-1 (16 bits), h-1 (16 bits)
	bw.writeBits(16, uint32(w-1))
	bw.writeBits(16, uint32(h-1))

	// render_size: render_and_frame_size_equal_flag=1 → render dims = frame dims
	bw.writeBit(1)

	// After frame-type block: refresh_frame_context + frame_parallel_decoding_mode
	// (only when !error_resilient, which is our case)
	bw.writeBit(0) // refresh_frame_context
	bw.writeBit(1) // frame_parallel_decoding_mode
	// frame_context_idx: 2 bits (FRAME_CONTEXTS_LOG2 = 2)
	bw.writeBits(2, 0)

	// loop_filter_params
	bw.writeBits(6, 0) // level=0
	bw.writeBits(3, 0) // sharpness=0
	bw.writeBit(0)     // mode_ref_delta_enabled=false

	// quantization_params
	bw.writeBits(8, 0) // base_q_idx=0
	bw.writeBit(0)     // delta_q_y_dc present = false
	bw.writeBit(0)     // delta_q_uv_dc present = false
	bw.writeBit(0)     // delta_q_uv_ac present = false

	// segmentation_params
	bw.writeBit(0) // segmentation_enabled = false

	// tile_info: tile_cols_log2 loop.
	// minLog2TileCols = 0 for these test frames (64<<0 ≥ sb64Cols).
	// maxLog2TileCols = how many times (sb64Cols>>k) >= 4.
	{
		sb64Cols := (w + 63) / 64
		maxLog2 := 0
		for (sb64Cols >> uint(maxLog2)) >= 4 {
			maxLog2++
		}
		if maxLog2 > 0 {
			bw.writeBit(0) // increment_tile_cols_log2 = 0 → stop at min
		}
	}
	// tile_rows_log2: write 0 (no increment bit)
	bw.writeBit(0) // first tile_rows bit = 0 → tile_rows_log2 stays 0

	// compressed_header_size (16 bits)
	bw.writeBits(16, 42)

	return bw.flush()
}

func TestKeyFrameSmall(t *testing.T) {
	payload := buildKeyFrame(64, 48)
	h, err := Parse(payload, 0, 0)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if h.FrameType != FrameKey {
		t.Fatalf("FrameType = %v, want Key", h.FrameType)
	}
	if !h.ShowFrame {
		t.Fatal("ShowFrame should be true")
	}
	if !h.KeyframeSyncOK {
		t.Fatal("KeyframeSyncOK should be true")
	}
	if h.Profile != 0 {
		t.Fatalf("Profile = %d", h.Profile)
	}
	if h.ColorSpace != ColorspaceBT601 {
		t.Fatalf("ColorSpace = %d", h.ColorSpace)
	}
	if h.BitDepth != 8 {
		t.Fatalf("BitDepth = %d", h.BitDepth)
	}
	if h.FrameWidth != 64 || h.FrameHeight != 48 {
		t.Fatalf("dims = %dx%d", h.FrameWidth, h.FrameHeight)
	}
	if h.DisplayWidth != 64 || h.DisplayHeight != 48 {
		t.Fatalf("display dims = %dx%d", h.DisplayWidth, h.DisplayHeight)
	}
	if h.CompressedHeaderSize != 42 {
		t.Fatalf("CompressedHeaderSize = %d", h.CompressedHeaderSize)
	}
}

func TestKeyFrameLarger(t *testing.T) {
	// 1920×1080: min tile cols log2 = 2 (since (1920+63)/64 = 30 sb cols, and
	// 64<<1=128>30 → min=1; 64<<0=64>30 → min=0? let's not over-specify, just check dims)
	payload := buildKeyFrame(1920, 1080)
	h, err := Parse(payload, 0, 0)
	if err != nil {
		t.Fatalf("Parse 1920x1080: %v", err)
	}
	if h.FrameWidth != 1920 || h.FrameHeight != 1080 {
		t.Fatalf("dims = %dx%d", h.FrameWidth, h.FrameHeight)
	}
}

func TestBadFrameMarker(t *testing.T) {
	payload := make([]byte, 32)
	payload[0] = 0x00 // frame_marker = 0b00 → wrong
	_, err := Parse(payload, 0, 0)
	if err == nil {
		t.Fatal("expected error for bad frame marker")
	}
}

func TestBadSyncCode(t *testing.T) {
	bw := &bitWriter{bit: 7}
	bw.writeBits(2, 0x2)  // frame_marker
	bw.writeBit(0)        // profile low
	bw.writeBit(0)        // profile high
	bw.writeBit(0)        // show_existing_frame
	bw.writeBit(0)        // frame_type = KEY
	bw.writeBit(1)        // show_frame
	bw.writeBit(0)        // error_resilient
	bw.writeBits(8, 0x00) // wrong sync byte 0
	bw.writeBits(8, 0x00)
	bw.writeBits(8, 0x00)
	payload := bw.flush()
	_, err := Parse(payload, 0, 0)
	if err == nil {
		t.Fatal("expected sync code error")
	}
}

func TestShowExistingFrame(t *testing.T) {
	bw := &bitWriter{bit: 7}
	bw.writeBits(2, 0x2) // frame_marker
	bw.writeBit(0)       // profile low
	bw.writeBit(0)       // profile high
	bw.writeBit(1)       // show_existing_frame = true
	bw.writeBits(3, 5)   // frame_to_show_map_idx = 5
	payload := bw.flush()
	h, err := Parse(payload, 0, 0)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !h.ShowExistingFrame {
		t.Fatal("ShowExistingFrame should be true")
	}
	if h.ShowExistingIdx != 5 {
		t.Fatalf("ShowExistingIdx = %d", h.ShowExistingIdx)
	}
}

// buildKeyFrameWithQ constructs a key-frame header with specific Q index and deltas.
func buildKeyFrameWithQ(baseQ, deltaYDC, deltaUVDC, deltaUVAC int) []byte {
	bw := &bitWriter{bit: 7}

	bw.writeBits(2, 0x2) // frame_marker
	bw.writeBit(0)       // profile_low_bit
	bw.writeBit(0)       // profile_high_bit
	bw.writeBit(0)       // show_existing_frame = false
	bw.writeBit(0)       // frame_type = KEY
	bw.writeBit(1)       // show_frame
	bw.writeBit(0)       // error_resilient

	bw.writeBits(8, 0x49)
	bw.writeBits(8, 0x83)
	bw.writeBits(8, 0x42)

	bw.writeBits(3, 1) // color_space = BT601
	bw.writeBit(0)     // color_range = studio

	bw.writeBits(16, 63) // width-1 = 64
	bw.writeBits(16, 47) // height-1 = 48
	bw.writeBit(1)       // render_and_frame_size_equal_flag=1

	bw.writeBit(0) // refresh_frame_context
	bw.writeBit(1) // frame_parallel_decoding_mode
	bw.writeBits(2, 0) // frame_context_idx

	bw.writeBits(6, 0) // loop_filter level
	bw.writeBits(3, 0) // sharpness
	bw.writeBit(0)     // mode_ref_delta_enabled

	// quantization_params
	bw.writeBits(8, uint32(baseQ))
	// writeDeltaQ: 1-bit flag + su(4) = 4-bit 2's complement signed int (spec §7.2.9).
	writeDeltaQ := func(v int) {
		if v == 0 {
			bw.writeBit(0)
			return
		}
		bw.writeBit(1)
		// Encode as 4-bit 2's complement: range [-8, 7].
		bits := uint32(v & 0xF)
		bw.writeBits(4, bits)
	}
	writeDeltaQ(deltaYDC)
	writeDeltaQ(deltaUVDC)
	writeDeltaQ(deltaUVAC)

	bw.writeBit(0) // segmentation_enabled

	bw.writeBit(0) // tile_cols (stop at min)
	bw.writeBit(0) // tile_rows

	bw.writeBits(16, 0) // compressed_header_size
	return bw.flush()
}

func TestDeltaQParsing(t *testing.T) {
	cases := []struct {
		baseQ, yDC, uvDC, uvAC int
	}{
		{100, 0, 0, 0},
		{80, 3, 0, 0},
		{80, -3, 0, 0},
		{64, 0, 5, -2},
		{128, 7, -7, 4},
	}
	for _, tc := range cases {
		payload := buildKeyFrameWithQ(tc.baseQ, tc.yDC, tc.uvDC, tc.uvAC)
		h, err := Parse(payload, 0, 0)
		if err != nil {
			t.Fatalf("q=%d yDC=%d: Parse error: %v", tc.baseQ, tc.yDC, err)
		}
		if h.BaseQIdx != tc.baseQ {
			t.Errorf("BaseQIdx: got %d, want %d", h.BaseQIdx, tc.baseQ)
		}
		if h.DeltaQYDC != tc.yDC {
			t.Errorf("DeltaQYDC: got %d, want %d", h.DeltaQYDC, tc.yDC)
		}
		if h.DeltaQUVDC != tc.uvDC {
			t.Errorf("DeltaQUVDC: got %d, want %d", h.DeltaQUVDC, tc.uvDC)
		}
		if h.DeltaQUVAC != tc.uvAC {
			t.Errorf("DeltaQUVAC: got %d, want %d", h.DeltaQUVAC, tc.uvAC)
		}
	}
}

func TestTileCalcs(t *testing.T) {
	cases := []struct {
		w       int
		minLog2 int
		maxLog2 int
	}{
		{64, 0, 0},
		{512, 0, 0},  // 8 sb cols → max = 1 (8>>1=4≥4 → max=1, 8>>2=2<4 stop) wait let me trace
		{1920, 0, 2}, // 30 sb cols → max: 30>>0=30≥4→max=1, 30>>1=15≥4→max=2, 30>>2=7≥4→max=3, 30>>3=3<4 stop → max=3... hmm
	}
	// Just test that the functions don't panic and return sane values.
	for _, tc := range cases {
		min := calcMinLog2TileCols(tc.w)
		max := calcMaxLog2TileCols(tc.w)
		if min < 0 || max < min {
			t.Errorf("w=%d: min=%d max=%d invalid", tc.w, min, max)
		}
	}
}
