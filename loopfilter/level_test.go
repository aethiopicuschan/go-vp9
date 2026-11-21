package loopfilter

import "testing"

// libvpx vp9_loop_filter_frame_init scales the ref/mode deltas by
// 1 << (default_filt_lvl >> 5).
func TestLevelForBlockDeltaScale(t *testing.T) {
	ref := [4]int{1, 0, -1, -1}
	mode := [2]int{0, 2}
	cases := []struct {
		base, refFrame int
		inter, zeroMV  bool
		want           int
	}{
		{31, 0, false, false, 32}, // scale 1
		{31, 2, true, true, 30},
		{31, 2, true, false, 32},
		{32, 0, false, false, 34}, // scale 2
		{40, 2, true, true, 38},
		{40, 2, true, false, 42},
		{63, 0, false, false, 63}, // clamped
		{0, 0, false, false, 1},   // deltas apply at level 0 too
	}
	for _, c := range cases {
		got := LevelForBlock(c.base, false, 0, false, true, c.refFrame, c.inter, c.zeroMV, ref, mode)
		if got != c.want {
			t.Errorf("base %d ref %d inter %v zero %v: got %d, want %d", c.base, c.refFrame, c.inter, c.zeroMV, got, c.want)
		}
	}
	if got := LevelForBlock(40, false, 0, false, false, 0, false, false, ref, mode); got != 40 {
		t.Errorf("deltas disabled: got %d, want 40", got)
	}
}

// Segment ALT_LF (libvpx vp9_loop_filter_frame_init): delta or absolute level,
// clamped to 0..63 before the ref/mode deltas, which still apply when the
// segment level is 0 and are scaled by the frame level, not the segment's.
func TestLevelForBlockSegment(t *testing.T) {
	ref := [4]int{1, 0, -1, -1}
	mode := [2]int{0, 0}
	cases := []struct {
		base     int
		data     int16
		abs      bool
		deltas   bool
		refFrame int
		inter    bool
		want     int
	}{
		{20, -5, false, false, 0, false, 15},
		{20, -5, false, true, 0, false, 16},
		{20, -63, false, true, 0, false, 1}, // segment level 0, intra delta +1
		{20, -63, false, true, 1, true, 0},
		{20, -63, false, false, 0, false, 0},
		{20, 7, true, false, 1, true, 7}, // absolute level
		{20, 7, true, true, 2, true, 6},
		{40, 7, true, true, 2, true, 5},     // scale from the frame level (40 >> 5)
		{40, -63, false, true, 0, false, 2}, // 0 + 1*2
		{60, 10, false, true, 1, true, 63},
	}
	for _, c := range cases {
		got := LevelForBlock(c.base, true, c.data, c.abs, c.deltas, c.refFrame, c.inter, false, ref, mode)
		if got != c.want {
			t.Errorf("base %d data %d abs %v deltas %v ref %d: got %d, want %d",
				c.base, c.data, c.abs, c.deltas, c.refFrame, got, c.want)
		}
	}
}
