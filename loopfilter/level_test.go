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
		{0, 0, false, false, 0},   // level 0 stays off
	}
	for _, c := range cases {
		got := LevelForBlock(c.base, false, 0, true, c.refFrame, c.inter, c.zeroMV, ref, mode)
		if got != c.want {
			t.Errorf("base %d ref %d inter %v zero %v: got %d, want %d", c.base, c.refFrame, c.inter, c.zeroMV, got, c.want)
		}
	}
	if got := LevelForBlock(40, false, 0, false, 0, false, false, ref, mode); got != 40 {
		t.Errorf("deltas disabled: got %d, want 40", got)
	}
}
