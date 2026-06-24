package loopfilter

import "testing"

// ---- Params tests -----------------------------------------------------------

func TestParamsLevel0(t *testing.T) {
	p := ParamsForLevel(0, 0)
	if p.BLimit == 0 {
		t.Fatal("BLimit must be > 0 even at level 0 (limit >= 1)")
	}
}

func TestParamsLevel63(t *testing.T) {
	p := ParamsForLevel(63, 0)
	if p.BLimit < p.Limit {
		t.Fatalf("BLimit %d < Limit %d at level 63", p.BLimit, p.Limit)
	}
}

func TestParamsClamp(t *testing.T) {
	p0 := ParamsForLevel(-1, 0)
	p1 := ParamsForLevel(0, 0)
	if p0 != p1 {
		t.Fatal("negative level should clamp to 0")
	}
	p254 := ParamsForLevel(300, 0)
	p255 := ParamsForLevel(63, 0)
	if p254 != p255 {
		t.Fatal("level > 63 should clamp to 63")
	}
}

// ---- Mask tests -------------------------------------------------------------

// A flat uniform block passes the mask (returns true) — the filter has no effect
// but is technically invoked. VP9 only SKIPS filtering when differences exceed limits.
func TestNeedsFilterUniform(t *testing.T) {
	buf := make([]byte, 8)
	for i := range buf {
		buf[i] = 128
	}
	p := ParamsForLevel(20, 0)
	if !needsFilter(buf, 4, 1, p.BLimit, p.Limit) {
		t.Fatal("uniform block passes the mask — needsFilter should return true")
	}
}

// A small step (within blimit) across an edge should also trigger the mask.
func TestNeedsFilterSmallStep(t *testing.T) {
	buf := make([]byte, 8)
	for i := 0; i < 4; i++ {
		buf[i] = 100
		buf[4+i] = 120 // diff = 20; *2 = 40 ≤ BLimit at level=30
	}
	p := ParamsForLevel(30, 0) // BLimit = 2*30 + lim30 = 60+3 = 63
	if !needsFilter(buf, 4, 1, p.BLimit, p.Limit) {
		t.Fatal("small step should pass the filter mask")
	}
}

// A step too large for any blimit should NOT trigger the filter.
func TestNeedsFilterHugeStep(t *testing.T) {
	buf := make([]byte, 8)
	for i := 0; i < 4; i++ {
		buf[i] = 10
		buf[4+i] = 200 // diff = 190; *2 = 380 > max BLimit (142)
	}
	p := ParamsForLevel(63, 0) // tightest useful BLimit = 2*63+16 = 142
	if needsFilter(buf, 4, 1, p.BLimit, p.Limit) {
		t.Fatal("huge discontinuity exceeds blimit — needsFilter should return false")
	}
}

// ---- filter4 tests ----------------------------------------------------------

// A perfectly flat edge (equal pixels on both sides) should be unchanged
// even when the filter is applied.
func TestFilter4FlatEdge(t *testing.T) {
	buf := make([]byte, 8)
	for i := range buf {
		buf[i] = 128
	}
	applyFilter4(buf, 4, 1, false)
	for i, v := range buf {
		if v != 128 {
			t.Fatalf("filter4 flat: buf[%d] = %d, want 128", i, v)
		}
	}
}

// A small but non-zero edge should be reduced (not increase) the discontinuity.
func TestFilter4ReducesStep(t *testing.T) {
	buf := []byte{100, 100, 100, 100, 140, 140, 140, 140}
	before := int(buf[4]) - int(buf[3]) // = 40
	applyFilter4(buf, 4, 1, false)
	after := int(buf[4]) - int(buf[3])
	if after < 0 {
		after = -after
	}
	if before < 0 {
		before = -before
	}
	if after > before {
		t.Fatalf("filter4 increased the edge discontinuity: before=%d after=%d", before, after)
	}
}

// ---- filter8 tests ----------------------------------------------------------

// Uniform region stays uniform after filter8.
func TestFilter8Uniform(t *testing.T) {
	buf := make([]byte, 8)
	for i := range buf {
		buf[i] = 128
	}
	applyFilter8(buf, 4, 1)
	for i, v := range buf {
		if v != 128 {
			t.Fatalf("filter8 uniform: buf[%d] = %d, want 128", i, v)
		}
	}
}

// ---- FilterEdge4 integration ------------------------------------------------

// With level=0, FilterEdge4 should skip filtering (BLimit should stop the mask).
func TestFilterEdge4Level0Uniform(t *testing.T) {
	buf := make([]byte, 8)
	for i := range buf {
		buf[i] = 100
	}
	p := ParamsForLevel(0, 0)
	FilterEdge4(buf, 4, 1, 8, p)
	for i, v := range buf {
		if v != 100 {
			t.Fatalf("level0 uniform: buf[%d] = %d, want 100", i, v)
		}
	}
}

// ---- FilterPlane smoke test -------------------------------------------------

// A uniform plane must be unchanged after FilterPlane.
func TestFilterPlaneUniform(t *testing.T) {
	const W, H = 16, 16
	buf := make([]byte, W*H)
	for i := range buf {
		buf[i] = 128
	}
	p := ParamsForLevel(30, 0)
	FilterPlane(buf, W, W, H, 2, p)
	for i, v := range buf {
		if v != 128 {
			t.Fatalf("FilterPlane uniform: buf[%d] = %d, want 128", i, v)
		}
	}
}

// A plane with a small block-boundary discontinuity (within blimit) must be
// softened. VP9 only filters edges within the blimit range.
func TestFilterPlaneSoftensEdge(t *testing.T) {
	const W, H = 16, 16
	buf := make([]byte, W*H)
	// diff = 20; *2 = 40, within BLimit at level=50 (≈ 110).
	for r := 0; r < H; r++ {
		v := byte(110)
		if r >= H/2 {
			v = 130 // diff = 20
		}
		for c := 0; c < W; c++ {
			buf[r*W+c] = v
		}
	}
	before := int(buf[8*W]) - int(buf[7*W]) // = 20
	p := ParamsForLevel(50, 0)
	FilterPlane(buf, W, W, H, 2, p)
	after := int(buf[8*W]) - int(buf[7*W])
	if after < 0 {
		after = -after
	}
	if before < 0 {
		before = -before
	}
	if after >= before {
		t.Fatalf("FilterPlane did not soften edge: before=%d after=%d", before, after)
	}
}
