package decoder

import (
	"testing"

	"github.com/aethiopicuschan/go-vp9/blockmode"
	"github.com/aethiopicuschan/go-vp9/ctx"
	"github.com/aethiopicuschan/go-vp9/mv"
)

func TestAppendSub8x8BlockZeroWithoutMVRefs(t *testing.T) {
	d := &Decoder{
		miW4:             16,
		miH4:             16,
		curTileColStart4: 0,
		curTileColEnd4:   16,
		miGrid:           make([]miBlock, 16*16),
	}

	_, _, count := d.findMVRefs(0, 0, 2, 2, ctx.RefLast, 0, false, false, d.signBias, d.allowHP)
	if count != 0 {
		t.Fatalf("findMVRefs at frame origin found %d candidates, want 0", count)
	}

	for _, mode := range []int{blockmode.NearestMV, blockmode.NearMV} {
		if got := d.appendSub8x8(ctx.RefLast, 0, 0, 0, 2, 2, 0, mode, [4][2]mv.MV{}); got != (mv.MV{}) {
			t.Errorf("appendSub8x8 block 0 mode %d with no candidates = %v, want ZeroMv", mode, got)
		}
	}
}

func TestAppendSub8x8BlockZeroNearUsesSecondCandidateSlot(t *testing.T) {
	d := &Decoder{
		miW4:             16,
		miH4:             16,
		curTileColStart4: 0,
		curTileColEnd4:   16,
		miGrid:           make([]miBlock, 16*16),
	}
	wantNearest := mv.MV{Row: 8, Col: -4}
	d.miGrid[2] = miBlock{
		inter: true,
		ref:   [2]int{ctx.RefLast, ctx.RefIntra},
		mv:    [2]mv.MV{wantNearest, {}},
	}

	if got := d.appendSub8x8(ctx.RefLast, 0, 2, 2, 2, 2, 0, blockmode.NearestMV, [4][2]mv.MV{}); got != wantNearest {
		t.Fatalf("NEARESTMV = %v, want %v", got, wantNearest)
	}
	if got := d.appendSub8x8(ctx.RefLast, 0, 2, 2, 2, 2, 0, blockmode.NearMV, [4][2]mv.MV{}); got != (mv.MV{}) {
		t.Fatalf("NEARMV with only one candidate = %v, want ZeroMv fallback", got)
	}
}
