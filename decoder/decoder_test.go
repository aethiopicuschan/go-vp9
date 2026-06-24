package decoder

import (
	"testing"

	bm "github.com/mgvs/go-vp9/blockmode"
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/ctx"
	"github.com/mgvs/go-vp9/mv"
	"github.com/mgvs/go-vp9/mvpred"
)

// ---- New / init -------------------------------------------------------------

func TestNewDecoder(t *testing.T) {
	d := New()
	if d == nil {
		t.Fatal("New() returned nil")
	}
	if d.probs == nil {
		t.Fatal("probs not initialised")
	}
}

// ---- qSteps -----------------------------------------------------------------

func TestQStepsZero(t *testing.T) {
	dc, ac := qStepsForPlane(0, 0, 0)
	// Q=0 → lossless, steps = 4,4 (spec Table A-6).
	if dc != 4 || ac != 4 {
		t.Fatalf("qStepsForPlane(0,0,0): dc=%d ac=%d, want 4,4", dc, ac)
	}
}

func TestQStepsMid(t *testing.T) {
	dc, ac := qStepsForPlane(10, 0, 0)
	if dc <= 0 || ac <= 0 {
		t.Fatalf("qStepsForPlane(10,0,0): dc=%d ac=%d, want >0", dc, ac)
	}
}

func TestQStepsWithDelta(t *testing.T) {
	// Positive delta shifts to higher Q.
	dc0, _ := qStepsForPlane(10, 0, 0)
	dc1, _ := qStepsForPlane(10, 2, 0)
	if dc1 <= dc0 {
		t.Fatalf("positive delta should increase DC step: dc0=%d dc1=%d", dc0, dc1)
	}
	// Negative delta shifts to lower Q (clamped to 0).
	dc2, _ := qStepsForPlane(2, -5, 0)
	dc3, _ := qStepsForPlane(0, 0, 0)
	if dc2 != dc3 {
		t.Fatalf("clamped negative delta: dc2=%d, want dc3=%d", dc2, dc3)
	}
	// Clamped at 255.
	dc4, _ := qStepsForPlane(250, 10, 0)
	dc5, _ := qStepsForPlane(255, 0, 0)
	if dc4 != dc5 {
		t.Fatalf("clamped positive delta: dc4=%d, want dc5=%d", dc4, dc5)
	}
}

// ---- resolveMVs – ZERO mode -------------------------------------------------

func TestResolveMVsZeroMV(t *testing.T) {
	d := &Decoder{miW4: 16, miH4: 16}
	d.miGrid = make([]miBlock, 16*16)

	bmo := bm.BlockMode{
		IsInter:   true,
		InterMode: 2, // ZeroMV
		RefFrame:  [2]int{ctx.RefLast, ctx.RefIntra},
	}
	fi := bm.FrameInfo{
		IsKeyframe:   false,
		ReferenceMode: 0,
		CompFixedRef:  ctx.RefAltRef,
		CompVarRef:    [2]int{ctx.RefLast, ctx.RefGolden},
	}
	row := ctx.NewRow(64)
	col := ctx.NewCol()

	mv0, _ := d.resolveMVs(bmo, fi, row, col, 4, 4, 2, 2)
	if mv0 != (mv.MV{}) {
		t.Fatalf("ZERO mode mv0: got %v, want {0,0}", mv0)
	}
}

// ---- resolveMVs – NEAREST from above neighbour ------------------------------

func TestResolveMVsNearestFromAbove(t *testing.T) {
	d := &Decoder{miW4: 16, miH4: 16}
	d.miGrid = make([]miBlock, 16*16)
	// Set above block at (3, 4) to have LAST ref and a non-zero MV.
	wantMV := mv.MV{Row: 8, Col: -4}
	d.miGrid[3*16+4] = miBlock{
		inter:  true,
		ref:    [2]int{ctx.RefLast, ctx.RefIntra},
		mv:     [2]mv.MV{wantMV, {}},
		zeroMV: false,
	}

	bmo := bm.BlockMode{
		IsInter:   true,
		InterMode: 0, // NearestMV
		RefFrame:  [2]int{ctx.RefLast, ctx.RefIntra},
	}
	fi := bm.FrameInfo{
		IsKeyframe:   false,
		ReferenceMode: 0,
		CompFixedRef:  ctx.RefAltRef,
		CompVarRef:    [2]int{ctx.RefLast, ctx.RefGolden},
	}
	row := ctx.NewRow(64)
	col := ctx.NewCol()

	mv0, _ := d.resolveMVs(bmo, fi, row, col, 4, 4, 2, 2)
	if mv0 != wantMV {
		t.Fatalf("NEAREST from above: got %v, want %v", mv0, wantMV)
	}
}

// ---- setMI / getNeighbourMV -------------------------------------------------

func TestSetMIAndGet(t *testing.T) {
	d := &Decoder{miW4: 8, miH4: 8}
	d.miGrid = make([]miBlock, 8*8)

	bmo := bm.BlockMode{
		IsInter:  true,
		RefFrame: [2]int{ctx.RefLast, ctx.RefIntra},
	}
	mv0 := mv.MV{Row: 16, Col: 0}
	d.setMI(2, 2, 2, 2, bmo, mv0, mv.MV{}, 0)

	// All 4 MI positions in the 2×2 block should reflect mv0.
	for r := 2; r < 4; r++ {
		for c := 2; c < 4; c++ {
			got := d.getNeighbourMV(r, c, ctx.RefLast)
			if got.MV != mv0 {
				t.Fatalf("[%d][%d] MV: got %v, want %v", r, c, got.MV, mv0)
			}
			if got.Ref != ctx.RefLast {
				t.Fatalf("[%d][%d] Ref: got %d, want RefLast=%d", r, c, got.Ref, ctx.RefLast)
			}
		}
	}
}

func TestGetNeighbourMVOutOfBounds(t *testing.T) {
	d := &Decoder{miW4: 8, miH4: 8}
	d.miGrid = make([]miBlock, 64)

	got := d.getNeighbourMV(-1, 0, ctx.RefLast)
	if got.Ref != ctx.RefIntra {
		t.Fatalf("above OOB: got Ref=%d, want RefIntra=%d", got.Ref, ctx.RefIntra)
	}
	got = d.getNeighbourMV(0, -1, ctx.RefLast)
	if got.Ref != ctx.RefIntra {
		t.Fatalf("left OOB: got Ref=%d, want RefIntra=%d", got.Ref, ctx.RefIntra)
	}
	got = d.getNeighbourMV(0, 8, ctx.RefLast) // col beyond width
	if got.Ref != ctx.RefIntra {
		t.Fatalf("right OOB: got Ref=%d, want RefIntra=%d", got.Ref, ctx.RefIntra)
	}
}

func TestGetNeighbourMVIntraBlock(t *testing.T) {
	d := &Decoder{miW4: 4, miH4: 4}
	d.miGrid = make([]miBlock, 16)
	// Intra block (inter=false) at (1,1).
	d.miGrid[1*4+1] = miBlock{inter: false, ref: [2]int{0, 0}}

	got := d.getNeighbourMV(1, 1, ctx.RefLast)
	if got.Ref != ctx.RefIntra {
		t.Fatalf("intra neighbour: got Ref=%d, want RefIntra=0", got.Ref)
	}
}

// ---- mvpred integration -----------------------------------------------------

func TestFindBestRefMVsNoNeighbours(t *testing.T) {
	noBlock := mvpred.BlockMV{Ref: ctx.RefIntra, MV: mv.MV{}}
	nearest, near := mvpred.FindBestRefMVs(noBlock, noBlock, noBlock, noBlock, mvpred.BlockMV{}, ctx.RefLast)
	if nearest != (mv.MV{}) || near != (mv.MV{}) {
		t.Fatalf("no neighbours: nearest=%v near=%v, want {0,0}", nearest, near)
	}
}

// ---- boolcoder roundtrip sanity ---------------------------------------------

func TestBoolcoderRoundtrip(t *testing.T) {
	w := boolcoder.NewWriter()
	w.WriteBool(128, 1)
	w.WriteBool(128, 0)
	w.WriteBool(128, 1)
	data := w.Bytes()
	br := boolcoder.NewReader(data)
	if br.ReadBool(128) != 1 {
		t.Fatal("bit 0: want 1")
	}
	if br.ReadBool(128) != 0 {
		t.Fatal("bit 1: want 0")
	}
	if br.ReadBool(128) != 1 {
		t.Fatal("bit 2: want 1")
	}
}
