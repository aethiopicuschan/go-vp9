package blockmode

import (
	"testing"

	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/compheader"
	"github.com/aethiopicuschan/go-vp9/ctx"
	"github.com/aethiopicuschan/go-vp9/modeinfo"
)

// encodeIntraBlock writes a minimal key-frame intra block bit sequence:
// skip (prob 128, the NewProbs value) + Y-mode tree + UV-mode tree.
//
// Key (and intra-only) frames do not read the modes with the frame context:
// Decode uses the fixed tables KFYModeProbs[above][left] (DC/DC for a block
// with no neighbours) and KFUVModeProbs[yMode] (libvpx
// read_intra_frame_mode_info), so the tree bits must be written with exactly
// those probabilities — not with 128.
func encodeIntraBlock(skip bool, yMode, uvMode modeinfo.IntraMode) []byte {
	w := boolcoder.NewWriter()
	if skip {
		w.WriteBool(128, 1)
	} else {
		w.WriteBool(128, 0)
	}
	// Paths through intraModeTree as {probability index (= node/2), bit}:
	// root→ -DC(0)/2; node2→ -TM(9)/4; node4→ -V(1)/6; node6→ {8,12};
	// node8→ -H(2)/10; node10→ -D135/-D117; node12→ -D45/14;
	// node14→ -D63/16; node16→ -D153/-D207.
	paths := map[modeinfo.IntraMode][][2]int{
		modeinfo.DC:   {{0, 0}},
		modeinfo.TM:   {{0, 1}, {1, 0}},
		modeinfo.V:    {{0, 1}, {1, 1}, {2, 0}},
		modeinfo.H:    {{0, 1}, {1, 1}, {2, 1}, {3, 0}, {4, 0}},
		modeinfo.D135: {{0, 1}, {1, 1}, {2, 1}, {3, 0}, {4, 1}, {5, 0}},
		modeinfo.D45:  {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {6, 0}},
	}
	writeMode := func(mode modeinfo.IntraMode, probs [9]uint8) {
		for _, pb := range paths[mode] {
			w.WriteBool(probs[pb[0]], pb[1])
		}
	}
	writeMode(yMode, modeinfo.KFYModeProbs[modeinfo.DC][modeinfo.DC])
	writeMode(uvMode, modeinfo.KFUVModeProbs[yMode])
	return w.Bytes()
}

func newKeyframeInfo() FrameInfo {
	return FrameInfo{
		IsKeyframe:    true,
		ReferenceMode: compheader.RefModeSingle,
		InterpFilter:  compheader.InterpEightTap,
	}
}

// ---- IntraBlock skip --------------------------------------------------------

func TestDecodeSkipTrue(t *testing.T) {
	probs := compheader.NewProbs()
	probs.TxMode = compheader.TxModeOnly4x4 // no TX select
	fi := newKeyframeInfo()

	data := encodeIntraBlock(true, modeinfo.DC, modeinfo.DC)
	br := boolcoder.NewReader(data)
	row := ctx.NewRow(64)
	col := ctx.NewCol(16)

	bm := Decode(br, probs, nil, fi, row, col, 4, 4, 2, 2, Sub8x8None, nil, nil) // 8×8 block
	if !bm.Skip {
		t.Fatal("expected skip=true")
	}
}

func TestDecodeSkipFalse(t *testing.T) {
	probs := compheader.NewProbs()
	probs.TxMode = compheader.TxModeOnly4x4
	fi := newKeyframeInfo()

	data := encodeIntraBlock(false, modeinfo.DC, modeinfo.DC)
	br := boolcoder.NewReader(data)
	row := ctx.NewRow(64)
	col := ctx.NewCol(16)

	bm := Decode(br, probs, nil, fi, row, col, 4, 4, 2, 2, Sub8x8None, nil, nil)
	if bm.Skip {
		t.Fatal("expected skip=false")
	}
}

// ---- Intra Y mode -----------------------------------------------------------

func TestDecodeYModeDC(t *testing.T) {
	probs := compheader.NewProbs()
	probs.TxMode = compheader.TxModeOnly4x4
	fi := newKeyframeInfo()

	data := encodeIntraBlock(false, modeinfo.DC, modeinfo.DC)
	br := boolcoder.NewReader(data)
	row := ctx.NewRow(64)
	col := ctx.NewCol(16)

	bm := Decode(br, probs, nil, fi, row, col, 0, 0, 2, 2, Sub8x8None, nil, nil)
	if bm.YMode != modeinfo.DC {
		t.Fatalf("YMode: got %v, want DC", bm.YMode)
	}
	if bm.UVMode != modeinfo.DC {
		t.Fatalf("UVMode: got %v, want DC", bm.UVMode)
	}
}

func TestDecodeYModeV(t *testing.T) {
	// Non-DC modes on both planes: fails unless writer and Decode agree on the
	// key-frame probability tables for every tree node on the path.
	cases := []struct{ y, uv modeinfo.IntraMode }{
		{modeinfo.V, modeinfo.H},
		{modeinfo.TM, modeinfo.D45},
		{modeinfo.H, modeinfo.D135},
		{modeinfo.D135, modeinfo.V},
		{modeinfo.D45, modeinfo.TM},
	}
	for _, c := range cases {
		probs := compheader.NewProbs()
		probs.TxMode = compheader.TxModeOnly4x4
		fi := newKeyframeInfo()

		data := encodeIntraBlock(false, c.y, c.uv)
		br := boolcoder.NewReader(data)
		row := ctx.NewRow(64)
		col := ctx.NewCol(16)

		bm := Decode(br, probs, nil, fi, row, col, 0, 0, 2, 2, Sub8x8None, nil, nil)
		if bm.YMode != c.y {
			t.Fatalf("YMode: got %v, want %v", bm.YMode, c.y)
		}
		if bm.UVMode != c.uv {
			t.Fatalf("UVMode: got %v, want %v (y=%v)", bm.UVMode, c.uv, c.y)
		}
	}
}

// ---- TX size select ---------------------------------------------------------

func TestDecodeTxSizeSelect8x8(t *testing.T) {
	probs := compheader.NewProbs() // all probs = 128
	probs.TxMode = compheader.TxModeSelect
	fi := newKeyframeInfo()

	// All decoded with prob=128; write with prob=128 to match.
	w := boolcoder.NewWriter()
	w.WriteBool(128, 0) // skip=false  (SkipProbs[0]=128)
	w.WriteBool(128, 1) // tx=1 → TX8x8 (TxProbs8[0]=128)
	w.WriteBool(128, 0) // yMode=DC
	w.WriteBool(128, 0) // uvMode=DC

	br := boolcoder.NewReader(w.Bytes())
	row := ctx.NewRow(64)
	col := ctx.NewCol(16)

	bm := Decode(br, probs, nil, fi, row, col, 0, 0, 2, 2, Sub8x8None, nil, nil) // bsLog2=1 → maxTx=TX8x8
	if bm.TxSize != ctx.TX8x8 {
		t.Fatalf("TxSize: got %d, want TX8x8=%d", bm.TxSize, ctx.TX8x8)
	}
}

func TestDecodeTxSizeSelect4x4(t *testing.T) {
	probs := compheader.NewProbs()
	probs.TxMode = compheader.TxModeSelect
	fi := newKeyframeInfo()

	w := boolcoder.NewWriter()
	w.WriteBool(128, 0) // skip=false (SkipProbs[0]=128)
	w.WriteBool(128, 0) // tx=0 → TX4x4 (TxProbs8[0]=128)
	w.WriteBool(128, 0) // yMode=DC
	w.WriteBool(128, 0) // uvMode=DC

	br := boolcoder.NewReader(w.Bytes())
	row := ctx.NewRow(64)
	col := ctx.NewCol(16)

	bm := Decode(br, probs, nil, fi, row, col, 0, 0, 2, 2, Sub8x8None, nil, nil)
	if bm.TxSize != ctx.TX4x4 {
		t.Fatalf("TxSize: got %d, want TX4x4=%d", bm.TxSize, ctx.TX4x4)
	}
}

// ---- IsInter = false for keyframe ------------------------------------------

func TestDecodeKeyframeAlwaysIntra(t *testing.T) {
	probs := compheader.NewProbs()
	probs.TxMode = compheader.TxModeOnly4x4
	fi := newKeyframeInfo()

	data := encodeIntraBlock(false, modeinfo.DC, modeinfo.DC)
	br := boolcoder.NewReader(data)
	row := ctx.NewRow(64)
	col := ctx.NewCol(16)

	bm := Decode(br, probs, nil, fi, row, col, 0, 0, 4, 4, Sub8x8None, nil, nil) // 16×16 block
	if bm.IsInter {
		t.Fatal("keyframe block should always be intra")
	}
	if bm.RefFrame != [2]int{ctx.RefIntra, ctx.RefIntra} {
		t.Fatalf("keyframe RefFrame: got %v, want intra", bm.RefFrame)
	}
}

// ---- CompFixedVarRef --------------------------------------------------------

func TestCompFixedVarRefLastAltBias(t *testing.T) {
	// LAST and ALTREF have same bias → GOLDEN is fixed.
	fixed, varRef := CompFixedVarRef([3]bool{true, false, true})
	if fixed != ctx.RefGolden {
		t.Fatalf("fixed: got %d, want RefGolden=%d", fixed, ctx.RefGolden)
	}
	if varRef[0] != ctx.RefLast || varRef[1] != ctx.RefAltRef {
		t.Fatalf("varRef: got %v", varRef)
	}
}

func TestCompFixedVarRefLastGoldenBias(t *testing.T) {
	// LAST and GOLDEN same bias → ALTREF is fixed.
	fixed, _ := CompFixedVarRef([3]bool{true, true, false})
	if fixed != ctx.RefAltRef {
		t.Fatalf("fixed: got %d, want RefAltRef=%d", fixed, ctx.RefAltRef)
	}
}

// ---- RefCtx no-neighbour baseline -------------------------------------------

func TestSingleRefCtxP1NoNeighbours(t *testing.T) {
	// No neighbours → context = 2 (from libvpx).
	got := SingleRefCtxP1(false, false,
		[2]int{ctx.RefIntra, ctx.RefIntra}, [2]int{ctx.RefIntra, ctx.RefIntra},
		false, false)
	if got != 2 {
		t.Fatalf("no neighbours: got %d, want 2", got)
	}
}

func TestCompModeCtxNoNeighbours(t *testing.T) {
	fi := FrameInfo{CompFixedRef: ctx.RefAltRef, CompVarRef: [2]int{ctx.RefLast, ctx.RefGolden}}
	got := CompModeCtx(fi, false, false,
		[2]int{ctx.RefIntra, ctx.RefIntra}, [2]int{ctx.RefIntra, ctx.RefIntra},
		false, false)
	// libvpx vp9_get_reference_mode_context: no edges available → ctx 1.
	if got != 1 {
		t.Fatalf("no neighbours: got %d, want 1", got)
	}
}
