package blockmode

import (
	"testing"

	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/compheader"
	"github.com/aethiopicuschan/go-vp9/ctx"
	"github.com/aethiopicuschan/go-vp9/modeinfo"
)

// encodeIntraBlock writes a minimal intra block bit sequence with prob=128:
// skip(1 bit) + Y-mode tree (up to 9 bits) + UV-mode tree.
// Returns bytes and the expected skip/yMode/uvMode.
func encodeIntraBlock(skip bool, yMode, uvMode modeinfo.IntraMode) []byte {
	w := boolcoder.NewWriter()
	if skip {
		w.WriteBool(128, 1)
	} else {
		w.WriteBool(128, 0)
	}
	// Y mode tree with all probs=128: write the path through intraModeTree.
	// intraModeTree: root→ -DC(0)/2; node2→ -TM(9)/4; node4→ -V(1)/6;
	//               node6→ {8,12}; node8→ -H(2)/10; node10→ -D135/−D117;
	//               node12→ -D45/14; node14→ -D63/16; node16→ -D153/−D207
	writeMode := func(mode modeinfo.IntraMode) {
		switch mode {
		case modeinfo.DC: // 0→leaf
			w.WriteBool(128, 0)
		case modeinfo.TM: // 1→0→leaf
			w.WriteBool(128, 1)
			w.WriteBool(128, 0)
		case modeinfo.V: // 1→1→0→leaf
			w.WriteBool(128, 1)
			w.WriteBool(128, 1)
			w.WriteBool(128, 0)
		case modeinfo.H: // 1→1→1→0→0→leaf
			w.WriteBool(128, 1)
			w.WriteBool(128, 1)
			w.WriteBool(128, 1)
			w.WriteBool(128, 0)
			w.WriteBool(128, 0)
		case modeinfo.D135: // 1→1→1→0→1→0→leaf
			w.WriteBool(128, 1)
			w.WriteBool(128, 1)
			w.WriteBool(128, 1)
			w.WriteBool(128, 0)
			w.WriteBool(128, 1)
			w.WriteBool(128, 0)
		case modeinfo.D45: // 1→1→1→1→0→leaf
			w.WriteBool(128, 1)
			w.WriteBool(128, 1)
			w.WriteBool(128, 1)
			w.WriteBool(128, 1)
			w.WriteBool(128, 0)
		}
	}
	writeMode(yMode)
	writeMode(uvMode)
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
	probs := compheader.NewProbs()
	probs.TxMode = compheader.TxModeOnly4x4
	fi := newKeyframeInfo()

	data := encodeIntraBlock(false, modeinfo.V, modeinfo.H)
	br := boolcoder.NewReader(data)
	row := ctx.NewRow(64)
	col := ctx.NewCol(16)

	bm := Decode(br, probs, nil, fi, row, col, 0, 0, 2, 2, Sub8x8None, nil, nil)
	if bm.YMode != modeinfo.V {
		t.Fatalf("YMode: got %v, want V", bm.YMode)
	}
	if bm.UVMode != modeinfo.H {
		t.Fatalf("UVMode: got %v, want H", bm.UVMode)
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
