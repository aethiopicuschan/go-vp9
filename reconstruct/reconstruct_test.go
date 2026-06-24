package reconstruct

import (
	"testing"

	"github.com/mgvs/go-vp9/blockmode"
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
	"github.com/mgvs/go-vp9/ctx"
	"github.com/mgvs/go-vp9/modeinfo"
	"github.com/mgvs/go-vp9/mv"
	"github.com/mgvs/go-vp9/refbuf"
)

// ---- genScan ----------------------------------------------------------------

func TestGenScan4x4(t *testing.T) {
	// VP9 default 4×4 scan (from spec / libvpx).
	want := []int{0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15}
	got := genScan(4)
	if len(got) != len(want) {
		t.Fatalf("len: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("scan[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestGenScan8x8FirstFew(t *testing.T) {
	// First 10 entries of the VP9 8×8 default scan.
	want8 := []int{0, 8, 1, 2, 9, 16, 24, 17, 10, 3}
	got := genScan(8)
	for i, w := range want8 {
		if got[i] != w {
			t.Fatalf("8x8 scan[%d]: got %d, want %d", i, got[i], w)
		}
	}
}

func TestGenScanCoverAll(t *testing.T) {
	// Each raster position should appear exactly once.
	for _, n := range []int{4, 8, 16, 32} {
		scan := genScan(n)
		seen := make([]bool, n*n)
		for _, v := range scan {
			if v < 0 || v >= n*n {
				t.Fatalf("n=%d: out of range %d", n, v)
			}
			if seen[v] {
				t.Fatalf("n=%d: duplicate raster position %d", n, v)
			}
			seen[v] = true
		}
	}
}

// ---- TxTypeFor --------------------------------------------------------------

func TestTxTypeForIntraDC(t *testing.T) {
	tt := TxTypeFor(modeinfo.DC, ctx.TX4x4, false)
	if tt != 0 { // DCT_DCT
		t.Fatalf("DC intra: got %d, want DCT_DCT=0", tt)
	}
}

func TestTxTypeForIntraV(t *testing.T) {
	tt := TxTypeFor(modeinfo.V, ctx.TX4x4, false)
	if tt != 1 { // DCT_ADST
		t.Fatalf("V intra 4x4: got %d, want DCT_ADST=1", tt)
	}
}

func TestTxTypeForIntraH(t *testing.T) {
	tt := TxTypeFor(modeinfo.H, ctx.TX4x4, false)
	if tt != 2 { // ADST_DCT
		t.Fatalf("H intra 4x4: got %d, want ADST_DCT=2", tt)
	}
}

func TestTxTypeForIntra32x32ForceDCT(t *testing.T) {
	// 32×32 blocks always use DCT_DCT even for TM mode.
	tt := TxTypeFor(modeinfo.TM, ctx.TX32x32, false)
	if tt != 0 {
		t.Fatalf("TM intra 32x32: got %d, want DCT_DCT=0", tt)
	}
}

func TestTxTypeForInter(t *testing.T) {
	// Inter always DCT_DCT regardless of mode or TX size.
	tt := TxTypeFor(modeinfo.TM, ctx.TX4x4, true)
	if tt != 0 {
		t.Fatalf("inter: got %d, want DCT_DCT=0", tt)
	}
}

// ---- mvSubPelUV -------------------------------------------------------------

func TestMvSubPelUVPositive(t *testing.T) {
	m := mv.MV{Row: 8, Col: 4}
	c, r := mvSubPelUV(m)
	// col=(4+1)>>1=2, row=(8+1)>>1=4
	if c != 2 || r != 4 {
		t.Fatalf("got col=%d row=%d, want col=2 row=4", c, r)
	}
}

func TestMvSubPelUVNegative(t *testing.T) {
	m := mv.MV{Row: -8, Col: -4}
	c, r := mvSubPelUV(m)
	// col=-(4+1>>1)=-2, row=-(8+1>>1)=-4
	if c != -2 || r != -4 {
		t.Fatalf("got col=%d row=%d, want col=-2 row=-4", c, r)
	}
}

// ---- intra reconstruction (DC mode, 8×8 block, no residual) ----------------

func TestBlockIntraDCSkip(t *testing.T) {
	// Decode a skip intra DC block into an 8×8 frame; verify fill is 128
	// (DC from all-128 borders = 128).
	const w, h = 8, 8
	dst := refbuf.NewFrame(w, h)
	// Pre-fill borders at row-1 and col-1 with 128 (default; frame is zeroed
	// so we need to set the border row first — for row=0 we fall back to 128).

	probs := compheader.NewProbs()
	// No coefficient bits to read (skip=true).
	br := boolcoder.NewReader([]byte{})

	bm := blockmode.BlockMode{
		IsInter: false,
		Skip:    true,
		TxSize:  ctx.TX8x8,
		YMode:   modeinfo.DC,
		UVMode:  modeinfo.DC,
		RefFrame: [2]int{ctx.RefIntra, ctx.RefIntra},
	}

	row := ctx.NewRow(w)
	col := ctx.NewCol()

	Block(br, probs, nil, bm, false,
		mv.MV{}, mv.MV{},
		[3]int16{0, 0, 0}, [3]int16{0, 0, 0},
		dst, [3]*refbuf.Frame{},
		row, col,
		0, 0, 2, 2, // row4=0, col4=0, bsW4=2, bsH4=2 (8×8)
	)

	// DC prediction from all-128 borders should produce 128.
	base := dst.YBase()
	for r := 0; r < h; r++ {
		for c := 0; c < w; c++ {
			p := dst.Y[base+r*dst.YStride+c]
			if p != 128 {
				t.Fatalf("Y[%d][%d] = %d, want 128", r, c, p)
			}
		}
	}
}

// ---- uvTxSizeFor ------------------------------------------------------------

func TestUvTxSizeFor8x8(t *testing.T) {
	// Y TX8x8, 8×8 block (bsW4=2, bsH4=2): UV block 4×4 → max TX4x4
	got := uvTxSizeFor(ctx.TX8x8, 2, 2)
	if got != ctx.TX4x4 {
		t.Fatalf("got %d, want TX4x4=%d", got, ctx.TX4x4)
	}
}

func TestUvTxSizeFor16x16(t *testing.T) {
	// Y TX16x16, 16×16 block (bsW4=4, bsH4=4): UV block 8×8 → max TX8x8
	got := uvTxSizeFor(ctx.TX16x16, 4, 4)
	if got != ctx.TX8x8 {
		t.Fatalf("got %d, want TX8x8=%d", got, ctx.TX8x8)
	}
}

func TestUvTxSizeForSmallY(t *testing.T) {
	// Y TX4x4 in 16×16 block: UV inherits TX4x4 (min of TX4x4 and TX8x8).
	got := uvTxSizeFor(ctx.TX4x4, 4, 4)
	if got != ctx.TX4x4 {
		t.Fatalf("got %d, want TX4x4=%d", got, ctx.TX4x4)
	}
}

func TestUvTxSizeForNonSquare(t *testing.T) {
	// 64×32 block (bsW4=16, bsH4=8): UV block 32×16 → min=16→log2=2 → max TX16x16
	got := uvTxSizeFor(ctx.TX32x32, 16, 8)
	if got != ctx.TX16x16 {
		t.Fatalf("64x32: got %d, want TX16x16=%d", got, ctx.TX16x16)
	}
	// 16×8 block (bsW4=4, bsH4=2): UV block 8×4 → min=4→log2=1 → max TX8x8 → capped at TX4x4 (=1)?
	// Actually min(uvW4=2, uvH4=1)=1 → log2(1)=0 → maxUV=TX4x4
	got2 := uvTxSizeFor(ctx.TX8x8, 4, 2)
	if got2 != ctx.TX4x4 {
		t.Fatalf("16x8: got %d, want TX4x4=%d", got2, ctx.TX4x4)
	}
}
