package mvpred

import (
	"testing"

	"github.com/aethiopicuschan/go-vp9/mv"
)

func bm(row, col int16, ref int) BlockMV {
	return BlockMV{MV: mv.MV{Row: row, Col: col}, Ref: ref}
}

// ---- List.Add / BestMVs tests -----------------------------------------------

func TestListEmpty(t *testing.T) {
	var l List
	n, nr := l.BestMVs()
	if n != zeroMV || nr != zeroMV {
		t.Fatalf("empty list: got {%v,%v}, want zeros", n, nr)
	}
}

func TestListOneCandidate(t *testing.T) {
	var l List
	l.Add(bm(8, 16, RefLast), RefLast)
	nearest, near := l.BestMVs()
	if nearest.Row != 8 || nearest.Col != 16 {
		t.Fatalf("nearest: got %v, want {8,16}", nearest)
	}
	if near != zeroMV {
		t.Fatalf("near should be zero, got %v", near)
	}
}

func TestListTwoCandidates(t *testing.T) {
	var l List
	l.Add(bm(8, 16, RefLast), RefLast)
	l.Add(bm(4, 4, RefLast), RefLast)
	nearest, near := l.BestMVs()
	// Both have count=1; first added wins (insertion sort is stable).
	if nearest.Row != 8 || nearest.Col != 16 {
		t.Fatalf("nearest: got %v, want {8,16}", nearest)
	}
	if near.Row != 4 || near.Col != 4 {
		t.Fatalf("near: got %v, want {4,4}", near)
	}
}

func TestListDuplicateCountsHigher(t *testing.T) {
	var l List
	// Same MV from above and aboveRight → count=2; different left → count=1.
	l.Add(bm(8, 0, RefLast), RefLast)  // count=1
	l.Add(bm(8, 0, RefLast), RefLast)  // count=2 (dedup)
	l.Add(bm(4, 0, RefLast), RefLast)  // count=1
	nearest, near := l.BestMVs()
	if nearest.Row != 8 {
		t.Fatalf("most-frequent MV should be nearest, got row=%d", nearest.Row)
	}
	if near.Row != 4 {
		t.Fatalf("near should be the unique MV, got row=%d", near.Row)
	}
}

func TestListWrongRef(t *testing.T) {
	var l List
	l.Add(bm(8, 0, RefGolden), RefLast)  // different ref — skip
	l.Add(bm(4, 0, RefLast), RefLast)
	nearest, near := l.BestMVs()
	if nearest.Row != 4 {
		t.Fatalf("only matching ref should be nearest, got row=%d", nearest.Row)
	}
	if near != zeroMV {
		t.Fatalf("near should be zero, got %v", near)
	}
}

func TestListIntraRef(t *testing.T) {
	var l List
	l.Add(bm(0, 0, RefIntra), RefLast)  // intra — always skipped
	nearest, near := l.BestMVs()
	if nearest != zeroMV || near != zeroMV {
		t.Fatalf("intra neighbour should be ignored")
	}
}

// ---- FindBestRefMVs ---------------------------------------------------------

func TestFindBestRefMVsAllSame(t *testing.T) {
	mv8 := bm(8, 0, RefLast)
	nearest, near := FindBestRefMVs(mv8, mv8, mv8, mv8, BlockMV{}, RefLast)
	if nearest.Row != 8 {
		t.Fatalf("nearest: got %v, want row=8", nearest)
	}
	// Only 1 unique MV, so near = zero.
	if near != zeroMV {
		t.Fatalf("near should be zero (all same), got %v", near)
	}
}

func TestFindBestRefMVsNoNeighbours(t *testing.T) {
	empty := BlockMV{}
	n, nr := FindBestRefMVs(empty, empty, empty, empty, BlockMV{}, RefLast)
	if n != zeroMV || nr != zeroMV {
		t.Fatalf("no neighbours: got {%v,%v}, want zeros", n, nr)
	}
}

// ---- ClampMV ----------------------------------------------------------------

func TestClampMVInside(t *testing.T) {
	m := mv.MV{Row: 10, Col: -10}
	got := ClampMV(m, -100, 100, -100, 100)
	if got != m {
		t.Fatalf("inside bounds: got %v, want %v", got, m)
	}
}

func TestClampMVRowLow(t *testing.T) {
	m := mv.MV{Row: -200, Col: 0}
	got := ClampMV(m, -100, 100, -100, 100)
	if got.Row != -100 {
		t.Fatalf("row clamped low: got %d, want -100", got.Row)
	}
}

func TestClampMVColHigh(t *testing.T) {
	m := mv.MV{Row: 0, Col: 500}
	got := ClampMV(m, -100, 100, -100, 100)
	if got.Col != 100 {
		t.Fatalf("col clamped high: got %d, want 100", got.Col)
	}
}
