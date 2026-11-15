package partition

import (
	"testing"

	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/compheader"
)

// encodePartition writes a partition type onto the bool writer using the
// inverse of the partition_tree (for roundtrip testing).
func encodePartition(w *boolcoder.Writer, ctx int, probs [16][3]uint8, pt Type) {
	p := probs[ctx]
	switch pt {
	case None:
		w.WriteBool(p[0], 0)
	case Horz:
		w.WriteBool(p[0], 1)
		w.WriteBool(p[1], 0)
	case Vert:
		w.WriteBool(p[0], 1)
		w.WriteBool(p[1], 1)
		w.WriteBool(p[2], 0)
	case Split:
		w.WriteBool(p[0], 1)
		w.WriteBool(p[1], 1)
		w.WriteBool(p[2], 1)
	}
}

func TestPartitionRoundtrip(t *testing.T) {
	probs := compheader.DefaultProbs().PartitionProbs
	cases := []struct {
		ctx int
		pt  Type
	}{
		{0, None},
		{0, Horz},
		{0, Vert},
		{0, Split},
		{3, None},
		{3, Split},
		{12, None},  // 64x64 context
		{15, Split}, // 64x64 all-split context
	}
	for _, tc := range cases {
		w := boolcoder.NewWriter()
		encodePartition(w, tc.ctx, probs, tc.pt)
		br := boolcoder.NewReader(w.Bytes())
		got := Read(br, tc.ctx, probs, nil)
		if got != tc.pt {
			t.Errorf("ctx=%d pt=%d: roundtrip got %d", tc.ctx, tc.pt, got)
		}
	}
}

func TestContextValues(t *testing.T) {
	// ctx 0..15 must span the expected range
	cases := []struct{ above, left bool; bsl, want int }{
		{false, false, 1, 0},  // 8x8, no above/left → (0*2+0) + 0 = 0
		{true, false, 1, 1},   // 8x8, above → (0*2+1) + 0 = 1
		{false, true, 1, 2},   // 8x8, left  → (1*2+0) + 0 = 2
		{true, true, 1, 3},    // 8x8, both  → 3
		{false, false, 4, 12}, // 64x64, none → (0) + 12
		{true, true, 4, 15},   // 64x64, both → 3 + 12
	}
	for _, tc := range cases {
		got := Context(tc.above, tc.left, tc.bsl)
		if got != tc.want {
			t.Errorf("Context(above=%v, left=%v, bsl=%d) = %d, want %d",
				tc.above, tc.left, tc.bsl, got, tc.want)
		}
	}
}

func TestAllContextsAllTypes(t *testing.T) {
	probs := compheader.DefaultProbs().PartitionProbs
	for ctx := 0; ctx < 16; ctx++ {
		for _, pt := range []Type{None, Horz, Vert, Split} {
			w := boolcoder.NewWriter()
			encodePartition(w, ctx, probs, pt)
			br := boolcoder.NewReader(w.Bytes())
			got := Read(br, ctx, probs, nil)
			if got != pt {
				t.Errorf("ctx=%d pt=%d: roundtrip got %d", ctx, pt, got)
			}
		}
	}
}
