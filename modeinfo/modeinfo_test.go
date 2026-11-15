package modeinfo

import (
	"testing"

	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/compheader"
)

// encodeIntraMode writes an IntraMode onto a bool writer by traversing the
// intraModeTree in reverse (for roundtrip testing).
func encodeIntraMode(w *boolcoder.Writer, probs []uint8, mode IntraMode) {
	type step struct {
		prob uint8
		bit  int
	}
	var path []step
	// nodeIdx is an index into intraModeTree (never a leaf value).
	var walk func(nodeIdx int) bool
	walk = func(nodeIdx int) bool {
		prob := probs[nodeIdx/2]
		for bit := 0; bit <= 1; bit++ {
			child := int(intraModeTree[nodeIdx+bit])
			if child <= 0 {
				if IntraMode(-child) == mode {
					path = append(path, step{prob, bit})
					return true
				}
			} else {
				if walk(child) {
					path = append(path, step{prob, bit})
					return true
				}
			}
		}
		return false
	}
	walk(0)
	// path is in reverse order (deepest first); write it in order
	for i := len(path) - 1; i >= 0; i-- {
		w.WriteBool(path[i].prob, path[i].bit)
	}
}

func TestYModeRoundtrip(t *testing.T) {
	probs := compheader.DefaultProbs()
	for sg := 0; sg < 4; sg++ {
		for m := 0; m < Count; m++ {
			mode := IntraMode(m)
			w := boolcoder.NewWriter()
			encodeIntraMode(w, probs.YModeProbs[sg][:], mode)
			br := boolcoder.NewReader(w.Bytes())
			got := ReadYMode(br, sg, probs.YModeProbs, nil)
			if got != mode {
				t.Errorf("sizeGroup=%d mode=%d: roundtrip got %d", sg, mode, got)
			}
		}
	}
}

func TestUVModeRoundtrip(t *testing.T) {
	probs := compheader.DefaultProbs()
	for yMode := 0; yMode < Count; yMode++ {
		for uvMode := 0; uvMode < Count; uvMode++ {
			w := boolcoder.NewWriter()
			encodeIntraMode(w, probs.UVModeProbs[yMode][:], IntraMode(uvMode))
			br := boolcoder.NewReader(w.Bytes())
			got := ReadUVMode(br, IntraMode(yMode), probs.UVModeProbs, nil)
			if got != IntraMode(uvMode) {
				t.Errorf("yMode=%d uvMode=%d: roundtrip got %d", yMode, uvMode, got)
			}
		}
	}
}

func TestBlockSizeGroup(t *testing.T) {
	cases := []struct{ w, h, want int }{
		{4, 4, 0},
		{4, 8, 1},
		{8, 8, 1},
		{8, 16, 2},
		{16, 16, 2},
		{16, 32, 3},
		{32, 32, 3},
		{64, 64, 3},
	}
	for _, tc := range cases {
		got := BlockSizeGroup(tc.w, tc.h)
		if got != tc.want {
			t.Errorf("BlockSizeGroup(%d,%d) = %d, want %d", tc.w, tc.h, got, tc.want)
		}
	}
}
