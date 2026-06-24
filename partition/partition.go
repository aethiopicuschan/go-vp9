// Package partition decodes VP9 partition types for superblocks (spec §6.3.5).
// VP9 recursively splits 64×64 superblocks into smaller blocks using a binary
// tree with four possible outcomes: NONE (keep current size), HORZ (split
// into top/bottom halves), VERT (split into left/right halves), or SPLIT
// (split into four equal quadrants).
package partition

import (
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
)

// Type is a VP9 partition type.
type Type int

const (
	None  Type = 0 // PARTITION_NONE
	Horz  Type = 1 // PARTITION_HORZ
	Vert  Type = 2 // PARTITION_VERT
	Split Type = 3 // PARTITION_SPLIT
)

// partitionTree is the VP9 partition binary tree (spec §7.5.4).
// ReadTree uses: node index i, read ReadBool(probs[i/2]),
// follow tree[i + bit]; negative values are leaves (-Type).
var partitionTree = []int8{
	-int8(None), 2,
	-int8(Horz), 4,
	-int8(Vert), -int8(Split),
}

// Read decodes a partition type from the bool coder.
// ctx is the partition context (0..15), derived from block size level and
// above/left partition flags.  probs is Probs.PartitionProbs[ctx][:].
// counts is optional (nil disables accumulation).
func Read(br *boolcoder.Reader, ctx int, probs [16][3]uint8, counts *compheader.Counts) Type {
	p := probs[ctx][:]
	var c [][2]uint32
	if counts != nil {
		c = counts.PartitionProbs[ctx][:]
	}
	return Type(br.ReadTreeCount(partitionTree, p, c))
}

// Context computes the partition context index (0..15) for a block at the
// given MI-unit position.
//
//   - above: true if the MI unit directly above this block is the start of a
//     partition at the same block-size level.
//   - left: true for the MI unit to the left.
//   - bsl: block size log2 in MI units (1=8x8, 2=16x16, 3=32x32, 4=64x64).
//
// The 16 contexts are grouped into 4 levels (bsl 1..4), with 4 above/left
// combinations each, matching default_partition_probs in spec Appendix B.
func Context(above, left bool, bsl int) int {
	a := 0
	if above {
		a = 1
	}
	l := 0
	if left {
		l = 1
	}
	return (l*2+a) + (bsl-1)*4
}
