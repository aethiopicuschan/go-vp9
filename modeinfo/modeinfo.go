// Package modeinfo decodes VP9 intra prediction modes for key frames (spec §6.3.4).
// VP9 has 10 intra prediction modes for luma (Y) and the same 10 for chroma
// (UV). Both use the same binary tree; the probability tables differ.
package modeinfo

import "github.com/aethiopicuschan/go-vp9/boolcoder"

// IntraMode is a VP9 intra prediction mode (spec §7.5.4).
type IntraMode int

const (
	DC    IntraMode = 0 // DC_PRED
	V     IntraMode = 1 // V_PRED (vertical)
	H     IntraMode = 2 // H_PRED (horizontal)
	D45   IntraMode = 3 // D45_PRED
	D135  IntraMode = 4 // D135_PRED
	D117  IntraMode = 5 // D117_PRED
	D153  IntraMode = 6 // D153_PRED
	D207  IntraMode = 7 // D207_PRED
	D63   IntraMode = 8 // D63_PRED
	TM    IntraMode = 9 // TM_PRED (true-motion)
	Count           = 10
)

// intraModeTree is the VP9 intra-mode binary tree (spec Table 5 / §7.5.4).
// Node i uses prob probs[i/2]; tree[i+bit] is the next node (positive) or
// negated leaf (negative). Transcribed from libvpx vp9_intra_mode_tree.
var intraModeTree = []int8{
	-int8(DC), 2,
	-int8(TM), 4,
	-int8(V), 6,
	8, 12,
	-int8(H), 10,
	-int8(D135), -int8(D117),
	-int8(D45), 14,
	-int8(D63), 16,
	-int8(D153), -int8(D207),
}

// ReadYMode decodes the luma intra prediction mode for a block.
// sizeGroup selects the probability row (0..3) from YModeProbs:
//
//	0 = blocks < 8×8, 1 = 8×8..15×15, 2 = 16×16..31×31, 3 = ≥32×32.
//
// probs is Probs.YModeProbs. counts is optional (nil disables accumulation).
func ReadYMode(br *boolcoder.Reader, sizeGroup int, yModeProbs [4][9]uint8, counts [][2]uint32) IntraMode {
	return IntraMode(br.ReadTreeCount(intraModeTree, yModeProbs[sizeGroup][:], counts))
}

// ReadUVMode decodes the chroma intra prediction mode.
// yMode is the luma mode for this block (used as probability table index).
// probs is Probs.UVModeProbs. counts is optional (nil disables accumulation).
func ReadUVMode(br *boolcoder.Reader, yMode IntraMode, uvModeProbs [10][9]uint8, counts [][2]uint32) IntraMode {
	return IntraMode(br.ReadTreeCount(intraModeTree, uvModeProbs[int(yMode)][:], counts))
}

// BlockSizeGroup returns the size group index (0..3) for a block given its
// width and height in pixels.  Used to select the Y-mode probability row.
func BlockSizeGroup(w, h int) int {
	sz := w
	if h > sz {
		sz = h
	}
	switch {
	case sz < 8:
		return 0
	case sz < 16:
		return 1
	case sz < 32:
		return 2
	default:
		return 3
	}
}
