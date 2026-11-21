// Package segment decodes VP9 segment IDs and resolves per-segment parameters
// (spec §8.3.3, §8.7.2).
//
// VP9 supports 8 segments (IDs 0..7).  Each segment can independently override:
//
//   - FeatAltQ     (0): quantizer — delta or absolute
//   - FeatAltLF    (1): loop-filter level delta
//   - FeatRefFrame (2): forced reference frame
//   - FeatSkip     (3): force skip (no residual coefficients)
//
// The segment map probs (tree_probs[7]) are stored in the uncompressed header
// (frame.Header.SegTreeProbs).  The per-segment feature enables and values are
// also in the uncompressed header (frame.Header.SegFeat*).
package segment

import "github.com/aethiopicuschan/go-vp9/boolcoder"

// Feature indices.
const (
	FeatAltQ     = 0
	FeatAltLF    = 1
	FeatRefFrame = 2
	FeatSkip     = 3
)

// FeatureBits[f] is the number of value bits for feature f (spec Table 3).
// FeatSkip has 0 value bits (its presence alone signals skip).
var FeatureBits = [4]int{8, 6, 2, 0}

// Params holds the resolved segmentation parameters for one block.
type Params struct {
	// AltQ: FeatAltQ is enabled for this segment. QIndexDelta is then added to
	// (or replaces, when AbsOrDelta=true) the base quantizer index.
	AltQ        bool
	QIndexDelta int
	AbsOrDelta  bool // true = feature data is absolute; false = delta

	// AltLF: FeatAltLF is enabled for this segment. LFDelta is then added to
	// (or replaces, when AbsOrDelta=true) the frame-level loop-filter level.
	AltLF   bool
	LFDelta int

	// RefFrame is the forced reference frame when FeatRefFrame is enabled.
	// -1 means the feature is not active and prediction uses the coded ref.
	RefFrame int

	// Skip forces all residual coefficients to zero when FeatSkip is enabled.
	Skip bool
}

// DecodeSegID decodes a segment ID from the bool coder using the 7 segment tree
// probabilities (spec §8.3.3, libvpx vp9_segment_tree).
//
// The tree is a balanced binary tree of depth 3 whose nodes are numbered
// breadth-first, so the probabilities are laid out level by level:
//
//	probs[0]:    0 → 0..3, 1 → 4..7
//	probs[1/2]:  pair within 0..3 / within 4..7
//	probs[3..6]: leaf within the pair {0,1} {2,3} {4,5} {6,7}
func DecodeSegID(br *boolcoder.Reader, probs [7]uint8) int {
	b0 := br.ReadBool(probs[0])
	b1 := br.ReadBool(probs[1+b0])
	b2 := br.ReadBool(probs[3+2*b0+b1])
	return 4*b0 + 2*b1 + b2
}

// ParamsFor returns the resolved Params for the given segment id (0..7).
// featEnabled and featData come directly from frame.Header.SegFeat*.
// absOrDelta is frame.Header.SegAbsOrDelta.
//
// If id is out of range or no features are enabled, a zero Params is returned
// (no overrides, RefFrame = -1).
func ParamsFor(featEnabled [8][4]bool, featData [8][4]int16, absOrDelta bool, id int) Params {
	p := Params{RefFrame: -1, AbsOrDelta: absOrDelta}
	if id < 0 || id >= 8 {
		return p
	}
	if featEnabled[id][FeatAltQ] {
		p.AltQ = true
		p.QIndexDelta = int(featData[id][FeatAltQ])
	}
	if featEnabled[id][FeatAltLF] {
		p.AltLF = true
		p.LFDelta = int(featData[id][FeatAltLF])
	}
	if featEnabled[id][FeatRefFrame] {
		p.RefFrame = int(featData[id][FeatRefFrame])
	}
	if featEnabled[id][FeatSkip] {
		p.Skip = true
	}
	return p
}

// ClampQIndex returns the quantizer index for a segment (libvpx
// vp9_get_qindex): the frame's BaseQIdx unless FeatAltQ is enabled, in which
// case the segment's value is applied as a delta or an absolute index and the
// result clamped to 0..255.
func ClampQIndex(baseQIdx int, p Params) int {
	if !p.AltQ {
		return baseQIdx
	}
	q := baseQIdx
	if p.AbsOrDelta {
		q = p.QIndexDelta
	} else {
		q += p.QIndexDelta
	}
	if q < 0 {
		return 0
	}
	if q > 255 {
		return 255
	}
	return q
}
