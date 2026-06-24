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

import "github.com/mgvs/go-vp9/boolcoder"

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
	// QIndexDelta is added to (or replaces, when AbsOrDelta=true) the base
	// quantizer index when FeatAltQ is enabled for this segment.
	QIndexDelta int
	AbsOrDelta  bool // true = QIndexDelta is absolute; false = delta to BaseQIdx

	// LFDelta is added to the frame-level loop-filter level when FeatAltLF
	// is enabled for this segment.
	LFDelta int

	// RefFrame is the forced reference frame when FeatRefFrame is enabled.
	// -1 means the feature is not active and prediction uses the coded ref.
	RefFrame int

	// Skip forces all residual coefficients to zero when FeatSkip is enabled.
	Skip bool
}

// DecodeSegID decodes a segment ID from the bool coder using the 7 segment tree
// probabilities (spec §8.3.3).
//
// The segment tree is a balanced binary tree of depth 3, producing values 0..7:
//
//	probs[0]: 0 → left half  (0..3)
//	          1 → right half (4..7)
//	probs[1/4]: further split
//	probs[2/3/5/6]: leaf selection
func DecodeSegID(br *boolcoder.Reader, probs [7]uint8) int {
	if br.ReadBool(probs[0]) == 0 {
		if br.ReadBool(probs[1]) == 0 {
			return br.ReadBool(probs[2])
		}
		return 2 + br.ReadBool(probs[3])
	}
	if br.ReadBool(probs[4]) == 0 {
		return 4 + br.ReadBool(probs[5])
	}
	return 6 + br.ReadBool(probs[6])
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
		p.QIndexDelta = int(featData[id][FeatAltQ])
	}
	if featEnabled[id][FeatAltLF] {
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

// ClampQIndex returns a valid quantizer index after applying the segment
// QIndexDelta (or absolute override) to the frame's BaseQIdx.
func ClampQIndex(baseQIdx int, p Params) int {
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
