// Package compheader decodes the VP9 compressed header (spec §7.3).
// The compressed header is bool-coded (boolcoder.Reader) and carries probability
// delta updates for all adaptive entropy-coding contexts in a VP9 frame.
// Original implementation from the public VP9 Bitstream & Decoding Process Spec.
package compheader

import "github.com/mgvs/go-vp9/boolcoder"

// ---- TX mode constants (spec §7.3.2) ----

const (
	TxModeOnly4x4    = 0
	TxModeAllow8x8   = 1
	TxModeAllow16x16 = 2
	TxModeAllow32x32 = 3
	TxModeSelect     = 4
)

// ---- Interpolation filter constants ----

const (
	InterpEightTap       = 0
	InterpEightTapSmooth = 1
	InterpEightTapSharp  = 2
	InterpBilinear       = 3
	InterpSwitchable     = 4
)

// ---- Reference mode constants ----

const (
	RefModeSingle   = 0
	RefModeCompound = 1
	RefModeSelect   = 2
)

// ---- internal table dimensions ----

const (
	txSizes           = 4 // 4×4 8×8 16×16 32×32
	txSizeContexts    = 2
	planeTypes        = 2
	refTypes          = 2
	coefBands         = 6
	prevCoefContexts  = 6 // PREV_COEF_CONTEXTS; band-0 only uses [0..2]
	uncostrainedNodes = 3 // UNCONSTRAINED_NODES
	skipContexts      = 3
	interModeContexts = 7
	interModeDelta    = 3 // INTER_MODES - 1
	switchFilterCtx   = 4 // SWITCHABLE_FILTER_CONTEXTS
	switchFilterDelta = 2 // SWITCHABLE_FILTERS - 1
	isInterContexts   = 4
	compInterContexts = 5
	refContexts       = 5 // REF_CONTEXTS (single/comp ref probs)
	blockSizeGroups   = 4
	intraModes        = 10
	intraModeDelta    = 9 // INTRA_MODES - 1
	partitionContexts = 16
	partitionDelta    = 3  // PARTITION_TYPES - 1
	mvJointDelta      = 3  // MV_JOINTS - 1
	mvComponents      = 2  // x and y
	mvClassDelta      = 10 // MV_CLASSES - 1
	class0FrContexts  = 2  // CLASS0_SIZE
	mvOffsetBits      = 10 // MV_OFFSET_BITS
	mvFrDelta         = 3  // MV_FP_SIZE - 1
)

// ---- Probs ----

// Probs holds all VP9 frame-level adaptive probability tables.
// These are updated by each inter frame's compressed header.
// Use NewProbs() to get a neutral starting state; DefaultProbs() (M3) for
// the spec Appendix B initial values.
type Probs struct {
	TxMode        int
	ReferenceMode int

	// Transform size probs (only when TxMode == TxModeSelect, §7.3.2).
	TxProbs8  [txSizeContexts]uint8    // 1 prob per context (TX_8X8 path)
	TxProbs16 [txSizeContexts][2]uint8 // 2 probs per context
	TxProbs32 [txSizeContexts][3]uint8 // 3 probs per context

	// Coefficient probabilities (§7.3.3).
	CoefProbs [txSizes][planeTypes][refTypes][coefBands][prevCoefContexts][uncostrainedNodes]uint8

	// Skip flag probabilities (§7.3.4).
	SkipProbs [skipContexts]uint8

	// Inter-frame probabilities (§7.3.5 – §7.3.12).
	InterModeProbs    [interModeContexts][interModeDelta]uint8
	InterpFilterProbs [switchFilterCtx][switchFilterDelta]uint8
	IsInterProbs      [isInterContexts]uint8
	CompInterProbs    [compInterContexts]uint8
	SingleRefProbs    [refContexts][2]uint8
	CompRefProbs      [refContexts]uint8
	YModeProbs        [blockSizeGroups][intraModeDelta]uint8
	UVModeProbs       [intraModes][intraModeDelta]uint8
	PartitionProbs    [partitionContexts][partitionDelta]uint8

	// MV probabilities (§7.3.12).
	MVJointProbs     [mvJointDelta]uint8
	MVSignProbs      [mvComponents]uint8
	MVClassProbs     [mvComponents][mvClassDelta]uint8
	MVClass0BitProbs [mvComponents]uint8
	MVBitsProbs      [mvComponents][mvOffsetBits]uint8
	MVClass0FrProbs  [mvComponents][class0FrContexts][mvFrDelta]uint8
	MVFrProbs        [mvComponents][mvFrDelta]uint8
	MVClass0HPProbs  [mvComponents]uint8
	MVHPProbs        [mvComponents]uint8
}

// NewProbs returns a Probs with every probability set to 128 (neutral/mid).
// Suitable for null-update testing; use DefaultProbs() (M3) for real decoding.
func NewProbs() *Probs {
	p := &Probs{}
	fillU8 := func(s []uint8) {
		for i := range s {
			s[i] = 128
		}
	}
	fillU8(p.TxProbs8[:])
	for i := range p.TxProbs16 {
		fillU8(p.TxProbs16[i][:])
	}
	for i := range p.TxProbs32 {
		fillU8(p.TxProbs32[i][:])
	}
	for i := range p.CoefProbs {
		for j := range p.CoefProbs[i] {
			for k := range p.CoefProbs[i][j] {
				for l := range p.CoefProbs[i][j][k] {
					for m := range p.CoefProbs[i][j][k][l] {
						fillU8(p.CoefProbs[i][j][k][l][m][:])
					}
				}
			}
		}
	}
	fillU8(p.SkipProbs[:])
	for i := range p.InterModeProbs {
		fillU8(p.InterModeProbs[i][:])
	}
	for i := range p.InterpFilterProbs {
		fillU8(p.InterpFilterProbs[i][:])
	}
	fillU8(p.IsInterProbs[:])
	fillU8(p.CompInterProbs[:])
	for i := range p.SingleRefProbs {
		fillU8(p.SingleRefProbs[i][:])
	}
	fillU8(p.CompRefProbs[:])
	for i := range p.YModeProbs {
		fillU8(p.YModeProbs[i][:])
	}
	for i := range p.UVModeProbs {
		fillU8(p.UVModeProbs[i][:])
	}
	for i := range p.PartitionProbs {
		fillU8(p.PartitionProbs[i][:])
	}
	fillU8(p.MVJointProbs[:])
	fillU8(p.MVSignProbs[:])
	for i := range p.MVClassProbs {
		fillU8(p.MVClassProbs[i][:])
	}
	fillU8(p.MVClass0BitProbs[:])
	for i := range p.MVBitsProbs {
		fillU8(p.MVBitsProbs[i][:])
	}
	for i := range p.MVClass0FrProbs {
		for j := range p.MVClass0FrProbs[i] {
			fillU8(p.MVClass0FrProbs[i][j][:])
		}
	}
	for i := range p.MVFrProbs {
		fillU8(p.MVFrProbs[i][:])
	}
	fillU8(p.MVClass0HPProbs[:])
	fillU8(p.MVHPProbs[:])
	return p
}

// ---- invMapTable (VP9 spec §7.3.5) ----

// invMapTable is the 254-entry inverse probability mapping table from the VP9 spec.
// It maps a decoded subexponential integer [0..253] to a centred delta value,
// which is then passed through invRecenterNonneg to obtain the updated probability.
var invMapTable = [254]uint8{
	7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176,
	189, 202, 215, 228, 241, 254, 1, 2, 3, 4, 5, 6, 8, 9,
	10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24,
	25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39,
	40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54,
	55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
	70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
	86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100,
	101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115,
	116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130,
	131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145,
	146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
	161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
	177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191,
	192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206,
	207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221,
	222, 223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236,
	237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
	252, 253,
}

// invRecenterNonneg is the inverse of the recentering function used in
// probability delta coding (VP9 spec §7.3.5 / libvpx inv_recenter_nonneg).
func invRecenterNonneg(v, m int) int {
	if v > m<<1 {
		return v
	}
	if v&1 == 0 {
		return (v >> 1) + m
	}
	return m - ((v + 1) >> 1)
}

// invRemapProb maps a decoded subexponential integer v and the current
// probability m to the updated probability (spec §7.3.5).
// v must be in [0, 253].
func invRemapProb(v int, m uint8) uint8 {
	if v >= len(invMapTable) {
		v = len(invMapTable) - 1
	}
	x := int(invMapTable[v])
	result := invRecenterNonneg(x, int(m)-1) + 1
	if result < 1 {
		return 1
	}
	if result > 255 {
		return 255
	}
	return uint8(result)
}

// decodeTermSubexp reads a subexponential-coded integer from the bool stream
// (VP9 spec §7.3.5 / libvpx decode_term_subexp). Returns a value in [0, 319];
// valid prob-delta inputs are [0, 253].
func decodeTermSubexp(br *boolcoder.Reader) int {
	if br.ReadBit() == 0 {
		return int(br.ReadLiteral(4)) // [0, 15]
	}
	if br.ReadBit() == 0 {
		return 16 + int(br.ReadLiteral(4)) // [16, 31]
	}
	if br.ReadBit() == 0 {
		return 32 + int(br.ReadLiteral(5)) // [32, 63]
	}
	return 64 + int(br.ReadLiteral(8)) // [64, 319]
}

// diffUpdateProb conditionally updates a probability from the bool stream
// (spec §7.3.5). Read prob 252 ≈ 98.4% chance of "no update".
func diffUpdateProb(br *boolcoder.Reader, p uint8) uint8 {
	if br.ReadBool(252) == 0 {
		return p
	}
	v := decodeTermSubexp(br)
	return invRemapProb(v, p)
}

// ---- Decode ----

// Decode reads the VP9 compressed header from br and updates probs in-place
// (spec §7.3). isIntra=true for key frames and intra-only frames.
// interpFilter is the interpolation filter from the uncompressed header
// (InterpSwitchable triggers reading interp filter probs).
// allowHPMV controls whether high-precision MV probs are read.
func Decode(br *boolcoder.Reader, isIntra bool, interpFilter int, allowHPMV bool, probs *Probs) error {
	probs.TxMode = readTxMode(br)
	if probs.TxMode == TxModeSelect {
		readTxModeProbs(br, probs)
	}
	readCoefProbs(br, probs)
	readSkipProbs(br, probs)
	if !isIntra {
		readInterModeProbs(br, probs)
		if interpFilter == InterpSwitchable {
			readInterpFilterProbs(br, probs)
		}
		readIsInterProbs(br, probs)
		readFrameReferenceMode(br, probs)
		readFrameReferenceModeProbs(br, probs)
		readYModeProbs(br, probs)
		readUVModeProbs(br, probs)
		readPartitionProbs(br, probs)
		readMVProbs(br, allowHPMV, probs)
	}
	return nil
}

// readTxMode reads tx_mode from the bool stream (§7.3.2).
func readTxMode(br *boolcoder.Reader) int {
	m := int(br.ReadLiteral(2))
	if m == 3 {
		m += br.ReadBit()
	}
	return m
}

// maxTxSizeForMode returns the largest TX size enabled by a given tx_mode.
func maxTxSizeForMode(txMode int) int {
	if txMode >= txSizes-1 {
		return txSizes - 1
	}
	return txMode
}

// readTxModeProbs reads the TX mode probability updates (§7.3.2).
func readTxModeProbs(br *boolcoder.Reader, p *Probs) {
	for i := range p.TxProbs8 {
		p.TxProbs8[i] = diffUpdateProb(br, p.TxProbs8[i])
	}
	for i := range p.TxProbs16 {
		for j := range p.TxProbs16[i] {
			p.TxProbs16[i][j] = diffUpdateProb(br, p.TxProbs16[i][j])
		}
	}
	for i := range p.TxProbs32 {
		for j := range p.TxProbs32[i] {
			p.TxProbs32[i][j] = diffUpdateProb(br, p.TxProbs32[i][j])
		}
	}
}

// bandCoeffContexts returns BAND_COEFF_CONTEXTS(band): 3 for band 0, 6 otherwise.
func bandCoeffContexts(band int) int {
	if band == 0 {
		return 3
	}
	return prevCoefContexts
}

// readCoefProbs reads coefficient probability updates (§7.3.3).
func readCoefProbs(br *boolcoder.Reader, p *Probs) {
	maxTx := maxTxSizeForMode(p.TxMode)
	for tx := 0; tx <= maxTx; tx++ {
		for pt := 0; pt < planeTypes; pt++ {
			for rt := 0; rt < refTypes; rt++ {
				for band := 0; band < coefBands; band++ {
					nCtx := bandCoeffContexts(band)
					for ctx := 0; ctx < nCtx; ctx++ {
						for n := 0; n < uncostrainedNodes; n++ {
							p.CoefProbs[tx][pt][rt][band][ctx][n] =
								diffUpdateProb(br, p.CoefProbs[tx][pt][rt][band][ctx][n])
						}
					}
				}
			}
		}
	}
}

// readSkipProbs reads skip flag probability updates (§7.3.4).
func readSkipProbs(br *boolcoder.Reader, p *Probs) {
	for i := range p.SkipProbs {
		p.SkipProbs[i] = diffUpdateProb(br, p.SkipProbs[i])
	}
}

// readInterModeProbs reads inter mode probability updates (§7.3.5).
func readInterModeProbs(br *boolcoder.Reader, p *Probs) {
	for i := range p.InterModeProbs {
		for j := range p.InterModeProbs[i] {
			p.InterModeProbs[i][j] = diffUpdateProb(br, p.InterModeProbs[i][j])
		}
	}
}

// readInterpFilterProbs reads switchable interp filter probability updates (§7.3.6).
func readInterpFilterProbs(br *boolcoder.Reader, p *Probs) {
	for i := range p.InterpFilterProbs {
		for j := range p.InterpFilterProbs[i] {
			p.InterpFilterProbs[i][j] = diffUpdateProb(br, p.InterpFilterProbs[i][j])
		}
	}
}

// readIsInterProbs reads is_inter probability updates (§7.3.7).
func readIsInterProbs(br *boolcoder.Reader, p *Probs) {
	for i := range p.IsInterProbs {
		p.IsInterProbs[i] = diffUpdateProb(br, p.IsInterProbs[i])
	}
}

// readFrameReferenceMode reads the reference mode field (§7.3.8).
func readFrameReferenceMode(br *boolcoder.Reader, p *Probs) {
	if br.ReadBit() == 1 {
		if br.ReadBit() == 1 {
			p.ReferenceMode = RefModeSelect
		} else {
			p.ReferenceMode = RefModeCompound
		}
	} else {
		p.ReferenceMode = RefModeSingle
	}
}

// readFrameReferenceModeProbs reads ref mode probability updates (§7.3.9).
func readFrameReferenceModeProbs(br *boolcoder.Reader, p *Probs) {
	if p.ReferenceMode == RefModeSelect {
		for i := range p.CompInterProbs {
			p.CompInterProbs[i] = diffUpdateProb(br, p.CompInterProbs[i])
		}
	}
	if p.ReferenceMode != RefModeSingle {
		for i := range p.CompRefProbs {
			p.CompRefProbs[i] = diffUpdateProb(br, p.CompRefProbs[i])
		}
	}
	if p.ReferenceMode != RefModeCompound {
		for i := range p.SingleRefProbs {
			p.SingleRefProbs[i][0] = diffUpdateProb(br, p.SingleRefProbs[i][0])
			p.SingleRefProbs[i][1] = diffUpdateProb(br, p.SingleRefProbs[i][1])
		}
	}
}

// readYModeProbs reads luma intra mode probability updates (§7.3.10).
func readYModeProbs(br *boolcoder.Reader, p *Probs) {
	for i := range p.YModeProbs {
		for j := range p.YModeProbs[i] {
			p.YModeProbs[i][j] = diffUpdateProb(br, p.YModeProbs[i][j])
		}
	}
}

// readUVModeProbs reads chroma intra mode probability updates (§7.3.10).
func readUVModeProbs(br *boolcoder.Reader, p *Probs) {
	for i := range p.UVModeProbs {
		for j := range p.UVModeProbs[i] {
			p.UVModeProbs[i][j] = diffUpdateProb(br, p.UVModeProbs[i][j])
		}
	}
}

// readPartitionProbs reads partition probability updates (§7.3.11).
func readPartitionProbs(br *boolcoder.Reader, p *Probs) {
	for i := range p.PartitionProbs {
		for j := range p.PartitionProbs[i] {
			p.PartitionProbs[i][j] = diffUpdateProb(br, p.PartitionProbs[i][j])
		}
	}
}

// readMVProbs reads motion vector probability updates (§7.3.12).
func readMVProbs(br *boolcoder.Reader, allowHP bool, p *Probs) {
	// Joint
	for i := range p.MVJointProbs {
		p.MVJointProbs[i] = updateMVProb(br, p.MVJointProbs[i])
	}
	// Per-component
	for c := 0; c < mvComponents; c++ {
		p.MVSignProbs[c] = updateMVProb(br, p.MVSignProbs[c])
		for i := range p.MVClassProbs[c] {
			p.MVClassProbs[c][i] = updateMVProb(br, p.MVClassProbs[c][i])
		}
		p.MVClass0BitProbs[c] = updateMVProb(br, p.MVClass0BitProbs[c])
		for i := range p.MVBitsProbs[c] {
			p.MVBitsProbs[c][i] = updateMVProb(br, p.MVBitsProbs[c][i])
		}
	}
	for c := 0; c < mvComponents; c++ {
		for j := range p.MVClass0FrProbs[c] {
			for i := range p.MVClass0FrProbs[c][j] {
				p.MVClass0FrProbs[c][j][i] = updateMVProb(br, p.MVClass0FrProbs[c][j][i])
			}
		}
		for i := range p.MVFrProbs[c] {
			p.MVFrProbs[c][i] = updateMVProb(br, p.MVFrProbs[c][i])
		}
	}
	if allowHP {
		for c := 0; c < mvComponents; c++ {
			p.MVClass0HPProbs[c] = updateMVProb(br, p.MVClass0HPProbs[c])
			p.MVHPProbs[c] = updateMVProb(br, p.MVHPProbs[c])
		}
	}
}

// updateMVProb reads a MV probability update: flag at prob 252, delta via subexp (§7.3.12).
// MV probs use the same diff_update_prob mechanism as coef probs.
func updateMVProb(br *boolcoder.Reader, p uint8) uint8 {
	return diffUpdateProb(br, p)
}
