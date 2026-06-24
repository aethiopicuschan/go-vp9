// Package ctx manages the above-row and left-column context arrays used for
// adaptive entropy coding in VP9 (spec §8.3).
//
// VP9 adapts almost every coded symbol (partition type, prediction mode, TX
// size, skip flag, reference frame, coefficients) to the decoded state of
// neighbouring blocks.  Two sets of arrays track that state:
//
//   - Row  — above-row contexts, spanning the full tile width.
//     Reset to intra/zero defaults at the start of each tile row.
//   - Col  — left-column contexts, spanning one SB height (16 MI rows).
//     Reset to intra/zero defaults at the start of each SB row.
//
// Arrays are indexed in units of 4×4 MI blocks (the smallest coding unit).
// The partition context uses 8×8-unit indexing (N4/2).
//
// Reference frame constants (shared with mvpred):
//
//	RefIntra = 0  (no inter reference, i.e. intra block)
//	RefLast  = 1
//	RefGolden = 2
//	RefAltRef = 3
package ctx

// Reference frame identifiers used in Ref arrays.
const (
	RefIntra  = 0
	RefLast   = 1
	RefGolden = 2
	RefAltRef = 3
)

// TX size constants (matching transform package).
const (
	TX4x4   = 0
	TX8x8   = 1
	TX16x16 = 2
	TX32x32 = 3
)

// Interpolation filter context sentinel: "no inter block / filter not set".
const InterpSwitchable = 3

// Row holds above-row context arrays for one tile row.
// Reset at the start of each tile row to "no-block" defaults.
// All arrays are indexed by 4×4-column position (0 .. N4-1).
type Row struct {
	N4 int // number of 4×4 columns in the tile

	// Partition context per 8×8 unit (length N4/2).
	// Each byte is a bitmask: bit k set means the 8×8-aligned position is
	// occupied by a sub-block of level k (libvpx bsl: 0=8×8, 1=16×16 etc.).
	// Read by PartitionAbove; written by SetPartition.
	Partition []uint8

	TxSize       []uint8   // TX size (TX4x4..TX32x32), length N4
	Skip         []bool    // skip flag, length N4
	Ref          [][2]int  // [primary, compound] reference frames, length N4
	Inter        []bool    // true = inter block, length N4
	ZeroMV       []bool    // true = inter block with zero primary MV, length N4
	YMode        []uint8   // intra Y mode (0=DC .. 9=TM), length N4
	NZC          [3][]bool // nonzero-coef flag per plane [Y/U/V], each length N4
	InterpFilter []uint8   // interpolation filter (0..3); InterpSwitchable(3) = not set
}

// Col holds left-column context for one superblock row (16 MI rows = 64 px).
// Reset at the start of each superblock row to "no-block" defaults.
// All arrays are indexed by 4×4-row position (0 .. N4-1).
type Col struct {
	N4 int // number of 4×4 rows in the SB (always 16 for 64-px SBs)

	Partition    []uint8
	TxSize       []uint8
	Skip         []bool
	Ref          [][2]int
	Inter        []bool
	ZeroMV       []bool
	YMode        []uint8
	NZC          [3][]bool
	InterpFilter []uint8
}

// NewRow allocates a Row for a tile of width w pixels.
func NewRow(w int) *Row {
	n4 := (w + 3) / 4
	r := &Row{N4: n4}
	r.Partition = make([]uint8, (n4+1)/2)
	r.TxSize = make([]uint8, n4)
	r.Skip = make([]bool, n4)
	r.Ref = make([][2]int, n4)
	r.Inter = make([]bool, n4)
	r.ZeroMV = make([]bool, n4)
	r.YMode = make([]uint8, n4)
	for p := 0; p < 3; p++ {
		r.NZC[p] = make([]bool, n4)
	}
	r.InterpFilter = make([]uint8, n4)
	r.Reset()
	return r
}

// NewCol allocates a Col for a superblock height of 64 pixels (16 MI rows).
func NewCol() *Col {
	const sbMI = 16 // 64 pixels / 4
	c := &Col{N4: sbMI}
	c.Partition = make([]uint8, sbMI/2)
	c.TxSize = make([]uint8, sbMI)
	c.Skip = make([]bool, sbMI)
	c.Ref = make([][2]int, sbMI)
	c.Inter = make([]bool, sbMI)
	c.ZeroMV = make([]bool, sbMI)
	c.YMode = make([]uint8, sbMI)
	for p := 0; p < 3; p++ {
		c.NZC[p] = make([]bool, sbMI)
	}
	c.InterpFilter = make([]uint8, sbMI)
	c.Reset()
	return c
}

// Reset clears the Row to "no above block" defaults:
//   - Partition bitmask = 0 (no occupancy)
//   - TxSize = TX32x32 (largest, so context defaults to "both big")
//   - Ref = [RefIntra, RefIntra]
//   - Inter, Skip, NZC = false
//   - YMode = 0 (DC_PRED)
func (r *Row) Reset() {
	for i := range r.Partition {
		r.Partition[i] = 0
	}
	for i := range r.TxSize {
		r.TxSize[i] = TX32x32
	}
	for i := range r.Skip {
		r.Skip[i] = false
	}
	for i := range r.Ref {
		r.Ref[i] = [2]int{RefIntra, RefIntra}
	}
	for i := range r.Inter {
		r.Inter[i] = false
	}
	for i := range r.ZeroMV {
		r.ZeroMV[i] = false
	}
	for i := range r.YMode {
		r.YMode[i] = 0
	}
	for p := 0; p < 3; p++ {
		for i := range r.NZC[p] {
			r.NZC[p][i] = false
		}
	}
	for i := range r.InterpFilter {
		r.InterpFilter[i] = InterpSwitchable
	}
}

// Reset clears the Col to "no left block" defaults (same semantics as Row.Reset).
func (c *Col) Reset() {
	for i := range c.Partition {
		c.Partition[i] = 0
	}
	for i := range c.TxSize {
		c.TxSize[i] = TX32x32
	}
	for i := range c.Skip {
		c.Skip[i] = false
	}
	for i := range c.Ref {
		c.Ref[i] = [2]int{RefIntra, RefIntra}
	}
	for i := range c.Inter {
		c.Inter[i] = false
	}
	for i := range c.ZeroMV {
		c.ZeroMV[i] = false
	}
	for i := range c.YMode {
		c.YMode[i] = 0
	}
	for p := 0; p < 3; p++ {
		for i := range c.NZC[p] {
			c.NZC[p][i] = false
		}
	}
	for i := range c.InterpFilter {
		c.InterpFilter[i] = InterpSwitchable
	}
}

// ---- Partition context -------------------------------------------------------

// PartitionAboveMasks[bsLog2] is the bitmask written into the above partition
// context array after decoding a PARTITION_NONE block of that size.
// bsLog2: 1=8×8, 2=16×16, 3=32×32, 4=64×64.
// These values match libvpx partition_context_lookup.above for subsize==bsize.
var PartitionAboveMasks = [5]uint8{
	0,  // unused (bsLog2=0)
	15, // 8×8   → 0b00001111
	7,  // 16×16 → 0b00000111
	3,  // 32×32 → 0b00000011
	0,  // 64×64 → 0b00000000
}

// PartitionLeftMasks mirrors PartitionAboveMasks for the left context.
var PartitionLeftMasks = PartitionAboveMasks

// PartitionAbove returns the above partition flag for partition.Context.
// col4 is the 4×4-column index; bsl is our bsLog2 (1..4).
// Returns true when the above context bit at level (bsl-1) is set.
func PartitionAbove(r *Row, col4, bsl int) bool {
	idx := col4 / 2
	if idx >= len(r.Partition) {
		return false
	}
	return (r.Partition[idx]>>(uint(bsl-1)))&1 != 0
}

// PartitionLeft returns the left partition flag for partition.Context.
func PartitionLeft(c *Col, row4, bsl int) bool {
	idx := row4 / 2
	if idx >= len(c.Partition) {
		return false
	}
	return (c.Partition[idx]>>(uint(bsl-1)))&1 != 0
}

// SetPartition writes bitmask to above partition context for n8 consecutive
// 8×8 units starting at 8×8-column index col8.
func (r *Row) SetPartition(col8, n8 int, mask uint8) {
	for i := 0; i < n8 && col8+i < len(r.Partition); i++ {
		r.Partition[col8+i] = mask
	}
}

// SetPartition writes bitmask to left partition context for n8 consecutive
// 8×8 units starting at 8×8-row index row8.
func (c *Col) SetPartition(row8, n8 int, mask uint8) {
	for i := 0; i < n8 && row8+i < len(c.Partition); i++ {
		c.Partition[row8+i] = mask
	}
}

// ---- Block-level setters (fill n4 MI units starting at index i) -------------

// SetBlock fills the above-row context for a decoded block at 4×4-column col4
// of width n4 MI units.
func (r *Row) SetBlock(col4, n4 int, txSize uint8, skip bool, ref [2]int, inter, zeroMV bool, yMode uint8, interpFilter uint8) {
	end := col4 + n4
	if end > r.N4 {
		end = r.N4
	}
	for i := col4; i < end; i++ {
		r.TxSize[i] = txSize
		r.Skip[i] = skip
		r.Ref[i] = ref
		r.Inter[i] = inter
		r.ZeroMV[i] = zeroMV
		r.YMode[i] = yMode
		r.InterpFilter[i] = interpFilter
	}
}

// SetBlock fills the left-column context for a decoded block at 4×4-row row4
// of height n4 MI units.
func (c *Col) SetBlock(row4, n4 int, txSize uint8, skip bool, ref [2]int, inter, zeroMV bool, yMode uint8, interpFilter uint8) {
	end := row4 + n4
	if end > c.N4 {
		end = c.N4
	}
	for i := row4; i < end; i++ {
		c.TxSize[i] = txSize
		c.Skip[i] = skip
		c.Ref[i] = ref
		c.Inter[i] = inter
		c.ZeroMV[i] = zeroMV
		c.YMode[i] = yMode
		c.InterpFilter[i] = interpFilter
	}
}

// SetNZC sets the nonzero-coefficient flag for plane p across n4 MI units.
func (r *Row) SetNZC(col4, n4, plane int, nzc bool) {
	end := col4 + n4
	if end > r.N4 {
		end = r.N4
	}
	for i := col4; i < end; i++ {
		r.NZC[plane][i] = nzc
	}
}

// SetNZC sets the nonzero-coefficient flag for plane p across n4 MI units.
func (c *Col) SetNZC(row4, n4, plane int, nzc bool) {
	end := row4 + n4
	if end > c.N4 {
		end = c.N4
	}
	for i := row4; i < end; i++ {
		c.NZC[plane][i] = nzc
	}
}

// ---- Context computation functions ------------------------------------------

// SkipCtx returns the skip probability context (0..2) from above and left
// skip flags (spec §8.3.3, skipContexts=3).
func SkipCtx(aboveSkip, leftSkip bool) int {
	n := 0
	if aboveSkip {
		n++
	}
	if leftSkip {
		n++
	}
	return n
}

// TxSizeCtx returns the TX-mode-select context (0 or 1) from above/left TX
// sizes and the maximum allowed TX size for the current block
// (spec §8.3.1, txSizeContexts=2).
//
// hasAbove/hasLeft indicate whether a valid block exists in that direction.
// When absent, that neighbour does not contribute to the context.
func TxSizeCtx(hasAbove, hasLeft bool, aboveTx, leftTx, maxTx uint8) int {
	ctx := 0
	if hasAbove && aboveTx >= maxTx {
		ctx++
	}
	if hasLeft && leftTx >= maxTx {
		ctx++
	}
	if ctx > 1 {
		ctx = 1
	}
	return ctx
}

// IsInterCtx returns the is_inter probability context (0..3) from above/left
// availability and their inter/intra status (spec §8.3.3, isInterContexts=4).
//
//	3 — both available and both intra
//	1 — both available but mixed inter/intra
//	0 — both available and both inter
//	2 — only one available and it is intra
//	0 — only one available and it is inter
//	0 — neither available
func IsInterCtx(hasAbove, hasLeft, aboveInter, leftInter bool) int {
	if hasAbove && hasLeft {
		aboveIntra := !aboveInter
		leftIntra := !leftInter
		if leftIntra && aboveIntra {
			return 3
		}
		if leftIntra || aboveIntra {
			return 1
		}
		return 0
	}
	if hasAbove || hasLeft {
		isInter := (hasAbove && aboveInter) || (hasLeft && leftInter)
		if !isInter {
			return 2
		}
		return 0
	}
	return 0
}

// NZCInitCtx returns the initial coefficient decode context (0..2) from above
// and left nonzero-coefficient flags for the given plane.
// This is the initCtx parameter for coeftoken.DecodeBlock.
//
//	0 — both zero (very likely EOB)
//	1 — one nonzero
//	2 — both nonzero
func NZCInitCtx(aboveNZC, leftNZC bool) int {
	n := 0
	if aboveNZC {
		n++
	}
	if leftNZC {
		n++
	}
	return n
}

// InterpFilterCtx returns the switchable interpolation filter context (0..3)
// from above and left inter blocks' interpolation filters.
// Returns InterpSwitchable (3) when no inter-block neighbours exist.
// This context indexes InterpFilterProbs[ctx][...] (switchFilterCtx=4).
func InterpFilterCtx(hasAboveInter, hasLeftInter bool, aboveFilter, leftFilter uint8) int {
	if hasAboveInter && hasLeftInter {
		if aboveFilter == leftFilter {
			return int(aboveFilter)
		}
		return InterpSwitchable
	}
	if hasAboveInter {
		return int(aboveFilter)
	}
	if hasLeftInter {
		return int(leftFilter)
	}
	return InterpSwitchable
}

// InterModeCtx returns the inter_mode probability context (0..6) for a block
// whose primary reference frame is ref.  above and left are the reference
// frames and modes of the above and left blocks (as returned by the Ref and
// Inter fields of Row/Col).
//
// The formula follows libvpx vp9_get_inter_mode_ctx (spec §8.3.3,
// interModeContexts=7).  A counter accumulates for each neighbour:
//
//	+0 for intra neighbours
//	+1 for inter neighbour with same reference but ZEROMV mode
//	+3 for inter neighbour with same reference and non-zero MV
//	+4 for inter neighbour with a different reference frame
//
// The final sum is clamped to [0, 6].
func InterModeCtx(hasAbove, hasLeft bool,
	aboveRef [2]int, leftRef [2]int,
	aboveInter, leftInter bool,
	aboveZeroMV, leftZeroMV bool,
	ref int) int {

	ctx := 0
	if hasAbove {
		if !aboveInter {
			// intra — contributes 0
		} else if aboveRef[0] != ref && aboveRef[1] != ref {
			ctx += 4
		} else if aboveZeroMV {
			ctx += 1
		} else {
			ctx += 3
		}
	}
	if hasLeft {
		if !leftInter {
			// intra — contributes 0
		} else if leftRef[0] != ref && leftRef[1] != ref {
			ctx += 4
		} else if leftZeroMV {
			ctx += 1
		} else {
			ctx += 3
		}
	}
	if ctx > 6 {
		ctx = 6
	}
	return ctx
}
