// Package blockmode decodes VP9 mode information for one coding block
// (spec §8.3.3 / libvpx vp9/decoder/vp9_decodemv.c).
//
// A VP9 block's mode info determines:
//   - Segment ID (when segmentation map update is enabled)
//   - Skip flag (no residual)
//   - Transform size (when TxModeSelect)
//   - Whether the block is intra or inter (inter frames only)
//   - Intra: Y/UV prediction modes
//   - Inter: reference frames, inter prediction mode, MV(s), interp filter
//
// All reads come from the bool coder (boolcoder.Reader) using the adaptive
// probability tables in compheader.Probs and the above/left context arrays
// from the ctx package.
package blockmode

import (
	"fmt"
	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/compheader"
	"github.com/aethiopicuschan/go-vp9/ctx"
	"github.com/aethiopicuschan/go-vp9/modeinfo"
	"github.com/aethiopicuschan/go-vp9/mv"
	"github.com/aethiopicuschan/go-vp9/segment"
	"io"
)

// DbgW: when non-nil, traces inter mode-info bool-coder range at mi(0,0).
var DbgW io.Writer

// Inter prediction mode constants (spec §8.3.3).
const (
	NearestMV = 0
	NearMV    = 1
	ZeroMV    = 2
	NewMV     = 3
)

// interModeTree is the VP9 inter-mode binary tree (spec §7.5.4, Table 9).
// Decodes NEARESTMV(0), NEARMV(1), ZEROMV(2), NEWMV(3) using 3 probs.
var interModeTree = []int8{
	-int8(ZeroMV), 2,
	-int8(NearestMV), 4,
	-int8(NearMV), -int8(NewMV),
}

// FrameInfo holds frame-level parameters needed for block mode decode.
// Populated by the frame decoder (M12) from frame.Header and compheader.Probs.
type FrameInfo struct {
	IsKeyframe    bool
	AllowHP       bool
	ReferenceMode int    // compheader.RefModeSingle / RefModeCompound / RefModeSelect
	InterpFilter  int    // frame-level filter; compheader.InterpSwitchable (4) = per-block
	CompFixedRef  int    // compound "fixed" reference frame (ctx.RefLast/Golden/AltRef)
	CompVarRef    [2]int // compound "variable" references
	CompFixedIdx  int    // ref_frame slot for the fixed ref = sign_bias[CompFixedRef]

	// Neighbour availability bounds (4×4 units). Left is tile-relative (libvpx
	// left_mi: mi_col > tile->mi_col_start); above is frame-relative (above_mi:
	// mi_row != 0), so the decoder keeps MiRowStart at 0 for every tile row.
	MiColStart int
	MiRowStart int

	// Segmentation state. SegEnabled/SegUpdateMap/SegTempUpdate and the
	// tree/pred probs are this frame's header values; the feature data is the
	// persistent state (it survives frames that do not recode it).
	SegEnabled     bool
	SegUpdateMap   bool
	SegTempUpdate  bool
	SegTreeProbs   [7]uint8
	SegPredProbs   [3]uint8
	SegFeatEnabled [8][4]bool
	SegFeatData    [8][4]int16
	SegAbsOrDelta  bool

	// FrameIsIntra: key frame or intra-only frame (libvpx frame_is_intra_only);
	// selects read_intra_segment_id over read_inter_segment_id.
	FrameIsIntra bool

	// Segmentation maps in 8×8 (MI) units, row-major with stride SegMiCols
	// (libvpx cm->last_frame_seg_map / cm->current_frame_seg_map). SegMapCur is
	// written by Decode; SegPredFlags holds this frame's per-MI seg_id_predicted
	// bits. All nil when the caller keeps no maps (unit tests): the predicted
	// segment id is then 0 and nothing is stored.
	SegMapPrev   []uint8
	SegMapCur    []uint8
	SegPredFlags []bool
	SegMiCols    int
	SegMiRows    int
}

// segFeature reports whether feature f is active for segment id (libvpx
// segfeature_active: requires segmentation to be enabled on this frame).
func (fi *FrameInfo) segFeature(id, f int) bool {
	return fi.SegEnabled && fi.SegFeatEnabled[id][f]
}

// readSegmentID decodes (or infers) the block's segment id and maintains the
// current segmentation map (libvpx read_intra_segment_id /
// read_inter_segment_id). The map area is the block clipped to the frame.
func readSegmentID(br *boolcoder.Reader, fi *FrameInfo, row4, col4, bsW4, bsH4 int) int {
	if !fi.SegEnabled {
		return 0
	}
	row8, col8 := row4>>1, col4>>1
	xMis, yMis := 0, 0
	haveMap := fi.SegMapCur != nil && row8 < fi.SegMiRows && col8 < fi.SegMiCols
	if haveMap {
		xMis, yMis = maxInt(bsW4>>1, 1), maxInt(bsH4>>1, 1)
		xMis = minInt(xMis, fi.SegMiCols-col8)
		yMis = minInt(yMis, fi.SegMiRows-row8)
	}
	off := row8*fi.SegMiCols + col8

	// dec_get_segment_id: minimum over the block in the previous map.
	predicted := 0
	if haveMap && fi.SegMapPrev != nil {
		predicted = 8
		for y := 0; y < yMis; y++ {
			for x := 0; x < xMis; x++ {
				if v := int(fi.SegMapPrev[off+y*fi.SegMiCols+x]); v < predicted {
					predicted = v
				}
			}
		}
		predicted &= 7
	}

	if !fi.SegUpdateMap {
		// copy_segment_id: carry the previous map over.
		for y := 0; y < yMis; y++ {
			for x := 0; x < xMis; x++ {
				var v uint8
				if fi.SegMapPrev != nil {
					v = fi.SegMapPrev[off+y*fi.SegMiCols+x]
				}
				fi.SegMapCur[off+y*fi.SegMiCols+x] = v
			}
		}
		if fi.FrameIsIntra {
			return 0 // read_intra_segment_id returns 0 here, not the map value
		}
		return predicted
	}

	var id int
	if !fi.FrameIsIntra && fi.SegTempUpdate {
		// vp9_get_pred_context_seg_id: above/left seg_id_predicted flags. The
		// above neighbour is available on any row but the frame's first; the
		// left one only inside the tile.
		c := 0
		if haveMap && fi.SegPredFlags != nil {
			if row8 > 0 && fi.SegPredFlags[off-fi.SegMiCols] {
				c++
			}
			if col4 > fi.MiColStart && fi.SegPredFlags[off-1] {
				c++
			}
		}
		pred := br.ReadBool(fi.SegPredProbs[c]) == 1
		if haveMap && fi.SegPredFlags != nil {
			// libvpx stores the flag in the block's single MODE_INFO, which
			// every MI position of the block points at.
			for y := 0; y < yMis; y++ {
				for x := 0; x < xMis; x++ {
					fi.SegPredFlags[off+y*fi.SegMiCols+x] = pred
				}
			}
		}
		if pred {
			id = predicted
		} else {
			id = segment.DecodeSegID(br, fi.SegTreeProbs)
		}
	} else {
		id = segment.DecodeSegID(br, fi.SegTreeProbs)
	}
	// set_segment_id
	for y := 0; y < yMis; y++ {
		for x := 0; x < xMis; x++ {
			fi.SegMapCur[off+y*fi.SegMiCols+x] = uint8(id)
		}
	}
	return id
}

// CompFixedVarRef computes the compound fixed and variable reference frames
// from reference frame sign biases (frame.Header.RefFrameSign).
// sign[0]=LAST, sign[1]=GOLDEN, sign[2]=ALTREF.
func CompFixedVarRef(sign [3]bool) (fixed int, varRef [2]int) {
	last, golden, altref := sign[0], sign[1], sign[2]
	switch {
	case last == golden:
		// ALTREF has opposite bias → it's the fixed ref
		return ctx.RefAltRef, [2]int{ctx.RefLast, ctx.RefGolden}
	case last == altref:
		return ctx.RefGolden, [2]int{ctx.RefLast, ctx.RefAltRef}
	default:
		return ctx.RefLast, [2]int{ctx.RefGolden, ctx.RefAltRef}
	}
}

// BlockMode holds all decoded mode information for one VP9 coding block.
type BlockMode struct {
	SegID  int
	Skip   bool
	TxSize uint8 // ctx.TX4x4..TX32x32

	IsInter bool

	// Intra fields (valid when !IsInter)
	YMode  modeinfo.IntraMode // primary Y mode (also SubYModes[0] when NumSubY>0)
	UVMode modeinfo.IntraMode

	// Sub-8×8 intra: per-4×4 Y modes (used when NumSubY > 1).
	// Layout for bsW4=2,bsH4=1 (8×4): [left4x4, right4x4]
	// Layout for bsW4=1,bsH4=2 (4×8): [top4x4, bottom4x4]
	// NumSubY=0 or 1 means use YMode for the whole block.
	SubYModes [4]modeinfo.IntraMode
	NumSubY   int

	// Inter fields (valid when IsInter)
	RefFrame     [2]int   // [0]=primary, [1]=compound ref (ctx.RefIntra if single)
	InterMode    int      // NearestMV..NewMV
	MV           [2]mv.MV // [0]=primary MV, [1]=compound MV (valid only for NewMV)
	InterpFilter int      // 0=EIGHTTAP 1=SMOOTH 2=SHARP 3=BILINEAR

	// Sub-8×8 inter: per-4×4 MV overrides (NumSubMV > 0 when active).
	SubMV    [4][2]mv.MV
	NumSubMV int

	// Invalid: the bitstream used the segment SKIP feature on a sub-8×8 inter
	// block, which libvpx rejects ("Invalid usage of segment feature on small
	// blocks"). The caller must fail the frame.
	Invalid bool

	// InvalidMV: a NEWMV motion vector came out of the range libvpx accepts
	// (is_mv_valid: MV_LOW < component < MV_UPP, i.e. -2^14 < c < 2^14-1).
	// libvpx marks the frame corrupt (xd->corrupted) and fails it once its
	// tiles are decoded; the caller must do the same.
	InvalidMV bool
}

// mvValid is libvpx is_mv_valid (MV_LOW < component < MV_UPP).
func mvValid(m mv.MV) bool {
	const low, upp = -(1 << 14), 1<<14 - 1
	return int(m.Row) > low && int(m.Row) < upp && int(m.Col) > low && int(m.Col) < upp
}

// Sub-8×8 partition codes for Decode's sub8x8 parameter. They match
// partition.Type values (None/Horz/Vert/Split) and apply only to an 8×8
// coding block, selecting its sub-prediction layout (spec §6.4.20):
//
//	Sub8x8None  → BLOCK_8X8: 1 prediction (regular block).
//	Sub8x8Horz  → BLOCK_8X4: top/bottom 8×4, 2 sub-modes.
//	Sub8x8Vert  → BLOCK_4X8: left/right 4×8, 2 sub-modes.
//	Sub8x8Split → BLOCK_4X4: 2×2 of 4×4, 4 sub-modes.
const (
	Sub8x8None  = 0
	Sub8x8Horz  = 1
	Sub8x8Vert  = 2
	Sub8x8Split = 3
)

// Decode decodes one block's mode info from the bool coder.
//
//   - col4, row4: 4×4-block position within the tile
//   - bsW4, bsH4: block width and height in 4×4 units (e.g. 2,2 = 8×8)
//   - sub8x8: for an 8×8 coding block, its sub-prediction layout (Sub8x8*);
//     pass Sub8x8None for all other block sizes.
//
// counts is optional (nil disables probability adaptation accumulation).
// The caller must update aboveRow and leftCol after this call using
// ctx.Row.SetBlock / ctx.Col.SetBlock with the returned BlockMode fields.
// mvRefFn runs the MV-reference search for a reference frame and returns the
// inter-mode probability context and the two best (clamped, precision-lowered)
// candidate MVs (libvpx vp9_find_mv_refs + vp9_find_best_ref_mvs). Pass nil to
// fall back to the simplified ctx.InterModeCtx heuristic with no candidates
// (used by unit tests).
// appendFn resolves the per-sub-block NEAREST/NEAR candidate MV for a sub-8×8
// prediction unit (libvpx append_sub8x8_mvs_for_idx). Pass nil to fall back to
// the block-level candidate list (unit tests).
func Decode(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts, fi FrameInfo,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	col4, row4, bsW4, bsH4, sub8x8 int,
	mvRefFn func(refFrame, row4, col4, bsW4, bsH4 int) (int, [2]mv.MV),
	appendFn func(refFrame, ref, row4, col4, bsW4, bsH4, blockIdx, bMode int, bmi [4][2]mv.MV) mv.MV) BlockMode {

	// Derive bsLog2 from the smaller dimension (used for TX size cap and mode group).
	bsLog2 := log2n4(minInt(bsW4, bsH4))

	hasAbove := row4 > fi.MiRowStart
	hasLeft := col4 > fi.MiColStart

	var aboveTx, leftTx uint8
	var aboveSkip, leftSkip bool
	var aboveRef, leftRef [2]int
	var aboveInter, leftInter bool
	var aboveZeroMV, leftZeroMV bool
	var aboveYMode, leftYMode uint8
	var aboveInterp, leftInterp uint8

	if hasAbove && col4 < aboveRow.N4 {
		aboveTx = aboveRow.TxSize[col4]
		aboveSkip = aboveRow.Skip[col4]
		aboveRef = aboveRow.Ref[col4]
		aboveInter = aboveRow.Inter[col4]
		aboveZeroMV = aboveRow.ZeroMV[col4]
		aboveYMode = aboveRow.YMode[col4]
		aboveInterp = aboveRow.InterpFilter[col4]
	} else {
		hasAbove = false
		aboveRef = [2]int{ctx.RefIntra, ctx.RefIntra}
	}
	if hasLeft && row4 < leftCol.N4 {
		leftTx = leftCol.TxSize[row4]
		leftSkip = leftCol.Skip[row4]
		leftRef = leftCol.Ref[row4]
		leftInter = leftCol.Inter[row4]
		leftZeroMV = leftCol.ZeroMV[row4]
		leftYMode = leftCol.YMode[row4]
		leftInterp = leftCol.InterpFilter[row4]
	} else {
		hasLeft = false
		leftRef = [2]int{ctx.RefIntra, ctx.RefIntra}
	}

	var bm BlockMode

	// 1. Segment ID (read, temporally predicted, or carried over from the
	// previous segmentation map).
	bm.SegID = readSegmentID(br, &fi, row4, col4, bsW4, bsH4)
	segSkip := fi.segFeature(bm.SegID, segment.FeatSkip)
	segRef := fi.segFeature(bm.SegID, segment.FeatRefFrame)

	// 2. Skip flag. The segment SKIP feature implies skip with no bit coded
	// (libvpx read_skip).
	skipCtx := ctx.SkipCtx(aboveSkip, leftSkip)
	if segSkip {
		bm.Skip = true
	} else {
		skipBit := br.ReadBool(probs.SkipProbs[skipCtx])
		if counts != nil {
			counts.SkipProbs[skipCtx][skipBit]++
		}
		bm.Skip = skipBit == 1
	}

	// 3. is_inter (inter frames only). Read BEFORE tx_size — VP9 §6.4.7
	// inter_frame_mode_info: read_is_inter() precedes read_tx_size(), and
	// tx_size's allow_select depends on the is_inter result.
	// With the segment REF_FRAME feature the flag is implied by the segment's
	// reference frame and nothing is coded (libvpx read_is_inter_block).
	if !fi.IsKeyframe {
		if segRef {
			bm.IsInter = fi.SegFeatData[bm.SegID][segment.FeatRefFrame] != ctx.RefIntra
		} else {
			isInterCtx := ctx.IsInterCtx(hasAbove, hasLeft, aboveInter, leftInter)
			isInterBit := br.ReadBool(probs.IsInterProbs[isInterCtx])
			if counts != nil {
				counts.IsInterProbs[isInterCtx][isInterBit]++
			}
			bm.IsInter = isInterBit == 1
		}
	}
	if DbgW != nil && ((row4 == 0 && col4 == 0) || (row4 == 6 && col4 == 2) || (row4 == 6 && col4 == 10) || (row4 == 4 && col4 == 10) || (row4 == 4 && col4 == 12)) {
		rng, top := br.State()
		fmt.Fprintf(DbgW, "IB go afterIsInter pos=%d,%d rng=%d t=%d isInter=%v skip=%v skipCtx=%d isInterCtx=%d\n", row4, col4, rng, top, bm.IsInter, bm.Skip, skipCtx, ctx.IsInterCtx(hasAbove, hasLeft, aboveInter, leftInter))
	}

	// 4. TX size. Sub-8×8 blocks (BLOCK_4X4/4X8/8X4) always use TX_4X4 and read
	// no tx_size element (spec §6.4.21: max_tx_size is TX_4X4).
	if sub8x8 != Sub8x8None {
		bm.TxSize = ctx.TX4x4
	} else {
		maxTx := uint8(bsLog2)
		if maxTx > ctx.TX32x32 {
			maxTx = ctx.TX32x32
		}
		// allow_select = !skip || !is_inter (libvpx read_tx_size): an intra block
		// always reads tx_size; an inter block only when not skipped. On a
		// keyframe is_inter is unset (false), so allow_select is always true.
		allowSelect := !bm.Skip || !bm.IsInter
		if allowSelect && probs.TxMode == compheader.TxModeSelect && maxTx > 0 {
			bm.TxSize = readTxSize(br, probs, counts, maxTx,
				ctx.TxSizeCtx(hasAbove, hasLeft, aboveSkip, leftSkip, aboveTx, leftTx, maxTx))
		} else {
			bm.TxSize = maxTx
			if probs.TxMode != compheader.TxModeSelect {
				cap := txModeMax(probs.TxMode)
				if bm.TxSize > cap {
					bm.TxSize = cap
				}
			}
		}
	}

	if DbgW != nil && ((row4 == 0 && col4 == 0) || (row4 == 6 && col4 == 2) || (row4 == 6 && col4 == 10) || (row4 == 4 && col4 == 10)) {
		rng, _ := br.State()
		fmt.Fprintf(DbgW, "IB go afterTx rng=%d tx=%d\n", rng, bm.TxSize)
	}

	// isSub8x8: an 8×8 coding block split into 4×4/4×8/8×4 prediction units.
	isSub8x8 := sub8x8 != Sub8x8None

	if !bm.IsInter {
		// 5. Intra block.
		sizeGroup := bsLog2
		if sizeGroup > 3 {
			sizeGroup = 3
		}
		if DbgW != nil && row4 == 4 && col4 == 12 {
			fmt.Fprintf(DbgW, "II go intra entry isSub8x8=%v sub8x8=%d bsLog2=%d bp=%d\n", isSub8x8, sub8x8, bsLog2, br.BitPos())
		}
		if isSub8x8 {
			// Sub-8×8 intra (spec §6.4.20): read modes over the 8×8 cell as a 2×2
			// grid of 4×4 prediction units, stepping by the sub-block size. Each
			// sub-mode's KF context uses the neighbouring sub-block modes within
			// the cell (or the block's above/left neighbour at the cell edge).
			num4x4w, num4x4h := sub8x8Dims(sub8x8)
			var yC [][2]uint32
			if counts != nil {
				yC = counts.YModeProbs[0][:]
			}
			// Per-position neighbour modes for the cell edges: the above context at
			// column idx reads the above neighbour's bottom sub-mode for that column
			// (aboveRow.YMode[col4+idx]); the left context at row idy reads the left
			// neighbour's right sub-mode for that row (leftCol.YMode[row4+idy]).
			aboveAt := func(idx int) int {
				if row4 > 0 && col4+idx < aboveRow.N4 {
					return int(aboveRow.YMode[col4+idx])
				}
				return 0 // DC_PRED
			}
			leftAt := func(idy int) int {
				if col4 > 0 && row4+idy < leftCol.N4 {
					return int(leftCol.YMode[row4+idy])
				}
				return 0 // DC_PRED
			}
			var sub [4]modeinfo.IntraMode // grid index = idy*2 + idx
			for idy := 0; idy < 2; idy += num4x4h {
				for idx := 0; idx < 2; idx += num4x4w {
					am := aboveAt(idx)
					if idy > 0 {
						am = int(sub[(idy-1)*2+idx])
					}
					lm := leftAt(idy)
					if idx > 0 {
						lm = int(sub[idy*2+idx-1])
					}
					var mode modeinfo.IntraMode
					if fi.IsKeyframe {
						mode = modeinfo.ReadYModeKF(br, am, lm, nil)
					} else {
						if DbgW != nil && row4 == 4 && col4 == 12 {
							rb, tb := br.State()
							fmt.Fprintf(DbgW, "IIY go sub(%d,%d) beforeY rng=%d t=%d yprob0=[%d %d %d ..] bp=%d\n", idy, idx, rb, tb, probs.YModeProbs[0][0], probs.YModeProbs[0][1], probs.YModeProbs[0][2], br.BitPos())
						}
						mode = modeinfo.ReadYMode(br, 0, probs.YModeProbs, yC)
						if DbgW != nil && row4 == 4 && col4 == 12 {
							rb, tb := br.State()
							fmt.Fprintf(DbgW, "IIY go sub(%d,%d) afterY mode=%d rng=%d t=%d bp=%d\n", idy, idx, mode, rb, tb, br.BitPos())
						}
					}
					for y2 := 0; y2 < num4x4h; y2++ {
						for x2 := 0; x2 < num4x4w; x2++ {
							sub[(idy+y2)*2+(idx+x2)] = mode
						}
					}
				}
			}
			bm.SubYModes = sub
			bm.NumSubY = 4
			bm.YMode = sub[3] // bottom-right sub-mode drives UV + neighbour context
			if DbgW != nil && row4 == 4 && col4 == 12 {
				fmt.Fprintf(DbgW, "II go subY=[%d %d %d %d] afterY bp=%d\n", sub[0], sub[1], sub[2], sub[3], br.BitPos())
			}
		} else {
			var yC [][2]uint32
			if counts != nil {
				yC = counts.YModeProbs[sizeGroup][:]
			}
			if fi.IsKeyframe {
				bm.YMode = modeinfo.ReadYModeKF(br, int(aboveYMode), int(leftYMode), nil)
			} else {
				bm.YMode = modeinfo.ReadYMode(br, sizeGroup, probs.YModeProbs, yC)
			}
		}
		var uvC [][2]uint32
		if counts != nil {
			uvC = counts.UVModeProbs[int(bm.YMode)][:]
		}
		if fi.IsKeyframe {
			bm.UVMode = modeinfo.ReadUVModeKF(br, bm.YMode, nil)
		} else {
			bm.UVMode = modeinfo.ReadUVMode(br, bm.YMode, probs.UVModeProbs, uvC)
		}
		bm.RefFrame = [2]int{ctx.RefIntra, ctx.RefIntra}
		bm.InterpFilter = ctx.InterpSwitchable
	} else {
		// 6. Inter block. Read order (VP9 §6.4.22 read_inter_block_mode_info):
		// ref_frames → inter_mode (bsize≥8×8) → interp_filter → MV. For sub-8×8
		// blocks the per-unit inter modes are read AFTER interp_filter.
		if segRef {
			// Segment REF_FRAME feature: single reference, nothing coded
			// (libvpx read_ref_frames).
			bm.RefFrame = [2]int{int(fi.SegFeatData[bm.SegID][segment.FeatRefFrame]), ctx.RefIntra}
		} else {
			bm.RefFrame, bm.IsInter = readRefFrames(br, probs, counts, fi,
				hasAbove, hasLeft, aboveRef, leftRef, aboveInter, leftInter)
		}

		var imCtx int
		var mvList0, mvList1 [2]mv.MV
		if mvRefFn != nil {
			imCtx, mvList0 = mvRefFn(bm.RefFrame[0], row4, col4, bsW4, bsH4)
			if bm.RefFrame[1] > ctx.RefIntra {
				_, mvList1 = mvRefFn(bm.RefFrame[1], row4, col4, bsW4, bsH4)
			}
		} else {
			imCtx = ctx.InterModeCtx(hasAbove, hasLeft, aboveRef, leftRef,
				aboveInter, leftInter, aboveZeroMV, leftZeroMV, bm.RefFrame[0])
		}
		var imC [][2]uint32
		if counts != nil {
			imC = counts.InterModeProbs[imCtx][:]
		}
		if DbgW != nil && ((row4 == 0 && col4 == 0) || (row4 == 6 && col4 == 2) || (row4 == 6 && col4 == 10) || (row4 == 4 && col4 == 10)) {
			rng, _ := br.State()
			fmt.Fprintf(DbgW, "IB go afterRef ctx=%d rng=%d bp=%d ref=%d,%d refMode=%d singleP1ctx=%d singleP1prob=%d\n",
				imCtx, rng, br.BitPos(), bm.RefFrame[0], bm.RefFrame[1], fi.ReferenceMode,
				SingleRefCtxP1(hasAbove, hasLeft, aboveRef, leftRef, aboveInter, leftInter),
				probs.SingleRefProbs[SingleRefCtxP1(hasAbove, hasLeft, aboveRef, leftRef, aboveInter, leftInter)][0])
		}

		// 7. Inter prediction mode for whole-block (≥8×8), BEFORE interp_filter.
		// The segment SKIP feature forces ZEROMV with no mode coded (and is
		// not allowed on sub-8×8 blocks).
		if segSkip {
			bm.InterMode = ZeroMV
			if isSub8x8 {
				bm.Invalid = true
				return bm
			}
		} else if !isSub8x8 {
			bm.InterMode = int(br.ReadTreeCount(interModeTree, probs.InterModeProbs[imCtx][:], imC))
		}
		if DbgW != nil && ((row4 == 0 && col4 == 0) || (row4 == 6 && col4 == 2) || (row4 == 6 && col4 == 10) || (row4 == 4 && col4 == 10)) {
			rng, _ := br.State()
			fmt.Fprintf(DbgW, "IB go afterMode rng=%d mode=%d\n", rng, bm.InterMode)
		}

		// 8. Interpolation filter.
		if fi.InterpFilter == compheader.InterpSwitchable {
			aboveHasInter := hasAbove && aboveInter
			leftHasInter := hasLeft && leftInter
			ifCtx := ctx.InterpFilterCtx(aboveHasInter, leftHasInter, aboveInterp, leftInterp)
			bm.InterpFilter = readInterpFilter(br, probs, counts, ifCtx)
		} else {
			bm.InterpFilter = fi.InterpFilter
		}
		if DbgW != nil && ((row4 == 0 && col4 == 0) || (row4 == 6 && col4 == 2) || (row4 == 6 && col4 == 10) || (row4 == 4 && col4 == 10)) {
			rng, _ := br.State()
			{
				rngB := br
				_ = rngB
				fmt.Fprintf(DbgW, "IB go afterInterp rng=%d filt=%d bp=%d\n", rng, bm.InterpFilter, br.BitPos())
			}
		}

		if isSub8x8 {
			// Sub-8×8 inter (spec §6.4.24): per prediction-unit inter mode + MV,
			// filling the 2×2 grid by stepping with the sub-block size. Ref frame
			// is shared across the cell.
			num4x4w, num4x4h := sub8x8Dims(sub8x8)
			compound := bm.RefFrame[1] != ctx.RefIntra
			lastSubMode := NearestMV
			var bmi [4][2]mv.MV // resolved per-sub MVs so far (libvpx mi->bmi)
		sub8x8Loop:
			for idy := 0; idy < 2; idy += num4x4h {
				for idx := 0; idx < 2; idx += num4x4w {
					j := idy*2 + idx
					subMode := int(br.ReadTreeCount(interModeTree, probs.InterModeProbs[imCtx][:], imC))
					lastSubMode = subMode
					var smv [2]mv.MV
					// libvpx assign_mv for a sub-8×8 unit: NEWMV uses the BLOCK-LEVEL
					// best ref MV (mvList[0]) for both use_mv_hp and the difference base;
					// NEAREST/NEAR use the per-sub append_sub8x8 candidate.
					switch subMode {
					case NewMV:
						diff0 := mv.ReadMV(br, probs, counts, fi.AllowHP && mv.UseHP(mvList0[0]))
						smv[0] = mv.MV{Row: mvList0[0].Row + diff0.Row, Col: mvList0[0].Col + diff0.Col}
						if compound {
							diff1 := mv.ReadMV(br, probs, counts, fi.AllowHP && mv.UseHP(mvList1[0]))
							smv[1] = mv.MV{Row: mvList1[0].Row + diff1.Row, Col: mvList1[0].Col + diff1.Col}
						}
						if !mvValid(smv[0]) || !mvValid(smv[1]) {
							// libvpx: xd->corrupted |= 1; return (the remaining
							// sub-blocks are not read).
							bm.InvalidMV = true
							break sub8x8Loop
						}
					case NearestMV, NearMV:
						if appendFn != nil {
							smv[0] = appendFn(bm.RefFrame[0], 0, row4, col4, bsW4, bsH4, j, subMode, bmi)
							if compound {
								smv[1] = appendFn(bm.RefFrame[1], 1, row4, col4, bsW4, bsH4, j, subMode, bmi)
							}
						} else if subMode == NearestMV {
							smv[0], smv[1] = mvList0[0], mvList1[0]
						} else {
							smv[0], smv[1] = mvList0[1], mvList1[1]
						}
					}
					// case ZeroMV: smv stays {0,0}.
					for y2 := 0; y2 < num4x4h; y2++ {
						for x2 := 0; x2 < num4x4w; x2++ {
							pos := (idy+y2)*2 + (idx + x2)
							bm.SubMV[pos] = smv
							bmi[pos] = smv
						}
					}
				}
			}
			bm.NumSubMV = 4
			// libvpx stores the whole-block mode/MV as the LAST sub-block's
			// (mi->mode = b_mode; copy_mv_pair(mi->mv, bmi[3])) — this feeds the
			// mode-context (mode_2_counter) and MV-ref search of later blocks.
			bm.InterMode = lastSubMode
			bm.MV = bm.SubMV[3]
		} else {
			// 9. Resolve the whole-block MV(s) from the candidate list (libvpx
			// assign_mv). NEAREST→mvList[0], NEAR→mvList[1], ZERO→0; NEWMV reads a
			// difference (1/8-pel gated by use_mv_hp on the best candidate) and adds
			// it to mvList[0]. bm.MV is the final resolved MV (fed to miGrid + recon).
			compound := bm.RefFrame[1] != ctx.RefIntra
			bm.MV[0] = resolveInterMV(br, probs, counts, bm.InterMode, mvList0, fi.AllowHP)
			if compound {
				bm.MV[1] = resolveInterMV(br, probs, counts, bm.InterMode, mvList1, fi.AllowHP)
			}
			if bm.InterMode == NewMV && (!mvValid(bm.MV[0]) || !mvValid(bm.MV[1])) {
				bm.InvalidMV = true // libvpx: xd->corrupted |= !assign_mv(...)
			}
		}
	}

	return bm
}

// resolveInterMV resolves the final MV for one reference of a whole-block
// (≥8×8) inter prediction from its candidate list (libvpx assign_mv). For NEWMV
// a difference is read (high precision gated by use_mv_hp on the best/nearest
// candidate) and added to mvList[0]; other modes select a candidate directly.
func resolveInterMV(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	mode int, mvList [2]mv.MV, allowHP bool) mv.MV {
	switch mode {
	case NewMV:
		diff := mv.ReadMV(br, probs, counts, allowHP && mv.UseHP(mvList[0]))
		return mv.MV{Row: mvList[0].Row + diff.Row, Col: mvList[0].Col + diff.Col}
	case NearMV:
		return mvList[1]
	case ZeroMV:
		return mv.MV{}
	default: // NearestMV
		return mvList[0]
	}
}

// ---- helpers ----------------------------------------------------------------

func readTxSize(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts, maxTx uint8, txCtx int) uint8 {
	switch maxTx {
	case ctx.TX8x8:
		bit := br.ReadBool(probs.TxProbs8[txCtx])
		if counts != nil {
			counts.TxProbs8[txCtx][bit]++
		}
		return uint8(bit)
	case ctx.TX16x16:
		b0 := br.ReadBool(probs.TxProbs16[txCtx][0])
		if counts != nil {
			counts.TxProbs16[txCtx][0][b0]++
		}
		tx := uint8(b0)
		if tx == 1 {
			b1 := br.ReadBool(probs.TxProbs16[txCtx][1])
			if counts != nil {
				counts.TxProbs16[txCtx][1][b1]++
			}
			tx += uint8(b1)
		}
		return tx
	default: // TX32x32
		b0 := br.ReadBool(probs.TxProbs32[txCtx][0])
		if counts != nil {
			counts.TxProbs32[txCtx][0][b0]++
		}
		tx := uint8(b0)
		if tx == 1 {
			b1 := br.ReadBool(probs.TxProbs32[txCtx][1])
			if counts != nil {
				counts.TxProbs32[txCtx][1][b1]++
			}
			tx += uint8(b1)
			if tx == 2 {
				b2 := br.ReadBool(probs.TxProbs32[txCtx][2])
				if counts != nil {
					counts.TxProbs32[txCtx][2][b2]++
				}
				tx += uint8(b2)
			}
		}
		return tx
	}
}

func txModeMax(txMode int) uint8 {
	switch txMode {
	case compheader.TxModeOnly4x4:
		return ctx.TX4x4
	case compheader.TxModeAllow8x8:
		return ctx.TX8x8
	case compheader.TxModeAllow16x16:
		return ctx.TX16x16
	default:
		return ctx.TX32x32
	}
}

func readRefFrames(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts, fi FrameInfo,
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) ([2]int, bool) {

	refMode := fi.ReferenceMode

	if refMode == compheader.RefModeSelect {
		ctx2 := CompModeCtx(fi, hasAbove, hasLeft, aboveRef, leftRef, aboveInter, leftInter)
		bit := br.ReadBool(probs.CompInterProbs[ctx2])
		if counts != nil {
			counts.CompInterProbs[ctx2][bit]++
		}
		if bit == 1 {
			refMode = compheader.RefModeCompound
		} else {
			refMode = compheader.RefModeSingle
		}
	}

	if refMode == compheader.RefModeCompound {
		ctx2 := CompRefCtx(fi, hasAbove, hasLeft, aboveRef, leftRef, aboveInter, leftInter)
		bit := br.ReadBool(probs.CompRefProbs[ctx2])
		if counts != nil {
			counts.CompRefProbs[ctx2][bit]++
		}
		// libvpx read_ref_frames: ref_frame[sign_bias[fixed]] = fixed,
		// ref_frame[!idx] = var[bit]. Storing them in the wrong slots corrupts
		// neighbour reference contexts for later compound blocks.
		var refs [2]int
		refs[fi.CompFixedIdx] = fi.CompFixedRef
		refs[1-fi.CompFixedIdx] = fi.CompVarRef[bit]
		return refs, true
	}

	// Single reference.
	ctx0 := SingleRefCtxP1(hasAbove, hasLeft, aboveRef, leftRef, aboveInter, leftInter)
	bit0 := br.ReadBool(probs.SingleRefProbs[ctx0][0])
	if counts != nil {
		counts.SingleRefProbs[ctx0][0][bit0]++
	}
	if bit0 == 0 {
		return [2]int{ctx.RefLast, ctx.RefIntra}, true
	}
	ctx1 := SingleRefCtxP2(hasAbove, hasLeft, aboveRef, leftRef, aboveInter, leftInter)
	bit1 := br.ReadBool(probs.SingleRefProbs[ctx1][1])
	if counts != nil {
		counts.SingleRefProbs[ctx1][1][bit1]++
	}
	if bit1 == 0 {
		return [2]int{ctx.RefGolden, ctx.RefIntra}, true
	}
	return [2]int{ctx.RefAltRef, ctx.RefIntra}, true
}

// sub8x8Dims returns the prediction sub-block size in 4×4 units (num4x4w,
// num4x4h) for a sub-8×8 partition code (Sub8x8Horz/Vert/Split).
func sub8x8Dims(sub8x8 int) (num4x4w, num4x4h int) {
	switch sub8x8 {
	case Sub8x8Horz: // BLOCK_8X4: 8 wide, 4 tall
		return 2, 1
	case Sub8x8Vert: // BLOCK_4X8: 4 wide, 8 tall
		return 1, 2
	default: // Sub8x8Split → BLOCK_4X4
		return 1, 1
	}
}

func log2n4(n int) int {
	l := 0
	for n > 1 {
		l++
		n >>= 1
	}
	return l
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func readInterpFilter(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts, ifCtx int) int {
	bit0 := br.ReadBool(probs.InterpFilterProbs[ifCtx][0])
	if counts != nil {
		counts.InterpFilterProbs[ifCtx][0][bit0]++
	}
	if bit0 == 0 {
		return compheader.InterpEightTap
	}
	bit1 := br.ReadBool(probs.InterpFilterProbs[ifCtx][1])
	if counts != nil {
		counts.InterpFilterProbs[ifCtx][1][bit1]++
	}
	if bit1 == 0 {
		return compheader.InterpEightTapSmooth
	}
	return compheader.InterpEightTapSharp
}
