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
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
	"github.com/mgvs/go-vp9/ctx"
	"github.com/mgvs/go-vp9/modeinfo"
	"github.com/mgvs/go-vp9/mv"
	"github.com/mgvs/go-vp9/segment"
)

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
	IsKeyframe     bool
	AllowHP        bool
	ReferenceMode  int // compheader.RefModeSingle / RefModeCompound / RefModeSelect
	InterpFilter   int // frame-level filter; compheader.InterpSwitchable (4) = per-block
	CompFixedRef   int // compound "fixed" reference frame (ctx.RefLast/Golden/AltRef)
	CompVarRef     [2]int // compound "variable" references

	// Segmentation state from frame.Header.
	SegEnabled    bool
	SegUpdateMap  bool
	SegTreeProbs  [7]uint8
	SegFeatEnabled [8][4]bool
	SegFeatData    [8][4]int16
	SegAbsOrDelta bool
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
	RefFrame     [2]int    // [0]=primary, [1]=compound ref (ctx.RefIntra if single)
	InterMode    int       // NearestMV..NewMV
	MV           [2]mv.MV // [0]=primary MV, [1]=compound MV (valid only for NewMV)
	InterpFilter int       // 0=EIGHTTAP 1=SMOOTH 2=SHARP 3=BILINEAR

	// Sub-8×8 inter: per-4×4 MV overrides (NumSubMV > 0 when active).
	SubMV    [4][2]mv.MV
	NumSubMV int
}

// Decode decodes one block's mode info from the bool coder.
//
//   - col4, row4: 4×4-block position within the tile
//   - bsW4, bsH4: block width and height in 4×4 units (e.g. 2,2 = 8×8)
//
// counts is optional (nil disables probability adaptation accumulation).
// The caller must update aboveRow and leftCol after this call using
// ctx.Row.SetBlock / ctx.Col.SetBlock with the returned BlockMode fields.
func Decode(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts, fi FrameInfo,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	col4, row4, bsW4, bsH4 int) BlockMode {

	// Derive bsLog2 from the smaller dimension (used for TX size cap and mode group).
	bsLog2 := log2n4(minInt(bsW4, bsH4))

	hasAbove := row4 > 0
	hasLeft := col4 > 0

	var aboveTx, leftTx uint8
	var aboveSkip, leftSkip bool
	var aboveRef, leftRef [2]int
	var aboveInter, leftInter bool
	var aboveZeroMV, leftZeroMV bool
	var aboveYMode uint8
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
		leftInterp = leftCol.InterpFilter[row4]
	} else {
		hasLeft = false
		leftRef = [2]int{ctx.RefIntra, ctx.RefIntra}
	}
	_ = aboveYMode // used implicitly via aboveRef context

	var bm BlockMode

	// 1. Segment ID.
	if fi.SegEnabled && fi.SegUpdateMap {
		bm.SegID = segment.DecodeSegID(br, fi.SegTreeProbs)
	}

	// 2. Skip flag.
	skipCtx := ctx.SkipCtx(aboveSkip, leftSkip)
	skipBit := br.ReadBool(probs.SkipProbs[skipCtx])
	if counts != nil {
		counts.SkipProbs[skipCtx][skipBit]++
	}
	bm.Skip = skipBit == 1

	// 3. TX size (when not skip, TX mode is select, and block is large enough).
	maxTx := uint8(bsLog2)
	if maxTx > ctx.TX32x32 {
		maxTx = ctx.TX32x32
	}
	if !bm.Skip && probs.TxMode == compheader.TxModeSelect && maxTx > 0 {
		bm.TxSize = readTxSize(br, probs, counts, maxTx,
			ctx.TxSizeCtx(hasAbove, hasLeft, aboveTx, leftTx, maxTx))
	} else {
		bm.TxSize = maxTx
		if probs.TxMode != compheader.TxModeSelect {
			cap := txModeMax(probs.TxMode)
			if bm.TxSize > cap {
				bm.TxSize = cap
			}
		}
	}

	// 4. is_inter (inter frames only).
	if !fi.IsKeyframe {
		isInterCtx := ctx.IsInterCtx(hasAbove, hasLeft, aboveInter, leftInter)
		isInterBit := br.ReadBool(probs.IsInterProbs[isInterCtx])
		if counts != nil {
			counts.IsInterProbs[isInterCtx][isInterBit]++
		}
		bm.IsInter = isInterBit == 1
	}

	// isSub8x8: block has multiple 4×4 sub-positions and at least one dim is 4px.
	numSubPos := bsW4 * bsH4
	isSub8x8 := numSubPos > 1 && (bsW4 < 2 || bsH4 < 2)

	if !bm.IsInter {
		// 5. Intra block.
		sizeGroup := bsLog2
		if sizeGroup > 3 {
			sizeGroup = 3
		}
		if isSub8x8 {
			// Sub-8×8 intra: per-4×4 Y mode (raster: ty=0..bsH4-1, tx=0..bsW4-1).
			var yC [][2]uint32
			if counts != nil {
				yC = counts.YModeProbs[0][:]
			}
			for i := 0; i < numSubPos; i++ {
				bm.SubYModes[i] = modeinfo.ReadYMode(br, 0, probs.YModeProbs, yC)
			}
			bm.NumSubY = numSubPos
			bm.YMode = bm.SubYModes[0]
		} else {
			var yC [][2]uint32
			if counts != nil {
				yC = counts.YModeProbs[sizeGroup][:]
			}
			bm.YMode = modeinfo.ReadYMode(br, sizeGroup, probs.YModeProbs, yC)
		}
		var uvC [][2]uint32
		if counts != nil {
			uvC = counts.UVModeProbs[int(bm.YMode)][:]
		}
		bm.UVMode = modeinfo.ReadUVMode(br, bm.YMode, probs.UVModeProbs, uvC)
		bm.RefFrame = [2]int{ctx.RefIntra, ctx.RefIntra}
		bm.InterpFilter = ctx.InterpSwitchable
	} else {
		// 6. Inter block.
		bm.RefFrame, bm.IsInter = readRefFrames(br, probs, counts, fi,
			hasAbove, hasLeft, aboveRef, leftRef, aboveInter, leftInter)

		imCtx := ctx.InterModeCtx(hasAbove, hasLeft, aboveRef, leftRef,
			aboveInter, leftInter, aboveZeroMV, leftZeroMV, bm.RefFrame[0])

		// 8. Interpolation filter.
		if fi.InterpFilter == compheader.InterpSwitchable {
			aboveHasInter := hasAbove && aboveInter
			leftHasInter := hasLeft && leftInter
			ifCtx := ctx.InterpFilterCtx(aboveHasInter, leftHasInter, aboveInterp, leftInterp)
			bm.InterpFilter = readInterpFilter(br, probs, counts, ifCtx)
		} else {
			bm.InterpFilter = fi.InterpFilter
		}

		var imC [][2]uint32
		if counts != nil {
			imC = counts.InterModeProbs[imCtx][:]
		}
		if isSub8x8 {
			// Sub-8×8 inter: per-4×4 inter mode + MV; ref frame shared.
			for i := 0; i < numSubPos; i++ {
				subMode := int(br.ReadTreeCount(interModeTree, probs.InterModeProbs[imCtx][:], imC))
				if subMode == NewMV {
					bm.SubMV[i][0] = mv.ReadMV(br, probs, counts, fi.AllowHP)
					if bm.RefFrame[1] != ctx.RefIntra {
						bm.SubMV[i][1] = mv.ReadMV(br, probs, counts, fi.AllowHP)
					}
				}
			}
			bm.NumSubMV = numSubPos
			bm.InterMode = NearestMV // fallback for context updates
			bm.MV = bm.SubMV[0]
		} else {
			// 7. Inter prediction mode (NEAREST/NEAR/ZERO/NEW).
			bm.InterMode = int(br.ReadTreeCount(interModeTree, probs.InterModeProbs[imCtx][:], imC))
			// 9. MV(s) for NEWMV mode.
			if bm.InterMode == NewMV {
				bm.MV[0] = mv.ReadMV(br, probs, counts, fi.AllowHP)
				if bm.RefFrame[1] != ctx.RefIntra {
					bm.MV[1] = mv.ReadMV(br, probs, counts, fi.AllowHP)
				}
			}
		}
	}

	return bm
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
		return [2]int{fi.CompFixedRef, fi.CompVarRef[bit]}, true
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
