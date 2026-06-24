// Package decoder is the VP9 frame decoder — the final layer (M12) that wires
// together all lower-level packages into a complete decode pipeline.
//
// Pipeline for each frame:
//
//  1. Parse uncompressed header (frame.Parse).
//  2. Decode compressed header probability updates (compheader.Decode).
//  3. Allocate the output frame (refbuf.NewFrame).
//  4. For each tile: create above-row context, then iterate superblock rows.
//  5. For each SB: partition tree (recursive NONE/HORZ/VERT/SPLIT decode).
//  6. For each coding block: blockmode.Decode → MV resolution → reconstruct.Block.
//  7. After all tiles: apply loop filter (loopfilter.FilterPlane).
//  8. Update reference buffer (refbuf.Buffer.Update, ExtendBorders).
//
// Usage:
//
//	d := decoder.New()
//	for each IVF packet payload:
//	    frame, err := d.DecodeFrame(payload)
package decoder

import (
	"errors"

	"github.com/mgvs/go-vp9/blockmode"
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
	"github.com/mgvs/go-vp9/ctx"
	"github.com/mgvs/go-vp9/dequant"
	"github.com/mgvs/go-vp9/frame"
	"github.com/mgvs/go-vp9/loopfilter"
	"github.com/mgvs/go-vp9/mv"
	"github.com/mgvs/go-vp9/mvpred"
	"github.com/mgvs/go-vp9/partition"
	"github.com/mgvs/go-vp9/reconstruct"
	"github.com/mgvs/go-vp9/refbuf"
	"github.com/mgvs/go-vp9/segment"
	"github.com/mgvs/go-vp9/tile"
)

// miBlock stores per-MI-block decoded state needed for MV prediction and
// loop filter level lookup of subsequent blocks.
type miBlock struct {
	inter   bool
	ref     [2]int
	mv      [2]mv.MV
	zeroMV  bool
	lfLevel int // per-block loop filter level (0..63)
}

// Decoder holds VP9 decode state across frames.
type Decoder struct {
	probs  *compheader.Probs
	refBuf refbuf.Buffer

	// Per-frame quantizer deltas (stored for per-block segment Q recompute).
	baseQIdx   int
	deltaQYDC  int
	deltaQUVDC int
	deltaQUVAC int

	// Per-frame loop filter state (for per-block level computation).
	lfBaseLevel    int
	lfSharpness    int
	lfDeltaEnabled bool
	lfRefDeltas    [4]int
	lfModeDeltas   [2]int

	// Stored frame dimensions (updated on key frame and intra-only inter frame).
	frameWidth, frameHeight int

	// Per-frame MI grid (row-major, W4 × H4).
	miW4, miH4 int
	miGrid     []miBlock

	// Temporal MV buffers: per-slot saved MI grids from previous frames.
	prevMIW4    [8]int
	prevMIH4    [8]int
	prevMIGrids [8][]miBlock
}

// New creates a Decoder with default probability tables.
func New() *Decoder {
	d := &Decoder{
		probs: compheader.DefaultProbs(),
	}
	return d
}

// DecodeFrame decodes one VP9 frame from a raw IVF packet payload (not including
// the 12-byte IVF frame header).  Returns the decoded 4:2:0 8-bit frame.
func (d *Decoder) DecodeFrame(payload []byte) (*refbuf.Frame, error) {
	// 1. Uncompressed header.
	h, err := frame.Parse(payload, d.frameWidth, d.frameHeight)
	if err != nil {
		return nil, err
	}
	if h.ShowExistingFrame {
		f := d.refBuf.Get(h.ShowExistingIdx)
		if f == nil {
			return nil, errors.New("vp9/decoder: show_existing_frame references empty slot")
		}
		return f, nil
	}
	// BitDepth is only read for key frames and intra-only inter frames.
	// Normal inter frames inherit bit depth from the sequence; accept 0 (not read).
	if h.BitDepth != 8 && h.BitDepth != 0 {
		return nil, errors.New("vp9/decoder: only 8-bit frames supported")
	}

	// 2. Compressed header (prob updates).
	// When CompressedHeaderSize exceeds the remaining payload the encoder is
	// signalling a "copy-from-reference" frame: all bool-coded reads past end
	// of data return 0 (no prob updates, no residual), so we clamp the slice.
	compEnd := h.Size + h.CompressedHeaderSize
	if compEnd > len(payload) {
		compEnd = len(payload)
	}
	compData := payload[h.Size:compEnd]
	compBR := boolcoder.NewReader(compData)
	isIntra := h.FrameType == frame.FrameKey
	if err := compheader.Decode(compBR, isIntra,
		h.InterpolationFilter, h.AllowHighPrecisionMV, d.probs); err != nil {
		return nil, err
	}

	// 3. Allocate output frame.
	// For inter frames that inherit dimensions via frame_size_with_refs,
	// h.FrameWidth/Height may be 0; fall back to stored decoder dimensions.
	fw, fh := h.FrameWidth, h.FrameHeight
	if fw == 0 || fh == 0 {
		fw, fh = d.frameWidth, d.frameHeight
	} else {
		d.frameWidth, d.frameHeight = fw, fh
	}
	dst := refbuf.NewFrame(fw, fh)
	w4 := (fw + 3) / 4
	h4 := (fh + 3) / 4

	// Resize MI grid if needed.
	if d.miW4 != w4 || d.miH4 != h4 {
		d.miW4 = w4
		d.miH4 = h4
		d.miGrid = make([]miBlock, w4*h4)
	} else {
		for i := range d.miGrid {
			d.miGrid[i] = miBlock{}
		}
	}

	// Gather reference frames.
	refs := [3]*refbuf.Frame{
		d.refBuf.Last(),
		d.refBuf.Golden(),
		d.refBuf.AltRef(),
	}

	// 4. Build FrameInfo for blockmode.
	fi := d.buildFrameInfo(h)

	// Store per-frame quantizer params for per-block segment Q recompute.
	d.baseQIdx = h.BaseQIdx
	d.deltaQYDC = h.DeltaQYDC
	d.deltaQUVDC = h.DeltaQUVDC
	d.deltaQUVAC = h.DeltaQUVAC

	// Store LF params for per-block level computation.
	d.lfBaseLevel = h.LoopFilterLevel
	d.lfSharpness = h.LoopFilterSharpness
	d.lfDeltaEnabled = h.LFDeltaEnabled
	d.lfRefDeltas = h.LFRefDeltas
	d.lfModeDeltas = h.LFModeDeltas

	// Default quantizer steps per plane (base Q index + per-plane deltas).
	// Y-DC uses DeltaQYDC; Y-AC has no delta. UV-DC/AC use their own deltas.
	lossless := h.BaseQIdx == 0 && h.DeltaQYDC == 0 && h.DeltaQUVDC == 0 && h.DeltaQUVAC == 0
	dcY, acY := qStepsForPlane(h.BaseQIdx, h.DeltaQYDC, 0)
	dcUV, acUV := qStepsForPlane(h.BaseQIdx, h.DeltaQUVDC, h.DeltaQUVAC)
	defaultDC := [3]int16{dcY, dcUV, dcUV}
	defaultAC := [3]int16{acY, acUV, acUV}

	// 5. Tile loop.
	tileReader, err := tile.NewReader(payload, h.Size, h.CompressedHeaderSize,
		h.TileColsLog2, h.TileRowsLog2)
	if err != nil {
		return nil, err
	}

	nTileCols := tileReader.NumCols()
	nTileRows := tileReader.NumRows()
	tileColW4 := (w4 + nTileCols - 1) / nTileCols

	// One above-row context per tile column (reset per tile row).
	aboveRows := make([]*ctx.Row, nTileCols)
	for tc := 0; tc < nTileCols; tc++ {
		tw4 := tileColW4
		if tc == nTileCols-1 {
			tw4 = w4 - tc*tileColW4
		}
		if tw4 <= 0 {
			tw4 = 1
		}
		aboveRows[tc] = ctx.NewRow(tw4 * 4)
	}

	// For inter frames, create a Counts struct to accumulate symbol counts
	// for probability adaptation after tile decoding.
	var counts *compheader.Counts
	if !isIntra {
		counts = new(compheader.Counts)
	}

	for tr := 0; tr < nTileRows; tr++ {
		for tc := 0; tc < nTileCols; tc++ {
			aboveRows[tc].Reset()
		}
		for tc := 0; tc < nTileCols; tc++ {
			tileData := tileReader.TileData(tr, tc)
			br := boolcoder.NewReader(tileData)

			// Tile bounds in 4×4 units.
			tileRow4 := tr * (h4 / nTileRows) // approximate row start
			tileCol4 := tc * tileColW4
			tileW4 := tileColW4
			if tc == nTileCols-1 {
				tileW4 = w4 - tileCol4
			}
			tileH4 := h4 / nTileRows
			if tr == nTileRows-1 {
				tileH4 = h4 - tileRow4
			}

			d.decodeTile(br, d.probs, counts, fi, dst, refs, aboveRows[tc],
				tileRow4, tileCol4, tileW4, tileH4,
				defaultDC, defaultAC, lossless)
		}
	}

	// 6a. Adapt probabilities for inter frames (spec §8.9).
	compheader.AdaptProbs(d.probs, counts)

	// 6b. Save MI grid to refreshed slots for temporal MV prediction.
	for i := 0; i < 8; i++ {
		if h.RefreshFrameFlags&(1<<uint(i)) != 0 {
			grid := make([]miBlock, len(d.miGrid))
			copy(grid, d.miGrid)
			d.prevMIGrids[i] = grid
			d.prevMIW4[i] = d.miW4
			d.prevMIH4[i] = d.miH4
		}
	}

	// 7. Loop filter with per-block levels.
	if h.LoopFilterLevel > 0 {
		sharpness := h.LoopFilterSharpness
		miW4 := d.miW4
		miGrid := d.miGrid
		paramsAt := func(row4, col4 int) loopfilter.Params {
			if row4 < 0 || col4 < 0 || row4 >= d.miH4 || col4 >= miW4 {
				return loopfilter.Params{}
			}
			lv := miGrid[row4*miW4+col4].lfLevel
			return loopfilter.ParamsForLevel(lv, sharpness)
		}
		paramsAtUV := func(row4, col4 int) loopfilter.Params {
			// UV is half-size: UV (row4, col4) maps to Y (row4*2, col4*2)
			return paramsAt(row4*2, col4*2)
		}
		loopfilter.FilterPlaneWithLevels(dst.Y, dst.YStride, dst.Width, dst.Height, 4, paramsAt)
		loopfilter.FilterPlaneWithLevels(dst.U, dst.UVStride, dst.Width/2, dst.Height/2, 3, paramsAtUV)
		loopfilter.FilterPlaneWithLevels(dst.V, dst.UVStride, dst.Width/2, dst.Height/2, 3, paramsAtUV)
	}

	// 8. Update refbuf.
	refbuf.ExtendBorders(dst)
	d.refBuf.Update(dst, h.RefreshFrameFlags)
	if isIntra {
		// Key frame: set LAST, GOLDEN, ALTREF all to slot 0 (the refreshed slot).
		d.refBuf.SetRef(0, 0)
		d.refBuf.SetRef(1, 0)
		d.refBuf.SetRef(2, 0)
	} else {
		for i, idx := range h.RefFrameIdx {
			d.refBuf.SetRef(i, idx)
		}
	}

	return dst, nil
}

// ---- tile / SB / partition --------------------------------------------------

func (d *Decoder) decodeTile(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	fi blockmode.FrameInfo, dst *refbuf.Frame, refs [3]*refbuf.Frame,
	aboveRow *ctx.Row,
	tileRow4, tileCol4, tileW4, tileH4 int,
	defaultDC, defaultAC [3]int16, lossless bool,
) {
	// SB size in 4×4 units = 16 (64 pixels).
	const sbN4 = 16

	for sbRow4 := tileRow4; sbRow4 < tileRow4+tileH4; sbRow4 += sbN4 {
		leftCol := ctx.NewCol()
		for sbCol4 := tileCol4; sbCol4 < tileCol4+tileW4; sbCol4 += sbN4 {
			d.decodePartition(br, probs, counts, fi, dst, refs,
				aboveRow, leftCol,
				sbRow4, sbCol4, 4, // bsLog2=4 → 64×64
				defaultDC, defaultAC, lossless)
		}
	}
}

// decodePartition recursively decodes the partition tree at (row4, col4) with
// block size bsLog2 (1..4, where 1=8×8 … 4=64×64).
func (d *Decoder) decodePartition(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	fi blockmode.FrameInfo, dst *refbuf.Frame, refs [3]*refbuf.Frame,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsLog2 int,
	defaultDC, defaultAC [3]int16, lossless bool,
) {
	if row4*4 >= dst.Height || col4*4 >= dst.Width {
		return
	}

	if bsLog2 == 0 {
		// 4×4 leaf: decode coding block directly (1×1 in 4×4 units).
		d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, 1, 1, defaultDC, defaultAC, lossless)
		return
	}

	// Read partition type.
	aboveP := ctx.PartitionAbove(aboveRow, col4, bsLog2)
	leftP := ctx.PartitionLeft(leftCol, row4, bsLog2)
	pctx := partition.Context(aboveP, leftP, bsLog2)
	pt := partition.Read(br, pctx, probs.PartitionProbs, counts)

	half := 1 << (bsLog2 - 1) // half-size in 4×4 units

	full := 1 << bsLog2 // width/height of this block in 4×4 units

	switch pt {
	case partition.None:
		// Square block: full × full.
		d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, full, full, defaultDC, defaultAC, lossless)

	case partition.Horz:
		// Two non-square blocks: full-wide × half-tall.
		d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, full, half, defaultDC, defaultAC, lossless)
		if row4+half < d.miH4 {
			d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
				row4+half, col4, full, half, defaultDC, defaultAC, lossless)
		}

	case partition.Vert:
		// Two non-square blocks: half-wide × full-tall.
		d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, half, full, defaultDC, defaultAC, lossless)
		if col4+half < d.miW4 {
			d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
				row4, col4+half, half, full, defaultDC, defaultAC, lossless)
		}

	case partition.Split:
		// Four quadrants, each recursively decoded at bsLog2-1.
		d.decodePartition(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, bsLog2-1, defaultDC, defaultAC, lossless)
		d.decodePartition(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4+half, bsLog2-1, defaultDC, defaultAC, lossless)
		d.decodePartition(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4+half, col4, bsLog2-1, defaultDC, defaultAC, lossless)
		d.decodePartition(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4+half, col4+half, bsLog2-1, defaultDC, defaultAC, lossless)
	}
}

// decodeCodingBlock decodes one leaf coding block.
func (d *Decoder) decodeCodingBlock(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	fi blockmode.FrameInfo, dst *refbuf.Frame, refs [3]*refbuf.Frame,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsW4, bsH4 int,
	defaultDC, defaultAC [3]int16, lossless bool,
) {
	// Decode mode info.
	bm := blockmode.Decode(br, probs, counts, fi, aboveRow, leftCol, col4, row4, bsW4, bsH4)

	// Per-segment quantizer + LF level.
	dc, ac := defaultDC, defaultAC
	var segP segment.Params
	if fi.SegEnabled {
		segP = segment.ParamsFor(fi.SegFeatEnabled, fi.SegFeatData, fi.SegAbsOrDelta, bm.SegID)
		qIdx := segment.ClampQIndex(d.baseQIdx, segP)
		if qIdx != d.baseQIdx {
			dc[0], ac[0] = qStepsForPlane(qIdx, d.deltaQYDC, 0)
			dc[1], ac[1] = qStepsForPlane(qIdx, d.deltaQUVDC, d.deltaQUVAC)
			dc[2], ac[2] = dc[1], ac[1]
		}
	}

	// Per-block loop filter level (spec §8.8.1).
	segLFActive := fi.SegEnabled && fi.SegFeatEnabled[bm.SegID][segment.FeatAltLF]
	lfLevel := loopfilter.LevelForBlock(
		d.lfBaseLevel,
		segLFActive, int16(segP.LFDelta),
		d.lfDeltaEnabled,
		bm.RefFrame[0], bm.IsInter, bm.MV[0].Row == 0 && bm.MV[0].Col == 0,
		d.lfRefDeltas, d.lfModeDeltas,
	)

	// Resolve motion vector for inter blocks.
	mv0, mv1 := d.resolveMVs(bm, fi, aboveRow, leftCol, row4, col4, bsW4, bsH4)

	// Reconstruct pixels.
	reconstruct.Block(br, probs, counts, bm, lossless, mv0, mv1, dc, ac,
		dst, refs, aboveRow, leftCol, row4, col4, bsW4, bsH4)

	// Update above/left context arrays (above uses block width, left uses height).
	interpFilter := uint8(bm.InterpFilter)
	if bm.InterpFilter < 0 {
		interpFilter = uint8(ctx.InterpSwitchable)
	}
	isZeroMV := bm.IsInter && bm.MV[0].Row == 0 && bm.MV[0].Col == 0
	aboveRow.SetBlock(col4, bsW4, bm.TxSize, bm.Skip, bm.RefFrame, bm.IsInter, isZeroMV,
		uint8(bm.YMode), interpFilter)
	leftCol.SetBlock(row4, bsH4, bm.TxSize, bm.Skip, bm.RefFrame, bm.IsInter, isZeroMV,
		uint8(bm.YMode), interpFilter)

	// Update partition context (above uses width, left uses height).
	n8W := bsW4 / 2
	if n8W < 1 {
		n8W = 1
	}
	n8H := bsH4 / 2
	if n8H < 1 {
		n8H = 1
	}
	aboveRow.SetPartition(col4/2, n8W, ctx.PartitionAboveMasks[log2n4(bsW4)])
	leftCol.SetPartition(row4/2, n8H, ctx.PartitionLeftMasks[log2n4(bsH4)])

	// Store MI info for MV prediction and LF.
	d.setMI(row4, col4, bsW4, bsH4, bm, mv0, mv1, lfLevel)
}

// ---- MV resolution ----------------------------------------------------------

func (d *Decoder) resolveMVs(bm blockmode.BlockMode, fi blockmode.FrameInfo,
	aboveRow *ctx.Row, leftCol *ctx.Col, row4, col4, bsW4, bsH4 int,
) (mv0, mv1 mv.MV) {
	if !bm.IsInter {
		return mv.MV{}, mv.MV{}
	}

	targetRef := bm.RefFrame[0]
	above := d.getNeighbourMV(row4-1, col4, targetRef)
	aboveRight := d.getNeighbourMV(row4-1, col4+1, targetRef)
	left := d.getNeighbourMV(row4, col4-1, targetRef)
	aboveLeft := d.getNeighbourMV(row4-1, col4-1, targetRef)

	// Temporal candidate from co-located block in LAST reference frame.
	// Center of current block (spec §8.3.4 / libvpx find_mv_refs).
	var temporal mvpred.BlockMV
	lastSlot := d.refBuf.Refs[0]
	if d.prevMIGrids[lastSlot] != nil {
		tRow4 := row4 + bsH4/2
		tCol4 := col4 + bsW4/2
		prevW4 := d.prevMIW4[lastSlot]
		prevH4 := d.prevMIH4[lastSlot]
		if tRow4 < prevH4 && tCol4 < prevW4 {
			tmpMI := d.prevMIGrids[lastSlot][tRow4*prevW4+tCol4]
			if tmpMI.inter {
				temporal = mvpred.BlockMV{Ref: tmpMI.ref[0], MV: tmpMI.mv[0]}
			}
		}
	}
	nearest, near := mvpred.FindBestRefMVs(above, aboveRight, left, aboveLeft, temporal, targetRef)

	switch bm.InterMode {
	case blockmode.NearestMV:
		mv0 = nearest
	case blockmode.NearMV:
		mv0 = near
	case blockmode.ZeroMV:
		mv0 = mv.MV{}
	case blockmode.NewMV:
		mv0 = bm.MV[0]
	}

	if bm.RefFrame[1] != ctx.RefIntra {
		targetRef1 := bm.RefFrame[1]
		above1 := d.getNeighbourMV(row4-1, col4, targetRef1)
		aboveRight1 := d.getNeighbourMV(row4-1, col4+1, targetRef1)
		left1 := d.getNeighbourMV(row4, col4-1, targetRef1)
		aboveLeft1 := d.getNeighbourMV(row4-1, col4-1, targetRef1)
		var temporal1 mvpred.BlockMV
		lastSlot1 := d.refBuf.Refs[0]
		if d.prevMIGrids[lastSlot1] != nil {
			tRow4 := row4 + bsH4/2
			tCol4 := col4 + bsW4/2
			prevW4 := d.prevMIW4[lastSlot1]
			prevH4 := d.prevMIH4[lastSlot1]
			if tRow4 < prevH4 && tCol4 < prevW4 {
				tmpMI := d.prevMIGrids[lastSlot1][tRow4*prevW4+tCol4]
				if tmpMI.inter && tmpMI.ref[1] == targetRef1 {
					temporal1 = mvpred.BlockMV{Ref: tmpMI.ref[1], MV: tmpMI.mv[1]}
				}
			}
		}
		nearest1, near1 := mvpred.FindBestRefMVs(above1, aboveRight1, left1, aboveLeft1, temporal1, targetRef1)

		switch bm.InterMode {
		case blockmode.NearestMV:
			mv1 = nearest1
		case blockmode.NearMV:
			mv1 = near1
		case blockmode.ZeroMV:
			mv1 = mv.MV{}
		case blockmode.NewMV:
			mv1 = bm.MV[1]
		}
	}

	return mv0, mv1
}

// getNeighbourMV reads MV info from the MI grid for the given neighbour block.
func (d *Decoder) getNeighbourMV(row4, col4, targetRef int) mvpred.BlockMV {
	if row4 < 0 || row4 >= d.miH4 || col4 < 0 || col4 >= d.miW4 {
		return mvpred.BlockMV{Ref: ctx.RefIntra, MV: mv.MV{}}
	}
	mi := d.miGrid[row4*d.miW4+col4]
	if !mi.inter {
		return mvpred.BlockMV{Ref: ctx.RefIntra, MV: mv.MV{}}
	}
	return mvpred.BlockMV{Ref: mi.ref[0], MV: mi.mv[0]}
}

// setMI stores the decoded MI block info for MV prediction and LF.
func (d *Decoder) setMI(row4, col4, bsW4, bsH4 int, bm blockmode.BlockMode, mv0, mv1 mv.MV, lfLevel int) {
	mi := miBlock{
		inter:   bm.IsInter,
		ref:     bm.RefFrame,
		mv:      [2]mv.MV{mv0, mv1},
		zeroMV:  mv0.Row == 0 && mv0.Col == 0,
		lfLevel: lfLevel,
	}
	for r := 0; r < bsH4 && row4+r < d.miH4; r++ {
		for c := 0; c < bsW4 && col4+c < d.miW4; c++ {
			d.miGrid[(row4+r)*d.miW4+(col4+c)] = mi
		}
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

// ---- quantizer helpers ------------------------------------------------------

// qStepsForPlane returns the DC and AC dequantisation step sizes for a given
// Q-index and per-plane deltas (spec §8.6.1 get_dc_quant / get_ac_quant).
// qIdx is the base (or segment-adjusted) Q index; dcDelta/acDelta are the
// per-plane additions from the frame header.
func qStepsForPlane(qIdx, dcDelta, acDelta int) (dc, ac int16) {
	clamp := func(q int) int {
		if q < 0 {
			return 0
		}
		if q > 255 {
			return 255
		}
		return q
	}
	return dequant.DCQuant(clamp(qIdx + dcDelta)), dequant.ACQuant(clamp(qIdx + acDelta))
}

// buildFrameInfo builds the FrameInfo for blockmode.Decode from a parsed header.
func (d *Decoder) buildFrameInfo(h *frame.Header) blockmode.FrameInfo {
	fi := blockmode.FrameInfo{
		IsKeyframe:     h.FrameType == frame.FrameKey,
		AllowHP:        h.AllowHighPrecisionMV,
		ReferenceMode:  d.probs.ReferenceMode,
		InterpFilter:   h.InterpolationFilter,
		SegEnabled:     h.SegEnabled,
		SegUpdateMap:   h.SegUpdateMap,
		SegTreeProbs:   h.SegTreeProbs,
		SegFeatEnabled: h.SegFeatEnabled,
		SegFeatData:    h.SegFeatData,
		SegAbsOrDelta:  h.SegAbsOrDelta,
	}
	if !fi.IsKeyframe {
		fi.CompFixedRef, fi.CompVarRef = blockmode.CompFixedVarRef(h.RefFrameSign)
	}
	return fi
}
