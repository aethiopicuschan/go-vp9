package decoder

import (
	"errors"
	"fmt"
	"os"

	"github.com/aethiopicuschan/go-vp9/blockmode"
	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/coeftoken"
	"github.com/aethiopicuschan/go-vp9/compheader"
	"github.com/aethiopicuschan/go-vp9/ctx"
	"github.com/aethiopicuschan/go-vp9/dequant"
	"github.com/aethiopicuschan/go-vp9/frame"
	"github.com/aethiopicuschan/go-vp9/intra16"
	"github.com/aethiopicuschan/go-vp9/loopfilter"
	"github.com/aethiopicuschan/go-vp9/mc16"
	"github.com/aethiopicuschan/go-vp9/partition"
	"github.com/aethiopicuschan/go-vp9/reconstruct"
	"github.com/aethiopicuschan/go-vp9/refbuf"
	"github.com/aethiopicuschan/go-vp9/tile"
	"github.com/aethiopicuschan/go-vp9/transform"
)

// High-bit-depth (10/12-bit) decode path — a parallel set of decodeOneFrame /
// decodePartition / decodeCodingBlock that reconstruct into a refbuf.Frame16 via
// the verified _16 pixel pipeline. The entropy decode (header, probs, block mode,
// MVs) is shared with the 8-bit path; only the pixel target differs.

// DecodeFrame16 decodes a VP9 profile 2/3 (10/12-bit) packet, returning the
// frame it outputs with 16-bit samples. It mirrors DecodeFrame, including the
// (nil, nil) result for a packet that shows no frame.
func (d *Decoder) DecodeFrame16(payload []byte) (*refbuf.Frame16, error) {
	decodeMu.Lock()
	defer decodeMu.Unlock()
	var shown *refbuf.Frame16
	// Packet structure (superframe index, trailing data) and the error state
	// follow libvpx: see decodePacket.
	err := d.decodePacket(payload, func(sub []byte) (int, error) {
		f, end, err := d.decodeOneFrame16(sub)
		if err != nil {
			return 0, err
		}
		// libvpx outputs at most one frame per packet, and only if the frame
		// decoded last is a shown one (vp9_get_raw_frame: cm->show_frame).
		shown = nil
		if d.lastShown {
			shown = f
		}
		return end, nil
	})
	if err != nil {
		return nil, err
	}
	return shown, nil
}

func (d *Decoder) decodeOneFrame16(payload []byte) (*refbuf.Frame16, int, error) {
	// 1. Uncompressed header.
	h, err := frame.ParseRefs(payload, d.refSize16)
	if err != nil {
		d.rejectedHeader(h, d.refSlots16())
		return nil, 0, err
	}
	if p := os.Getenv("BLK_DUMP_GO"); p != "" && blkDumpW == nil {
		if fpd, e := os.Create(p); e == nil {
			blkDumpW = fpd
		}
	}
	if !h.ShowExistingFrame {
		blkFrame++
	}
	if os.Getenv("VP9_HDRDBG") != "" {
		println("HDR type=", int(h.FrameType), "show=", h.ShowFrame, "showEx=", h.ShowExistingFrame,
			"w=", h.FrameWidth, "h=", h.FrameHeight,
			"refIdx=", h.RefFrameIdx[0], h.RefFrameIdx[1], h.RefFrameIdx[2],
			"refSign=", h.RefFrameSign[0], h.RefFrameSign[1], h.RefFrameSign[2],
			"refresh=", int(h.RefreshFrameFlags), "baseQ=", h.BaseQIdx,
			"compHdr=", h.CompressedHeaderSize, "hdrSize=", h.Size, "lf=", h.LoopFilterLevel)
	}
	if h.ShowExistingFrame {
		f := d.refBuf16.Get(h.ShowExistingIdx)
		if f == nil {
			return nil, 0, errors.New("vp9/decoder: show_existing_frame references empty slot")
		}
		// While the decoder waits for a key or intra-only frame after an
		// error, libvpx decodes a show_existing_frame without error but
		// outputs nothing (decoder_get_frame: ctx->need_resync).
		d.lastShown = !d.needResync
		return f, h.Size, nil
	}
	// BitDepth is only read for key frames and intra-only inter frames.
	// Normal inter frames inherit bit depth from the sequence; accept 0 (not read).
	// (bit depth 10/12 handled by this path)

	// Header checks that depend on the decoder state (resync after an error,
	// reference slots, header length): see beginFrame.
	if err := d.beginFrame(h, len(payload), d.refSlots16()); err != nil {
		return nil, 0, err
	}

	// 2. Compressed header (prob updates). beginFrame made sure it lies inside
	// the packet. libvpx read_compressed_header fails the frame when the bool
	// decoder's marker bit is set ("Failed to allocate bool decoder 0") and,
	// below, when the header over-reads its partition.
	compData := payload[h.Size : h.Size+h.CompressedHeaderSize]
	compBR := boolcoder.NewReaderVP9(compData)
	if compBR.MarkerError() {
		return nil, 0, errCompHdrMarker
	}
	isIntra := h.FrameType == frame.FrameKey
	frameIsIntra := isIntra || h.IntraOnly

	// VP9 probability frame-context model (§8.4 / libvpx vp9_setup_past_independence).
	// Intra (key/intra-only) frames and error-resilient frames reset frame
	// contexts; the working fc is loaded from frame_contexts[idx], gets the
	// compressed-header forward updates, and is later backward-adapted from the
	// pre-update context using this frame's symbol counts.
	// Those frames always work with context 0 (frameContextIndex).
	ctxIdx, err := d.frameContextIndex(h)
	if err != nil {
		return nil, 0, err
	}
	if frameIsIntra || h.ErrorResilient {
		// vp9_setup_past_independence forces frame_context_idx to 0 (done by
		// frameContextIndex: the coded index only selects the context that
		// reset_frame_context=2 resets) and zeroes ref_frame_sign_bias — after
		// the header's sign-bias bits were read, so an error-resilient inter
		// frame never has compound references.
		h.FrameContextIdx = ctxIdx
		h.RefFrameSign = [3]bool{}
	}
	preFc := d.frameContexts[ctxIdx]
	d.probs = preFc.Clone() // working fc (forward-updated below; used by buildFrameInfo)
	// Compound prediction is allowed only when some reference frame has a
	// different sign bias (libvpx vp9_compound_reference_allowed); otherwise the
	// frame reference_mode is SINGLE with no bits coded.
	compoundAllowed := h.RefFrameSign[1] != h.RefFrameSign[0] || h.RefFrameSign[2] != h.RefFrameSign[0]
	d.allowHP = h.AllowHighPrecisionMV
	d.signBias = [4]bool{false, h.RefFrameSign[0], h.RefFrameSign[1], h.RefFrameSign[2]}
	// use_prev_frame_mvs (libvpx): the decode-order previous frame's co-located
	// MVs are valid temporal candidates only when it had the same size, was a
	// shown non-key non-intra-only frame, and this frame is not error-resilient.
	d.usePrevFrameMVs = !h.ErrorResilient && d.prevFrameMI != nil &&
		h.FrameWidth == d.prevFrameWidth && h.FrameHeight == d.prevFrameHeight &&
		!d.prevFrameIntraOnly && d.prevFrameShow && !d.lastFrameKey
	// A lossless frame has no tx_mode in its compressed header (ONLY_4X4).
	hdrLossless := h.BaseQIdx == 0 && h.DeltaQYDC == 0 && h.DeltaQUVDC == 0 && h.DeltaQUVAC == 0
	if err := compheader.DecodeLossless(compBR, frameIsIntra,
		h.InterpolationFilter, h.AllowHighPrecisionMV, compoundAllowed, hdrLossless, d.probs); err != nil {
		return nil, 0, err
	}
	// libvpx read_compressed_header returns vpx_reader_has_error: "Decode
	// failed. Frame data header is corrupted."
	if compBR.Overread() {
		return nil, 0, errCompHdrOverread
	}

	if blkFrame == 1 {
		blockmode.DbgW = blkDumpW
		reconstruct.DbgW = blkDumpW
	} else {
		blockmode.DbgW = nil
		reconstruct.DbgW = nil
	}

	// Decode-time probs: intra frames read partition with the fixed
	// kf_partition_probs table (not the adapted fc partition probs).
	decodeProbs := d.probs
	if frameIsIntra {
		decodeProbs = d.probs.Clone()
		decodeProbs.PartitionProbs = kfPartitionProbs
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
	// bit_depth is only coded on key / intra-only frames; inter frames inherit it
	// from the sequence (the key frame stored it in d.bitDepth). Without this, an
	// inter frame got bd=0 → every bit-depth global (MaxVal/mid/shift) was garbage.
	bd := h.BitDepth
	if bd == 0 {
		bd = d.bitDepth
	}
	d.bitDepth = bd
	// Chroma format: signalled by key and intra-only frames, inherited by
	// inter frames.
	if frameIsIntra {
		d.ssX, d.ssY = 0, 0
		if h.SubsamplingX {
			d.ssX = 1
		}
		if h.SubsamplingY {
			d.ssY = 1
		}
	}
	dst := refbuf.NewFrame16SS(fw, fh, bd, d.ssX, d.ssY)
	mx := (1 << uint(bd)) - 1
	intra16.MaxVal = mx
	intra16.Base = 1 << uint(bd-1)
	mc16.MaxVal = mx
	reconstruct.MaxVal16 = mx
	transform.BD16 = bd
	coeftoken.BD = bd
	loopfilter.SetBitDepth16(bd)
	// The MI grid is libvpx's: whole 8×8 units (mi_cols = ceil(width/8)), here
	// counted in 4×4 units. A frame whose size is not a multiple of 8 still
	// codes, reconstructs and loop-filters the full last 8×8 column/row.
	w4 := (fw + 7) / 8 * 2
	h4 := (fh + 7) / 8 * 2

	// Transform blocks whose top-left lies outside the MI grid (right/bottom
	// frame edge) are not coded; reconstruct skips them (libvpx max_blocks_*).
	// This is package state of reconstruct, so it is set for every frame, not
	// only when this decoder's grid changes: another Decoder may have decoded
	// a frame of a different size in between.
	reconstruct.MiW4 = w4
	reconstruct.MiH4 = h4

	// A fresh (zeroed) MI grid for every frame: the finished grid is handed
	// over, not copied, as the temporal-MV source of the next frame and of
	// the refreshed reference slots, so it must not be written again.
	d.miW4 = w4
	d.miH4 = h4
	d.miGrid = make([]miBlock, w4*h4)

	// Map this frame's LAST/GOLDEN/ALTREF to ref slots BEFORE gathering them:
	// ref_frame_idx is per-frame and may differ from the previous frame's, so
	// the mapping must be applied here, not carried over (it was previously set
	// only at end-of-frame, which silently worked until a frame used a different
	// ref_frame_idx — e.g. an alt-ref with [0 2 1]).
	if !frameIsIntra {
		for i, idx := range h.RefFrameIdx {
			d.refBuf16.SetRef(i, idx)
		}
	}

	// Gather reference frames.
	refs := [3]*refbuf.Frame16{
		d.refBuf16.Last(),
		d.refBuf16.Golden(),
		d.refBuf16.AltRef(),
	}

	d.invalidBlock = false

	// 4. Build FrameInfo for blockmode.
	fi := d.buildFrameInfo(h)
	d.segBeginFrame(h, &fi, fw, fh)
	d.segInvalid = false

	// Store per-frame quantizer params for per-block segment Q recompute.
	d.baseQIdx = h.BaseQIdx
	d.deltaQYDC = h.DeltaQYDC
	d.deltaQUVDC = h.DeltaQUVDC
	d.deltaQUVAC = h.DeltaQUVAC

	// Store LF params for per-block level computation.
	d.lfBaseLevel = h.LoopFilterLevel
	d.lfSharpness = h.LoopFilterSharpness
	d.lfDeltaEnabled = h.LFDeltaEnabled
	// Loop-filter ref/mode deltas persist across frames; setup_past_independence
	// (intra/error-resilient frames) resets them to the VP9 defaults, then the
	// header's per-delta updates are applied on top (spec §7.2.8 / libvpx).
	if frameIsIntra || h.ErrorResilient {
		d.lfRefDeltas = [4]int{1, 0, -1, -1}
		d.lfModeDeltas = [2]int{0, 0}
	}
	for i := 0; i < 4; i++ {
		if h.LFRefDeltaUpd[i] {
			d.lfRefDeltas[i] = h.LFRefDeltas[i]
		}
	}
	for i := 0; i < 2; i++ {
		if h.LFModeDeltaUpd[i] {
			d.lfModeDeltas[i] = h.LFModeDeltas[i]
		}
	}

	// Default quantizer steps per plane (base Q index + per-plane deltas).
	// Y-DC uses DeltaQYDC; Y-AC has no delta. UV-DC/AC use their own deltas.
	lossless := h.BaseQIdx == 0 && h.DeltaQYDC == 0 && h.DeltaQUVDC == 0 && h.DeltaQUVAC == 0
	dcY, acY := qStepsForPlaneBD(h.BaseQIdx, h.DeltaQYDC, 0, bd)
	dcUV, acUV := qStepsForPlaneBD(h.BaseQIdx, h.DeltaQUVDC, h.DeltaQUVAC, bd)
	defaultDC := [3]int16{dcY, dcUV, dcUV}
	defaultAC := [3]int16{acY, acUV, acUV}

	// 5. Tile loop.
	tileReader, err := tile.NewReader(payload, h.Size, h.CompressedHeaderSize,
		h.TileColsLog2, h.TileRowsLog2)
	if err != nil {
		return nil, 0, err
	}

	nTileCols := tileReader.NumCols()
	nTileRows := tileReader.NumRows()

	// Tile boundaries are superblock-aligned (libvpx get_tile_offset): in 8×8 MI
	// units offset = ((idx * sb_cols) >> log2) << 3, clamped to mi_cols. go works
	// in 4×4 units, so ×2 (and clamp to the actual padded width/height).
	miCols := (fw + 7) >> 3
	miRows := (fh + 7) >> 3
	sbCols := (miCols + 7) >> 3
	sbRows := (miRows + 7) >> 3
	tileColStart := func(idx int) int {
		off := ((idx * sbCols) >> uint(h.TileColsLog2)) << 3
		if off > miCols {
			off = miCols
		}
		if o4 := off * 2; o4 < w4 {
			return o4
		}
		return w4
	}
	tileRowStart := func(idx int) int {
		off := ((idx * sbRows) >> uint(h.TileRowsLog2)) << 3
		if off > miRows {
			off = miRows
		}
		if o4 := off * 2; o4 < h4 {
			return o4
		}
		return h4
	}

	// One full-width above-row context, reset once per frame and carried across
	// tile rows (libvpx decode_tiles zeroes above_context / above_seg_context
	// before the tile-row loop only; tile rows are not independent).
	// Tile columns write disjoint column ranges, so a single shared array
	// (indexed by absolute col4) is correct.
	aboveRow := ctx.NewRow(w4 * 4)

	// Accumulate symbol counts for backward probability adaptation (all frames,
	// including key frames — libvpx adapts the saved context after every frame).
	counts := new(compheader.Counts)

	const sbN4 = 16   // superblock = 64 px = 16 4×4 units
	// libvpx's two per-tile bool-decoder error conditions (both fatal there):
	// markerErr = vpx_reader_init saw a 1 marker bit; overread =
	// vpx_reader_has_error, i.e. zero-fill reached the arithmetic window.
	overread, markerErr := false, false
	aboveRow.Reset()
	d.invalidMV = false
	// End of the frame's data as libvpx reports it to its packet loop:
	// vpx_reader_find_end of the last tile (decode_tiles).
	frameEnd := len(payload)
	for tr := 0; tr < nTileRows; tr++ {
		tileRow4 := tileRowStart(tr)
		tileBot4 := tileRowStart(tr + 1)

		// Per-tile-column bool-coder + left context, persistent across this tile
		// row's SB rows. Decode is row-interleaved across tile columns (libvpx
		// decode_tiles order), so a block's above-right neighbour in the next tile
		// column has already been reconstructed.
		brs := make([]*boolcoder.Reader, nTileCols)
		lefts := make([]*ctx.Col, nTileCols)
		for tc := 0; tc < nTileCols; tc++ {
			brs[tc] = boolcoder.NewReaderVP9(tileReader.TileData(tr, tc))
			lefts[tc] = ctx.NewCol(tileBot4)
		}
		for sbRow4 := tileRow4; sbRow4 < tileBot4; sbRow4 += sbN4 {
			for tc := 0; tc < nTileCols; tc++ {
				colStart := tileColStart(tc)
				colEnd := tileColStart(tc + 1)
				// Neighbour availability (entropy + intra prediction + MV-reference
				// search): left is tile-relative, above is frame-relative (libvpx
				// set_mi_row_col: above_mi = mi_row != 0, left_mi = mi_col >
				// tile->mi_col_start), so a tile row sees the tile row above it.
				fi.MiColStart = colStart
				fi.MiRowStart = 0
				reconstruct.TileColStart4 = colStart
				reconstruct.TileRowStart4 = 0
				d.curTileColStart4 = colStart
				d.curTileColEnd4 = colEnd
				for sbCol4 := colStart; sbCol4 < colEnd; sbCol4 += sbN4 {
					d.decodePartition16(brs[tc], decodeProbs, counts, fi, dst, refs,
						aboveRow, lefts[tc], sbRow4, sbCol4, 4,
						defaultDC, defaultAC, lossless)
				}
			}
		}
		for tc := 0; tc < nTileCols; tc++ {
			// Same verdict as libvpx's per-block sticky check (decode_block:
			// xd->corrupted |= vpx_reader_has_error(r)): the condition is
			// monotonic in the bits consumed, so once per tile suffices.
			if brs[tc].Overread() {
				overread = true
			}
			if brs[tc].MarkerError() {
				markerErr = true
			}
		}
		if tr == nTileRows-1 {
			frameEnd = tileReader.LastTileOffset() + brs[nTileCols-1].FindEnd()
		}
	}
	// An entropy desync (decoder bug, unsupported feature or corrupt/truncated
	// data) makes a tile consume more bits than it holds; the reconstruction is
	// then garbage. Reject the frame under exactly libvpx's criterion (see
	// boolcoder.Reader.Overread: the tile must end with at least 8 unconsumed
	// bits) so callers fall back instead of rendering a corrupt picture. The
	// limit is tight (in-sync libvpx tiles end 1..6 bits before it), so when
	// this fires on a stream libvpx accepts, the cause is an earlier decode
	// divergence in this decoder, not the check.
	if d.invalidBlock {
		return nil, 0, errors.New("vp9/decoder: invalid block size for chroma subsampling")
	}
	if markerErr {
		return nil, 0, errors.New("vp9/decoder: corrupt tile (bool-decoder marker bit set)")
	}
	if overread {
		return nil, 0, errors.New("vp9/decoder: entropy desync (bitstream over-read)")
	}
	// libvpx's third source of xd->corrupted: a NEWMV vector out of range.
	if d.invalidMV {
		return nil, 0, errInvalidMV
	}
	if d.segInvalid {
		// libvpx: "Invalid usage of segment feature on small blocks".
		return nil, 0, errors.New("vp9/decoder: segment skip feature on a sub-8x8 inter block")
	}

	// 6a. Backward-adapt the saved frame context from the PRE-update context
	// (spec §8.4.1 / libvpx vp9_adapt_*_probs uses pre_fc=frame_contexts[idx]).
	// Disabled by error_resilient or frame_parallel_decoding_mode.
	if !h.ErrorResilient && !h.FrameParallel {
		// Coefficient-prob adaptation factor (libvpx vp9_adapt_coef_probs):
		// count_sat is always 24; update_factor is 112 except for an inter frame
		// whose previous frame was a key frame (adapt quickly → 128).
		coefUpdateFactor := uint32(112)
		if !frameIsIntra && d.lastFrameKey {
			coefUpdateFactor = 128
		}
		// The implied hp=1 bit of non-hp MVs is counted (libvpx vp9_inc_mv), but
		// vp9_adapt_mv_probs adapts the hp probs only when the frame allows hp.
		if !h.AllowHighPrecisionMV {
			counts.MVClass0HPProbs = [2][2]uint32{}
			counts.MVHPProbs = [2][2]uint32{}
		}
		adapted := preFc.Clone()
		compheader.AdaptProbs(adapted, counts, !frameIsIntra, 24, coefUpdateFactor)
		if frameIsIntra {
			// Intra frames adapt only the coef probs; every other table keeps
			// the forward-updated cm->fc values (tx/skip deltas from this
			// frame's compressed header), not the pre-update context.
			coef := adapted.CoefProbs
			adapted = d.probs.Clone()
			adapted.CoefProbs = coef
		}
		d.probs = adapted
	}
	// Save the resulting context if requested.
	if h.RefreshFrameContext {
		d.frameContexts[ctxIdx] = d.probs
	}
	// Track this frame's type for the next frame's coef-adaptation speed.
	d.lastFrameKey = isIntra

	// Save this (decode-order) frame's MI grid + metadata as the temporal-MV
	// source for the next frame (libvpx cm->prev_frame).
	// The grid is shared read-only from here on (see its allocation).
	d.prevFrameMI = d.miGrid
	d.prevFrameW4 = d.miW4
	d.prevFrameWidth = h.FrameWidth
	d.prevFrameHeight = h.FrameHeight
	d.prevFrameIntraOnly = h.IntraOnly
	d.prevFrameShow = h.ShowFrame

	// 6b. Save MI grid to refreshed slots for temporal MV prediction.
	for i := 0; i < 8; i++ {
		if h.RefreshFrameFlags&(1<<uint(i)) != 0 {
			d.prevMIGrids[i] = d.miGrid
			d.prevMIW4[i] = d.miW4
			d.prevMIH4[i] = d.miH4
		}
	}

	// 7. Loop filter with per-block levels.
	if h.LoopFilterLevel > 0 && os.Getenv("VPX_NO_LF") == "" {
		sharpness := h.LoopFilterSharpness
		miW4, miH4 := d.miW4, d.miH4
		miGrid := d.miGrid
		yInfo := func(r4, c4 int) (int, uint8, bool, bool, bool) {
			if r4 < 0 || c4 < 0 || r4 >= miH4 || c4 >= miW4 {
				return 0, 0, false, false, false
			}
			m := &miGrid[r4*miW4+c4]
			// Prediction-block boundary: this 4×4 is the block's top (H) / left (V).
			return m.lfLevel, m.txSize, m.skipLF, r4 == m.bRow4, c4 == m.bCol4
		}
		// UV (4:2:0): a chroma 8×8 cell == one luma 16×16 region. libvpx builds the
		// chroma loop-filter masks (TX boundaries + internal 4×4) from ONLY the
		// top-left 8×8 block of that region (the "first block" rule, build_masks /
		// build_y_mask), and for a skipped inter block it sets just the prediction
		// boundary (early return, no TX/interior). So the filter LEVEL is taken
		// per-position (each 8×8's own lfl) but the TX size and skip flag that
		// decide which edges to filter come from the region's top-left 8×8.
		// Chroma loop filter: libvpx samples EVERYTHING (level + masks) from the
		// top-left 8×8 of the 16×16 region (vp9_filter_block_plane_ss11 reads
		// lfl_y[(r<<3)+(c<<1)] — both indices even). Round the chroma 4×4 index
		// down to even before scaling to luma 4×4 units.
		uvInfo := func(r4, c4 int) (int, uint8, bool, bool, bool) {
			cr, cc := (r4&^1)*2, (c4&^1)*2
			if cr < 0 || cc < 0 || cr >= miH4 || cc >= miW4 {
				return 0, 0, false, false, false
			}
			cell := &miGrid[cr*miW4+cc]
			return cell.lfLevel, cell.uvTxSize, cell.skipLF, r4 == cell.bRow4/2, c4 == cell.bCol4/2
		}
		// libvpx filters the whole MI grid (mi_rows × mi_cols 8×8 units), not
		// just the visible picture; a chroma 4×4 column/row per MI (4:2:0).
		yW4, yH4 := miW4, miH4
		uvW4, uvH4 := miW4/2, miH4/2
		if p := os.Getenv("LF_EDGEDUMP_GO"); p != "" {
			if fpd, err := os.Create(p); err == nil {
				loopfilter.DumpW = fpd
				defer fpd.Close()
			}
		}
		loopfilter.DumpPlane = 0
		loopfilter.FilterFrame16(dst.Y[dst.YBase():], dst.YStride, yW4, yH4, 16, sharpness, yInfo)
		switch {
		case dst.SubX == 1 && dst.SubY == 1:
			loopfilter.DumpPlane = 1
			loopfilter.FilterFrame16(dst.U[dst.UBase():], dst.UVStride, uvW4, uvH4, 8, sharpness, uvInfo)
			loopfilter.DumpPlane = 2
			loopfilter.FilterFrame16(dst.V[dst.VBase():], dst.UVStride, uvW4, uvH4, 8, sharpness, uvInfo)
		case dst.SubX == 0 && dst.SubY == 0:
			// 4:4:4: libvpx filters the chroma planes with the luma masks and
			// levels (vp9_filter_block_plane_ss00 for every plane).
			loopfilter.DumpPlane = 1
			loopfilter.FilterFrame16(dst.U[dst.UBase():], dst.UVStride, yW4, yH4, 16, sharpness, yInfo)
			loopfilter.DumpPlane = 2
			loopfilter.FilterFrame16(dst.V[dst.VBase():], dst.UVStride, yW4, yH4, 16, sharpness, yInfo)
		default:
			// 4:2:2 / 4:4:0: libvpx's generic vp9_filter_block_plane_non420,
			// driven per 8×8 mode-info cell.
			miInfo := func(miRow, miCol int) (int, uint8, bool, bool, bool) {
				r4, c4 := miRow*2, miCol*2
				if r4 >= miH4 || c4 >= miW4 {
					return 0, 0, false, false, false
				}
				cell := &miGrid[r4*miW4+c4]
				return cell.lfLevel, cell.uvTxSize, cell.skipLF, c4 == cell.bCol4, r4 == cell.bRow4
			}
			miRows, miCols := (miH4+1)/2, (miW4+1)/2
			for _, buf := range [][]uint16{dst.U[dst.UBase():], dst.V[dst.VBase():]} {
				loopfilter.FilterFrameNon420(dst.UVStride, miRows, miCols, dst.SubX, dst.SubY, sharpness, miInfo,
					func(q0, step, width int, p loopfilter.Params) {
						loopfilter.FilterEdge4_16(buf, q0, step, width, p)
					})
			}
		}
		loopfilter.DumpW = nil
	}

	// 8. Update refbuf. (The LAST/GOLDEN/ALTREF mapping for the next frame is set
	// at the start of its own decode from its ref_frame_idx, above.)
	refbuf.ExtendBorders16(dst)
	d.refBuf16.Update(dst, h.RefreshFrameFlags)
	for i := 0; i < 8; i++ {
		if h.RefreshFrameFlags&(1<<uint(i)) != 0 {
			d.slotW[i], d.slotH[i] = fw, fh
		}
	}

	d.segEndFrame(h)

	d.lastShown = h.ShowFrame
	return dst, frameEnd, nil
}

func (d *Decoder) decodePartition16(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	fi blockmode.FrameInfo, dst *refbuf.Frame16, refs [3]*refbuf.Frame16,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsLog2 int,
	defaultDC, defaultAC [3]int16, lossless bool,
) {
	if row4 >= d.miH4 || col4 >= d.miW4 || d.invalidBlock {
		return
	}

	// Read partition type, forcing SPLIT/HORZ/VERT at the frame edge when this
	// block would extend past the bottom/right (libvpx has_rows/has_cols).
	aboveP := ctx.PartitionAbove(aboveRow, col4, bsLog2)
	leftP := ctx.PartitionLeft(leftCol, row4, bsLog2)
	pctx := partition.Context(aboveP, leftP, bsLog2)
	half := 1 << (bsLog2 - 1) // half-size in 4×4 units
	hasRows := row4+half < d.miH4
	hasCols := col4+half < d.miW4
	var prngBefore uint32
	if blkDumpW != nil {
		prngBefore, _ = br.State()
	}
	pt := partition.ReadForced(br, pctx, probs.PartitionProbs, counts, hasRows, hasCols)
	if blkDumpW != nil {
		fmt.Fprintf(blkDumpW, "P %d %d %d bs=%d ctx=%d rng=%d p=%d->%d\n",
			blkFrame, row4/2, col4/2, bsLog2, pctx, prngBefore, probs.PartitionProbs[pctx][0], int(pt))
	}

	full := 1 << bsLog2 // width/height of this block in 4×4 units

	// At the 8×8 level (bsLog2==1) a non-NONE partition does NOT recurse: the
	// whole 8×8 MI cell is one coding block whose sub-prediction layout
	// (BLOCK_8X4/4X8/4X4) is selected by the partition (spec §6.4.4, the
	// subSize < BLOCK_8X8 branch). partition.Type values map 1:1 to Sub8x8*.
	if bsLog2 == 1 {
		d.decodeCodingBlock16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, full, full, int(pt), defaultDC, defaultAC, lossless)
		return
	}

	switch pt {
	case partition.None:
		// Square block: full × full.
		d.decodeCodingBlock16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, full, full, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)

	case partition.Horz:
		// Two non-square blocks: full-wide × half-tall.
		d.decodeCodingBlock16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, full, half, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)
		if row4+half < d.miH4 {
			d.decodeCodingBlock16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
				row4+half, col4, full, half, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)
		}

	case partition.Vert:
		// Two non-square blocks: half-wide × full-tall.
		d.decodeCodingBlock16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, half, full, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)
		if col4+half < d.miW4 {
			d.decodeCodingBlock16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
				row4, col4+half, half, full, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)
		}

	case partition.Split:
		// Four quadrants, each recursively decoded at bsLog2-1.
		d.decodePartition16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, bsLog2-1, defaultDC, defaultAC, lossless)
		d.decodePartition16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4+half, bsLog2-1, defaultDC, defaultAC, lossless)
		d.decodePartition16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4+half, col4, bsLog2-1, defaultDC, defaultAC, lossless)
		d.decodePartition16(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4+half, col4+half, bsLog2-1, defaultDC, defaultAC, lossless)
	}
}

func (d *Decoder) decodeCodingBlock16(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	fi blockmode.FrameInfo, dst *refbuf.Frame16, refs [3]*refbuf.Frame16,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsW4, bsH4, sub8x8 int,
	defaultDC, defaultAC [3]int16, lossless bool,
) {
	// libvpx decode_block: a block whose chroma size is not a valid block size
	// (4:2:2 / 4:4:0 only) makes the frame corrupt.
	if d.invalidBlock || invalidUVBlock(bsW4, bsH4, d.ssX, d.ssY) {
		d.invalidBlock = true
		return
	}
	if blkDumpW != nil {
		rng, top := br.State()
		fmt.Fprintf(blkDumpW, "%d %d %d %d %d %d t=%d bp=%d\n", blkFrame, row4/2, col4/2, bsW4, bsH4, rng, top, br.BitPos())
	}

	// Decode mode info.
	bm := blockmode.Decode(br, probs, counts, fi, aboveRow, leftCol, col4, row4, bsW4, bsH4, sub8x8, d.mvRefs, d.appendSub8x8)
	if bm.InvalidMV {
		d.invalidMV = true
	}

	if blkDumpW != nil {
		rng, _ := br.State()
		fmt.Fprintf(blkDumpW, "M %d %d %d inter=%v ref=%d,%d mode=%d mv=(%d,%d) skip=%v tx=%d filt=%d rng=%d\n",
			blkFrame, row4/2, col4/2, bm.IsInter, bm.RefFrame[0], bm.RefFrame[1], bm.InterMode,
			bm.MV[0].Row, bm.MV[0].Col, bm.Skip, bm.TxSize, bm.InterpFilter, rng)
	}

	if bm.Invalid {
		d.segInvalid = true
	}

	// Per-segment quantizer + LF level.
	dc, ac := defaultDC, defaultAC
	if qIdx := d.segQIndex(&fi, bm.SegID); qIdx != d.baseQIdx {
		dc[0], ac[0] = qStepsForPlaneBD(qIdx, d.deltaQYDC, 0, d.bitDepth)
		dc[1], ac[1] = qStepsForPlaneBD(qIdx, d.deltaQUVDC, d.deltaQUVAC, d.bitDepth)
		dc[2], ac[2] = dc[1], ac[1]
	}

	// Per-block loop filter level (spec §8.8.1).
	lfLevel := d.blockLFLevel(&fi, &bm)

	// Motion vectors are resolved in blockmode.Decode (NEAREST/NEAR/ZERO/NEW
	// from the find_mv_refs candidate list); use them directly.
	mv0, mv1 := bm.MV[0], bm.MV[1]

	// Reconstruct pixels.
	if blkDumpW != nil && row4 == 4 && col4 == 12 {
		rrng, rtop := br.State()
		fmt.Fprintf(blkDumpW, "RES go residual-start bp=%d rng=%d t=%d skip=%v tx=%d isInter=%v\n", br.BitPos(), rrng, rtop, bm.Skip, bm.TxSize, bm.IsInter)
	}
	eobTotal := reconstruct.Block16(br, probs, counts, bm, lossless, mv0, mv1, dc, ac,
		dst, refs, aboveRow, leftCol, row4, col4, bsW4, bsH4)
	// libvpx decode_block: "if (!less8x8 && eobtotal == 0) mi->skip = 1; // skip
	// loopfilter". An inter block of 8x8 or more that was coded as not skipped
	// but carries no coefficient at all counts as skipped from here on: for the
	// loop filter and, because the flag lives in the mode info, for the skip
	// and tx-size contexts of the blocks decoded after it. (A libvpx encoder
	// never codes such a block; other encoders and corrupt streams do.)
	if bm.IsInter && !bm.Skip && sub8x8 == blockmode.Sub8x8None && eobTotal == 0 {
		bm.Skip = true
	}
	if blkDumpW != nil && row4 == 4 && col4 == 12 {
		fmt.Fprintf(blkDumpW, "RES go residual-end bp=%d\n", br.BitPos())
	}

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

	// Sub-8×8 intra: the above/left mode context for the keyframe sub-mode tree
	// reads the neighbour's EDGE sub-mode, not its single block mode (libvpx
	// vp9_above_block_mode → bmi[b+2], vp9_left_block_mode → bmi[b+1]). Store the
	// bottom row (bmi[2],bmi[3]) in the above context and the right column
	// (bmi[1],bmi[3]) in the left context, per 4×4 position.
	if bm.NumSubY > 1 {
		if col4 < aboveRow.N4 {
			aboveRow.YMode[col4] = uint8(bm.SubYModes[2])
		}
		if col4+1 < aboveRow.N4 {
			aboveRow.YMode[col4+1] = uint8(bm.SubYModes[3])
		}
		if row4 < leftCol.N4 {
			leftCol.YMode[row4] = uint8(bm.SubYModes[1])
		}
		if row4+1 < leftCol.N4 {
			leftCol.YMode[row4+1] = uint8(bm.SubYModes[3])
		}
	}

	// Update partition context (libvpx partition_context_lookup, indexed by the
	// SUBSIZE: above byte depends on subsize width, left byte on its height —
	// byte = (0xF << log2(side4)) & 0xF). For 8×8 sub-partitions the subsize is
	// the sub-prediction block (8×4/4×8/4×4), not the 8×8 cell.
	subW4, subH4 := bsW4, bsH4
	switch sub8x8 {
	case blockmode.Sub8x8Horz: // BLOCK_8X4
		subW4, subH4 = 2, 1
	case blockmode.Sub8x8Vert: // BLOCK_4X8
		subW4, subH4 = 1, 2
	case blockmode.Sub8x8Split: // BLOCK_4X4
		subW4, subH4 = 1, 1
	}
	aboveMask := uint8((0xF << log2n4(subW4)) & 0xF)
	leftMask := uint8((0xF << log2n4(subH4)) & 0xF)
	n8W := bsW4 / 2
	if n8W < 1 {
		n8W = 1
	}
	n8H := bsH4 / 2
	if n8H < 1 {
		n8H = 1
	}
	aboveRow.SetPartition(col4/2, n8W, aboveMask)
	leftCol.SetPartition(row4/2, n8H, leftMask)

	// Store MI info for MV prediction and LF.
	d.setMI(row4, col4, bsW4, bsH4, bm, mv0, mv1, lfLevel)
}

func qStepsForPlaneBD(qIdx, dcDelta, acDelta, bd int) (dc, ac int16) {
	clamp := func(q int) int {
		if q < 0 {
			return 0
		}
		if q > 255 {
			return 255
		}
		return q
	}
	return dequant.DCQuantBD(clamp(qIdx+dcDelta), bd), dequant.ACQuantBD(clamp(qIdx+acDelta), bd)
}


// refSize16 returns the coded size of the frame in reference slot 0..7 of the
// 16-bit reference buffer, or 0, 0 when the slot is empty.
func (d *Decoder) refSize16(slot int) (w, h int) {
	if d.refBuf16.Get(slot) == nil {
		return 0, 0
	}
	return d.slotW[slot&7], d.slotH[slot&7]
}

// refSlots16 describes the 16-bit reference slots to beginFrame.
func (d *Decoder) refSlots16() refSlots {
	return refSlots{
		size:  d.refSize16,
		clear: func() { d.refBuf16 = refbuf.Buffer16{} },
		format: func(slot int) (int, int, int) {
			f := d.refBuf16.Get(slot)
			if f == nil {
				return -1, -1, 0
			}
			return f.SubX, f.SubY, f.BitDepth
		},
		depth: d.bitDepth,
	}
}
