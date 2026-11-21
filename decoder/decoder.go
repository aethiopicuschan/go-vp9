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
//	    frame, err := d.DecodeFrame(payload)   // DecodeFrame16 for profile 2/3
//	    if err != nil { ... }                  // see "Errors" below
//	    if frame == nil { continue }           // packet shows no frame
//
// # Output
//
// The decoder follows libvpx (vpx_codec_decode + vpx_codec_get_frame) packet
// by packet: same accept/reject verdict, same picture, same "no output".
//
//   - A packet outputs at most one frame. DecodeFrame returns (nil, nil) for
//     a packet that decodes but shows nothing: one whose last frame is hidden
//     (an alt-ref or intra-only frame alone in its packet), or a
//     show_existing_frame received while the decoder waits for a key frame
//     after an error.
//   - Frame.Width and Frame.Height are the true picture size, which may be
//     odd. The chroma planes are UVWidth() x UVHeight() samples, i.e.
//     (Width+SubX)>>SubX by (Height+SubY)>>SubY. Profile 1 and 3 streams are
//     not 4:2:0: always use SubX/SubY or UVWidth/UVHeight, never Width/2.
//   - Trailing bytes behind the last tile of a plain packet are skipped while
//     they are zero and otherwise decoded as a further frame, as libvpx does.
//
// # Errors
//
// After any decode error the decoder holds no usable references. Every
// following packet fails with ErrNeedResync until a key frame or an
// intra-only frame arrives; use errors.Is(err, ErrNeedResync) to request a
// key frame or simply keep feeding packets.
//
// Frames wider or taller than DefaultMaxFrameSize (16384) are rejected with an
// error wrapping ErrFrameSize before anything is allocated; SetMaxFrameSize
// changes the limit. Inter frames whose references have a different size
// (reference scaling) are not supported and are rejected.
//
// # Concurrency
//
// A Decoder is not safe for concurrent use. Different Decoder instances may
// be used from different goroutines and may be interleaved freely, but frame
// decoding is serialised process-wide (the lower-level packages keep
// per-frame parameters in package variables), so decoders do not run in
// parallel.
package decoder

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/aethiopicuschan/go-vp9/blockmode"
	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/compheader"
	"github.com/aethiopicuschan/go-vp9/ctx"
	"github.com/aethiopicuschan/go-vp9/dequant"
	"github.com/aethiopicuschan/go-vp9/frame"
	"github.com/aethiopicuschan/go-vp9/loopfilter"
	"github.com/aethiopicuschan/go-vp9/mv"
	"github.com/aethiopicuschan/go-vp9/mvpred"
	"github.com/aethiopicuschan/go-vp9/partition"
	"github.com/aethiopicuschan/go-vp9/reconstruct"
	"github.com/aethiopicuschan/go-vp9/refbuf"
	"github.com/aethiopicuschan/go-vp9/tile"
)

// kfPartitionProbs is libvpx vp9_kf_partition_probs — the fixed partition
// probability table used for keyframe and intra-only frames (spec §9.3.1).
// Indexed by partition context (0..15): 4 block-size levels × 4 above/left
// split combinations. Rows for ctx 12..15 are the 64×64→32×32 level.
var kfPartitionProbs = [16][3]uint8{
	{158, 97, 94}, {93, 24, 99}, {85, 119, 44}, {62, 59, 67},
	{149, 53, 53}, {94, 20, 48}, {83, 53, 24}, {52, 18, 18},
	{150, 40, 39}, {78, 12, 26}, {67, 33, 11}, {24, 7, 5},
	{174, 35, 49}, {68, 11, 27}, {57, 15, 9}, {12, 3, 3},
}

// miBlock stores per-MI-block decoded state needed for MV prediction and
// loop filter level lookup of subsequent blocks.
type miBlock struct {
	inter    bool
	ref      [2]int
	mv       [2]mv.MV
	subMV    [4][2]mv.MV // per-4×4 sub-block MVs (bmi); all = mv for non-sub-8×8
	sub8x8   bool        // sb_type < BLOCK_8X8 (per-sub MVs differ)
	zeroMV   bool
	mode     uint8 // libvpx-encoded mode (intra 0..9; inter NEARESTMV(10)+InterMode) for MV-ref search
	lfLevel  int   // per-block loop filter level (0..63)
	txSize   uint8 // luma transform size (TX_4X4..TX_32X32) for loop-filter edges
	uvTxSize uint8 // chroma transform size for loop-filter edges
	skipLF   bool  // skip && inter: suppress internal TX-edge filtering
	bRow4    int   // this block's top-left position (4×4 units) — for the loop
	bCol4    int   // filter's prediction-vs-internal edge distinction
}

// Decoder holds VP9 decode state across frames.
type Decoder struct {
	probs         *compheader.Probs    // working fc for the frame in progress
	frameContexts [4]*compheader.Probs // saved frame contexts (frame_context_idx)
	refBuf        refbuf.Buffer
	refBuf16      refbuf.Buffer16 // high-bit-depth (10/12-bit) reference slots
	bitDepth      int             // current frame bit depth (for highbd quant)

	// Per-frame quantizer deltas (stored for per-block segment Q recompute).
	baseQIdx   int
	deltaQYDC  int
	deltaQUVDC int
	deltaQUVAC int

	// Per-frame MV-reference search parameters.
	signBias [4]bool // ref_frame_sign_bias (index ctx.RefLast/Golden/AltRef)
	allowHP  bool    // high-precision MV allowed this frame

	// last_frame_type == KEY_FRAME, for coefficient-prob adaptation speed
	// (libvpx COEF_MAX_UPDATE_FACTOR_AFTER_KEY).
	lastFrameKey bool

	// lastShown: whether decodeOneFrame's last frame is displayable (show_frame
	// or show_existing). DecodeFrame uses it to pick the shown frame from a
	// superframe (skipping invisible alt-ref frames).
	lastShown bool

	// Temporal MV prediction (use_prev_frame_mvs): the decode-order previous
	// frame's MI grid and the metadata that gates whether its co-located MVs may
	// be used as candidates (libvpx cm->prev_frame->mvs + cm->use_prev_frame_mvs).
	usePrevFrameMVs    bool
	prevFrameMI        []miBlock
	prevFrameW4        int
	prevFrameWidth     int
	prevFrameHeight    int
	prevFrameIntraOnly bool
	prevFrameShow      bool

	// Segmentation state carried across frames, and the per-frame flag for a
	// block that used the segment SKIP feature illegally (see segmap.go).
	seg        segState
	segInvalid bool

	// Per-frame loop filter state (for per-block level computation).
	lfBaseLevel    int
	lfSharpness    int
	lfDeltaEnabled bool
	lfRefDeltas    [4]int
	lfModeDeltas   [2]int

	// Stored frame dimensions (updated on key frame and intra-only inter frame).
	frameWidth, frameHeight int

	// Chroma subsampling shifts of the sequence (subsampling_x/y; 1,1 = 4:2:0),
	// updated on key and intra-only frames.
	ssX, ssY int
	// invalidBlock: a block of the frame in progress had a chroma size that is
	// not a valid block size (corrupt frame).
	invalidBlock bool

	// Coded size of the frame in each reference slot (frame_size_with_refs).
	slotW, slotH [8]int

	// needResync: a frame failed to decode; until a key frame or intra-only
	// frame arrives every frame is rejected (libvpx pbi->need_resync).
	needResync bool

	// fcInit: frame context i has been set up (libvpx fc->initialized).
	fcInit [4]bool

	// invalidMV: a block of the current frame decoded a motion vector outside
	// the valid range (blockmode.BlockMode.InvalidMV); the frame is rejected.
	invalidMV bool

	// Frame size limit (SetMaxFrameSize); 0 = DefaultMaxFrameSize.
	maxWidth, maxHeight int

	// Per-frame MI grid (row-major, W4 × H4).
	miW4, miH4 int
	miGrid     []miBlock

	// Current tile's column bounds (4×4 units), set per tile column. The MV
	// reference search restricts spatial neighbours to the tile columns
	// (libvpx is_inside uses mi_col_start/mi_col_end; rows stay frame-relative).
	curTileColStart4, curTileColEnd4 int

	// Temporal MV buffers: per-slot saved MI grids from previous frames.
	prevMIW4    [8]int
	prevMIH4    [8]int
	prevMIGrids [8][]miBlock
}

// New creates a Decoder with default probability tables.
func New() *Decoder {
	d := &Decoder{
		probs: compheader.DefaultProbs(),
		ssX:   1,
		ssY:   1,
	}
	for i := range d.frameContexts {
		d.frameContexts[i] = compheader.DefaultProbs()
	}
	return d
}

// resetFrameContext loads the default probabilities into saved context i.
func (d *Decoder) resetFrameContext(i int) {
	d.frameContexts[i] = compheader.DefaultProbs()
	d.fcInit[i] = true
}

// NewWithProbs creates a Decoder with pre-populated probability tables (for testing).
func NewWithProbs(probs *compheader.Probs) *Decoder {
	return &Decoder{probs: probs, ssX: 1, ssY: 1}
}

// decodeMu serialises frame decoding across all Decoder instances. Several
// lower-level packages keep per-frame parameters in package variables (the MI
// grid size and tile origin in reconstruct, the bit depth in coeftoken,
// transform, intra16, mc16 and loopfilter); each frame sets them before use,
// so decoders may be interleaved freely, and the lock makes it safe to run
// them from different goroutines (they then take turns, one packet at a time).
var decodeMu sync.Mutex

// DecodeFrame decodes one IVF packet payload (not including the 12-byte IVF
// frame header) and returns the frame the packet outputs, as 8-bit planes
// (its SubX/SubY give the chroma subsampling: 4:2:0 for profile 0, any format
// for profile 1; Width/Height are the true, possibly odd, picture size and
// UVWidth()/UVHeight() the chroma plane size). A packet may be a VP9
// superframe - several coded frames (e.g. an invisible alt-ref plus the shown
// frame) with a trailing index; each is decoded in turn.
//
// Like libvpx's vpx_codec_get_frame, a packet outputs at most one frame, and
// the result is (nil, nil) when it decodes without error but shows nothing:
// its last frame is a hidden one (a lone alt-ref or intra-only frame in its
// own packet), or it is a show_existing_frame arriving while the decoder is
// still waiting for a key or intra-only frame after an error.
//
// After an error every packet is rejected with ErrNeedResync until a key
// frame or an intra-only frame arrives (see the package documentation).
func (d *Decoder) DecodeFrame(payload []byte) (*refbuf.Frame, error) {
	decodeMu.Lock()
	defer decodeMu.Unlock()
	var shown *refbuf.Frame
	// Packet structure (superframe index, trailing data) and the error state
	// follow libvpx: see decodePacket.
	err := d.decodePacket(payload, func(sub []byte) (int, error) {
		f, end, err := d.decodeOneFrame(sub)
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

// splitSuperframe returns the coded frames inside a VP9 superframe. A superframe
// ends with an index: a marker byte (top 3 bits 0b110) bracketing the per-frame
// sizes. A normal (single-frame) packet has no such index and is returned as-is.
func splitSuperframe(data []byte) [][]byte {
	single := [][]byte{data}
	if len(data) < 2 {
		return single
	}
	marker := data[len(data)-1]
	if marker&0xe0 != 0xc0 {
		return single
	}
	bytesPerSize := int((marker>>3)&0x3) + 1
	nFrames := int(marker&0x7) + 1
	indexSz := 2 + nFrames*bytesPerSize
	if len(data) < indexSz || data[len(data)-indexSz] != marker {
		return single // not a valid index — treat as one frame
	}
	frames := make([][]byte, 0, nFrames)
	off, p := 0, len(data)-indexSz+1
	for i := 0; i < nFrames; i++ {
		sz := 0
		for j := 0; j < bytesPerSize; j++ {
			sz |= int(data[p]) << (8 * j)
			p++
		}
		if off+sz > len(data)-indexSz {
			return single // malformed sizes
		}
		frames = append(frames, data[off:off+sz])
		off += sz
	}
	return frames
}

// decodeOneFrame decodes a single coded VP9 frame and records in d.lastShown
// whether it is the displayable frame (vs. an invisible reference like alt-ref).
// blkDumpW/blkFrame: per-block bool-coder-state trace (env BLK_DUMP_GO) to
// oracle-walk inter frames against libvpx (env BLK_DUMP).
var (
	blkDumpW io.Writer
	blkFrame = -1
)

func (d *Decoder) decodeOneFrame(payload []byte) (*refbuf.Frame, int, error) {
	// 1. Uncompressed header.
	h, err := frame.ParseRefs(payload, d.refSize)
	if err != nil {
		d.rejectedHeader(h, d.refSlots())
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
		f := d.refBuf.Get(h.ShowExistingIdx)
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
	if h.BitDepth != 8 && h.BitDepth != 0 {
		return nil, 0, errors.New("vp9/decoder: only 8-bit frames supported")
	}

	// Header checks that depend on the decoder state (resync after an error,
	// reference slots, header length): see beginFrame.
	if err := d.beginFrame(h, len(payload), d.refSlots()); err != nil {
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
	dst := refbuf.NewFrameSS(fw, fh, d.ssX, d.ssY)
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
			d.refBuf.SetRef(i, idx)
		}
	}

	// Gather reference frames.
	refs := [3]*refbuf.Frame{
		d.refBuf.Last(),
		d.refBuf.Golden(),
		d.refBuf.AltRef(),
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
	dcY, acY := qStepsForPlane(h.BaseQIdx, h.DeltaQYDC, 0)
	dcUV, acUV := qStepsForPlane(h.BaseQIdx, h.DeltaQUVDC, h.DeltaQUVAC)
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
					d.decodePartition(brs[tc], decodeProbs, counts, fi, dst, refs,
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
		loopfilter.FilterFrame(dst.Y[dst.YBase():], dst.YStride, yW4, yH4, 16, sharpness, yInfo)
		switch {
		case dst.SubX == 1 && dst.SubY == 1:
			loopfilter.DumpPlane = 1
			loopfilter.FilterFrame(dst.U[dst.UBase():], dst.UVStride, uvW4, uvH4, 8, sharpness, uvInfo)
			loopfilter.DumpPlane = 2
			loopfilter.FilterFrame(dst.V[dst.VBase():], dst.UVStride, uvW4, uvH4, 8, sharpness, uvInfo)
		case dst.SubX == 0 && dst.SubY == 0:
			// 4:4:4: libvpx filters the chroma planes with the luma masks and
			// levels (vp9_filter_block_plane_ss00 for every plane).
			loopfilter.DumpPlane = 1
			loopfilter.FilterFrame(dst.U[dst.UBase():], dst.UVStride, yW4, yH4, 16, sharpness, yInfo)
			loopfilter.DumpPlane = 2
			loopfilter.FilterFrame(dst.V[dst.VBase():], dst.UVStride, yW4, yH4, 16, sharpness, yInfo)
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
			for _, buf := range [][]byte{dst.U[dst.UBase():], dst.V[dst.VBase():]} {
				loopfilter.FilterFrameNon420(dst.UVStride, miRows, miCols, dst.SubX, dst.SubY, sharpness, miInfo,
					func(q0, step, width int, p loopfilter.Params) {
						loopfilter.FilterEdge4(buf, q0, step, width, p)
					})
			}
		}
		loopfilter.DumpW = nil
	}

	// 8. Update refbuf. (The LAST/GOLDEN/ALTREF mapping for the next frame is set
	// at the start of its own decode from its ref_frame_idx, above.)
	refbuf.ExtendBorders(dst)
	d.refBuf.Update(dst, h.RefreshFrameFlags)
	for i := 0; i < 8; i++ {
		if h.RefreshFrameFlags&(1<<uint(i)) != 0 {
			d.slotW[i], d.slotH[i] = fw, fh
		}
	}

	d.segEndFrame(h)

	d.lastShown = h.ShowFrame
	return dst, frameEnd, nil
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

	// Tile-relative neighbour availability: a left/above neighbour is only
	// available (entropy context AND intra prediction) if it lies inside the tile.
	fi.MiColStart = tileCol4
	fi.MiRowStart = tileRow4
	reconstruct.TileColStart4 = tileCol4
	reconstruct.TileRowStart4 = tileRow4

	// One left-context column for the whole tile height: each SB row uses a
	// disjoint absolute-row4 range, so this is an implicit per-SB-row reset
	// (the left context is indexed by absolute row4, not SB-relative). Size it to
	// cover the absolute row range [tileRow4, tileRow4+tileH4) for tile rows > 0.
	leftCol := ctx.NewCol(tileRow4 + tileH4)
	for sbRow4 := tileRow4; sbRow4 < tileRow4+tileH4; sbRow4 += sbN4 {
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
		d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, full, full, int(pt), defaultDC, defaultAC, lossless)
		return
	}

	switch pt {
	case partition.None:
		// Square block: full × full.
		d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, full, full, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)

	case partition.Horz:
		// Two non-square blocks: full-wide × half-tall.
		d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, full, half, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)
		if row4+half < d.miH4 {
			d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
				row4+half, col4, full, half, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)
		}

	case partition.Vert:
		// Two non-square blocks: half-wide × full-tall.
		d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
			row4, col4, half, full, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)
		if col4+half < d.miW4 {
			d.decodeCodingBlock(br, probs, counts, fi, dst, refs, aboveRow, leftCol,
				row4, col4+half, half, full, blockmode.Sub8x8None, defaultDC, defaultAC, lossless)
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

// decodeCodingBlock decodes one leaf coding block. sub8x8 is the sub-8×8
// prediction layout for an 8×8 block (blockmode.Sub8x8*); pass
// blockmode.Sub8x8None for all other sizes.
func (d *Decoder) decodeCodingBlock(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	fi blockmode.FrameInfo, dst *refbuf.Frame, refs [3]*refbuf.Frame,
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
		dc[0], ac[0] = qStepsForPlane(qIdx, d.deltaQYDC, 0)
		dc[1], ac[1] = qStepsForPlane(qIdx, d.deltaQUVDC, d.deltaQUVAC)
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
	eobTotal := reconstruct.Block(br, probs, counts, bm, lossless, mv0, mv1, dc, ac,
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

// invalidUVBlock reports whether a coding block of bsW4×bsH4 luma 4×4 units
// (8×8 or larger) has a chroma block that is not a valid VP9 block size
// (libvpx ss_size_lookup == BLOCK_INVALID): a 1:2 block subsampled along its
// short side only, e.g. 8×16 in 4:2:2 or 16×8 in 4:4:0.
func invalidUVBlock(bsW4, bsH4, ssX, ssY int) bool {
	w, h := bsW4>>uint(ssX), bsH4>>uint(ssY)
	return w > 2*h || h > 2*w
}

// uvLFTxSize returns the chroma transform size for a luma block of txSize yTx
// and size bsW4×bsH4 (4×4 units) with chroma subsampling shifts ssX, ssY —
// matches reconstruct.UVTxSize.
func uvLFTxSize(yTx uint8, bsW4, bsH4, ssX, ssY int) uint8 {
	uvW4 := bsW4 >> uint(ssX)
	if uvW4 < 1 {
		uvW4 = 1
	}
	uvH4 := bsH4 >> uint(ssY)
	if uvH4 < 1 {
		uvH4 = 1
	}
	m := uvW4
	if uvH4 < m {
		m = uvH4
	}
	var maxUV uint8
	for (1 << (maxUV + 1)) <= m {
		maxUV++
	}
	if yTx < maxUV {
		return yTx
	}
	return maxUV
}

// setMI stores the decoded MI block info for MV prediction and LF.
func (d *Decoder) setMI(row4, col4, bsW4, bsH4 int, bm blockmode.BlockMode, mv0, mv1 mv.MV, lfLevel int) {
	mode := uint8(bm.YMode)
	if bm.IsInter {
		mode = 10 + uint8(bm.InterMode) // NEARESTMV(10)+InterMode
	}
	var subMV [4][2]mv.MV
	if bm.NumSubMV > 0 {
		subMV = bm.SubMV
	} else {
		subMV = [4][2]mv.MV{{mv0, mv1}, {mv0, mv1}, {mv0, mv1}, {mv0, mv1}}
	}
	mi := miBlock{
		inter:    bm.IsInter,
		ref:      bm.RefFrame,
		mv:       [2]mv.MV{mv0, mv1},
		subMV:    subMV,
		sub8x8:   bm.NumSubMV > 0,
		zeroMV:   mv0.Row == 0 && mv0.Col == 0,
		mode:     mode,
		lfLevel:  lfLevel,
		txSize:   bm.TxSize,
		uvTxSize: uvLFTxSize(bm.TxSize, bsW4, bsH4, d.ssX, d.ssY),
		skipLF:   bm.Skip && bm.IsInter,
		bRow4:    row4,
		bCol4:    col4,
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
		// Intra-only frames code mode info exactly like key frames (libvpx
		// vp9_read_mode_info: frame_is_intra_only → read_intra_frame_mode_info).
		IsKeyframe:    h.FrameType == frame.FrameKey || h.IntraOnly,
		AllowHP:       h.AllowHighPrecisionMV,
		ReferenceMode: d.probs.ReferenceMode,
		InterpFilter:  h.InterpolationFilter,
		// Segmentation fields are filled by segBeginFrame.
	}
	if !fi.IsKeyframe {
		fi.CompFixedRef, fi.CompVarRef = blockmode.CompFixedVarRef(h.RefFrameSign)
		// Slot for the fixed ref = its sign bias (ref index = ctx.Ref* − 1).
		if h.RefFrameSign[fi.CompFixedRef-1] {
			fi.CompFixedIdx = 1
		}
	}
	return fi
}

// refSize returns the coded size of the frame in reference slot 0..7 of the
// 8-bit reference buffer, or 0, 0 when the slot is empty.
func (d *Decoder) refSize(slot int) (w, h int) {
	if d.refBuf.Get(slot) == nil {
		return 0, 0
	}
	return d.slotW[slot&7], d.slotH[slot&7]
}

// refSlots describes the 8-bit reference slots to beginFrame.
func (d *Decoder) refSlots() refSlots {
	return refSlots{
		size:  d.refSize,
		clear: func() { d.refBuf = refbuf.Buffer{} },
		format: func(slot int) (int, int, int) {
			f := d.refBuf.Get(slot)
			if f == nil {
				return -1, -1, 0
			}
			return f.SubX, f.SubY, 8
		},
		depth: 8,
	}
}
