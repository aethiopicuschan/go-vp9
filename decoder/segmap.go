package decoder

import (
	"github.com/aethiopicuschan/go-vp9/blockmode"
	"github.com/aethiopicuschan/go-vp9/frame"
	"github.com/aethiopicuschan/go-vp9/loopfilter"
	"github.com/aethiopicuschan/go-vp9/segment"
)

// segState is the segmentation state that outlives a frame (libvpx cm->seg
// plus the two ping-pong segmentation maps), shared by the 8-bit and the
// high-bit-depth decode paths.
type segState struct {
	// Feature data: replaced only by a frame coding segmentation_update_data,
	// cleared by vp9_setup_past_independence; it is kept (unused) across frames
	// that have segmentation disabled.
	featEnabled [8][4]bool
	featData    [8][4]int16
	absDelta    bool // segmentation_abs_or_delta_update of the last data update

	// maps[cur] is current_frame_seg_map, maps[cur^1] last_frame_seg_map, one
	// byte per 8×8 MI in raster order with stride miCols. The buffers are only
	// reallocated when the frame grows (libvpx seg_map_alloc_size), so the
	// content otherwise survives a size change exactly as it does there.
	maps           [2][]uint8
	cur            int
	miCols, miRows int
	width, height  int

	predFlags []bool // this frame's per-MI seg_id_predicted
}

// segBeginFrame applies libvpx's per-frame segmentation bookkeeping in header
// order — context-buffer resize (setup_frame_size), vp9_setup_past_independence
// and setup_segmentation — and fills the segmentation part of fi.
func (d *Decoder) segBeginFrame(h *frame.Header, fi *blockmode.FrameInfo, fw, fh int) {
	s := &d.seg
	frameIsIntra := h.FrameType == frame.FrameKey || h.IntraOnly

	// resize_context_buffers: on a size change the last map is cleared; both
	// are reallocated (zeroed, indices reset) only when they are too small.
	if fw != s.width || fh != s.height {
		miCols, miRows := (fw+7)>>3, (fh+7)>>3
		if (miCols > s.miCols || miRows > s.miRows) && len(s.maps[0]) < miCols*miRows {
			s.maps[0] = make([]uint8, miCols*miRows)
			s.maps[1] = make([]uint8, miCols*miRows)
			s.cur = 0
		}
		s.miCols, s.miRows = miCols, miRows
		s.width, s.height = fw, fh
		clear(s.maps[s.cur^1][:miCols*miRows])
	}
	n := s.miCols * s.miRows

	// vp9_setup_past_independence: features off, delta coding, both maps zero.
	if frameIsIntra || h.ErrorResilient {
		s.featEnabled = [8][4]bool{}
		s.featData = [8][4]int16{}
		s.absDelta = false
		clear(s.maps[0][:n])
		clear(s.maps[1][:n])
	}

	// setup_segmentation: a data update replaces all feature data (the header
	// parser starts from cleared features, as vp9_clearall_segfeatures does).
	if h.SegEnabled && h.SegUpdateData {
		s.featEnabled = h.SegFeatEnabled
		s.featData = h.SegFeatData
		s.absDelta = h.SegAbsOrDelta
	}

	if len(s.predFlags) < n {
		s.predFlags = make([]bool, n)
	} else {
		clear(s.predFlags[:n])
	}

	fi.SegEnabled = h.SegEnabled
	fi.SegUpdateMap = h.SegUpdateMap
	fi.SegTempUpdate = h.SegTempUpdate
	fi.SegTreeProbs = h.SegTreeProbs
	fi.SegPredProbs = h.SegPredProbs
	fi.SegFeatEnabled = s.featEnabled
	fi.SegFeatData = s.featData
	fi.SegAbsOrDelta = s.absDelta
	fi.FrameIsIntra = frameIsIntra
	fi.SegMapPrev = s.maps[s.cur^1]
	fi.SegMapCur = s.maps[s.cur]
	fi.SegPredFlags = s.predFlags
	fi.SegMiCols = s.miCols
	fi.SegMiRows = s.miRows
}

// segEndFrame swaps the current and last segmentation maps after a decoded
// (shown or hidden) frame that had segmentation enabled (libvpx
// vp9_receive_compressed_data: vp9_swap_current_and_last_seg_map). A frame
// with segmentation disabled leaves both maps as they are.
func (d *Decoder) segEndFrame(h *frame.Header) {
	if h.SegEnabled {
		d.seg.cur ^= 1
	}
}

// segQIndex returns the block's quantizer index (libvpx vp9_get_qindex).
func (d *Decoder) segQIndex(fi *blockmode.FrameInfo, segID int) int {
	if !fi.SegEnabled {
		return d.baseQIdx
	}
	return segment.ClampQIndex(d.baseQIdx,
		segment.ParamsFor(fi.SegFeatEnabled, fi.SegFeatData, fi.SegAbsOrDelta, segID))
}

// blockLFLevel returns the block's loop-filter level (libvpx
// vp9_loop_filter_frame_init + get_filter_level): the level is looked up by
// segment id, reference frame and mode_lf_lut[mode], whose only distinction is
// ZEROMV (delta 0) against the other inter modes (delta 1).
func (d *Decoder) blockLFLevel(fi *blockmode.FrameInfo, bm *blockmode.BlockMode) int {
	segLF := fi.SegEnabled && fi.SegFeatEnabled[bm.SegID][segment.FeatAltLF]
	return loopfilter.LevelForBlock(
		d.lfBaseLevel,
		segLF, fi.SegFeatData[bm.SegID][segment.FeatAltLF], fi.SegAbsOrDelta,
		d.lfDeltaEnabled,
		bm.RefFrame[0], bm.IsInter, bm.InterMode == blockmode.ZeroMV,
		d.lfRefDeltas, d.lfModeDeltas,
	)
}
