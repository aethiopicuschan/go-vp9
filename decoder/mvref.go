package decoder

import (
	"github.com/mgvs/go-vp9/blockmode"
	"github.com/mgvs/go-vp9/ctx"
	"github.com/mgvs/go-vp9/mv"
)

// VP9 motion-vector reference search (libvpx vp9/common/vp9_mvref_common.{c,h}):
// the inter-mode context plus the two best reference MV candidates (nearest,
// near) for an inter block. Spatial candidates only for now (temporal/prev-frame
// MVs are deferred; use_prev_frame_mvs is false for the first inter frame).

// mvRefBlocks[BLOCK_SIZE][MVREF_NEIGHBOURS] = {row, col} MI offsets in libvpx
// scan order. Index by blockSizeIdx(bsW4, bsH4).
var mvRefBlocks = [13][8][2]int8{
	{{-1, 0}, {0, -1}, {-1, -1}, {-2, 0}, {0, -2}, {-2, -1}, {-1, -2}, {-2, -2}}, // 4X4
	{{-1, 0}, {0, -1}, {-1, -1}, {-2, 0}, {0, -2}, {-2, -1}, {-1, -2}, {-2, -2}}, // 4X8
	{{-1, 0}, {0, -1}, {-1, -1}, {-2, 0}, {0, -2}, {-2, -1}, {-1, -2}, {-2, -2}}, // 8X4
	{{-1, 0}, {0, -1}, {-1, -1}, {-2, 0}, {0, -2}, {-2, -1}, {-1, -2}, {-2, -2}}, // 8X8
	{{0, -1}, {-1, 0}, {1, -1}, {-1, -1}, {0, -2}, {-2, 0}, {-2, -1}, {-1, -2}},  // 8X16
	{{-1, 0}, {0, -1}, {-1, 1}, {-1, -1}, {-2, 0}, {0, -2}, {-1, -2}, {-2, -1}},  // 16X8
	{{-1, 0}, {0, -1}, {-1, 1}, {1, -1}, {-1, -1}, {-3, 0}, {0, -3}, {-3, -3}},   // 16X16
	{{0, -1}, {-1, 0}, {2, -1}, {-1, -1}, {-1, 1}, {0, -3}, {-3, 0}, {-3, -3}},   // 16X32
	{{-1, 0}, {0, -1}, {-1, 2}, {-1, -1}, {1, -1}, {-3, 0}, {0, -3}, {-3, -3}},   // 32X16
	{{-1, 1}, {1, -1}, {-1, 2}, {2, -1}, {-1, -1}, {-3, 0}, {0, -3}, {-3, -3}},   // 32X32
	{{0, -1}, {-1, 0}, {4, -1}, {-1, 2}, {-1, -1}, {0, -3}, {-3, 0}, {2, -1}},    // 32X64
	{{-1, 0}, {0, -1}, {-1, 4}, {2, -1}, {-1, -1}, {-3, 0}, {0, -3}, {-1, 2}},    // 64X32
	{{-1, 3}, {3, -1}, {-1, 4}, {4, -1}, {-1, -1}, {-1, 0}, {0, -1}, {-1, 6}},    // 64X64
}

// mode2counter maps a candidate's prediction mode to its context-counter
// contribution (libvpx mode_2_counter): intra → 9, NEARESTMV/NEARMV → 0,
// ZEROMV → 3, NEWMV → 1. miBlock.mode is the libvpx-encoded mode
// (intra: 0..9; inter: NEARESTMV(10)+InterMode).
var mode2counter = [14]int{9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1}

// counterToContext maps the accumulated context counter to the inter-mode
// context (libvpx counter_to_context). The 9 (INVALID_CASE) entries are unused.
var counterToContext = [19]int{2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6}

// blockSizeIdx maps (bsW4, bsH4) in {1,2,4,8,16} to the BLOCK_SIZE enum (0..12).
func blockSizeIdx(bsW4, bsH4 int) int {
	wl, hl := log2n4(bsW4), log2n4(bsH4)
	if wl == hl {
		return 3 * wl
	}
	if wl < hl {
		return 3*wl + 1
	}
	return 3*hl + 2
}

// mvBorder is libvpx MV_BORDER (16 px in 1/8-pel units).
const mvBorder = 16 << 3

// findMVRefs runs the VP9 MV-reference search for the block at 4×4 position
// (row4,col4) of size (bsW4,bsH4) for reference frame refFrame. It returns the
// inter-mode probability context (counter_to_context over the first two
// neighbours' modes) and the two best clamped candidate MVs (nearest, near).
// signBias[ref] is the per-reference sign bias (index by ctx.RefLast/Golden/
// AltRef; [0]/intra unused). MI offsets are 8×8 units → doubled into the 4×4
// miGrid. Single-tile column bounds are assumed.
// idxNColToSub is libvpx idx_n_column_to_subblock: for a sub-8×8 candidate,
// selects which of its four bmi sub-blocks supplies the MV, indexed by the
// current block's sub-index and whether the neighbour shares our column.
var idxNColToSub = [4][2]int{{1, 2}, {1, 3}, {3, 2}, {3, 3}}

// getSubBlockMV returns a candidate neighbour's MV for reference index which
// (libvpx get_sub_block_mv). When searching for a sub-8×8 prediction unit
// (block ≥ 0) and the neighbour is itself sub-8×8, the relevant bmi sub-block
// MV is used; otherwise the whole-block MV.
func getSubBlockMV(cand *miBlock, which, searchCol, block int) mv.MV {
	if block >= 0 && cand.sub8x8 {
		col0 := 0
		if searchCol == 0 {
			col0 = 1
		}
		return cand.subMV[idxNColToSub[block][col0]][which]
	}
	return cand.mv[which]
}

// findMVRefs runs the VP9 MV-reference search (libvpx find_mv_refs_idx). block
// is the sub-8×8 prediction-unit index (or -1 for a whole block). earlyBreak
// stops after the first candidate (NEARESTMV/NEWMV; libvpx ADD_MV_REF_LIST_EB).
// lowerPrec applies vp9_find_best_ref_mvs' precision-lowering (whole-block
// nearest/near; the per-sub append path passes false). Returns the inter-mode
// context, the (≤2) candidate MVs and how many were found.
func (d *Decoder) findMVRefs(row4, col4, bsW4, bsH4, refFrame, block int, earlyBreak, lowerPrec bool, signBias [4]bool, allowHP bool) (modeContext int, mvList [2]mv.MV, count int) {
	nbrs := &mvRefBlocks[blockSizeIdx(bsW4, bsH4)]
	contextCounter := 0
	differentRefFound := false
	finished := false

	inside := func(dr, dc int) (*miBlock, bool) {
		r := row4 + dr*2
		c := col4 + dc*2
		// Rows are frame-bounded; columns are tile-bounded (libvpx is_inside uses
		// mi_col_start/mi_col_end), so a neighbour in another tile column is unseen.
		if r < 0 || c < d.curTileColStart4 || r >= d.miH4 || c >= d.curTileColEnd4 {
			return nil, false
		}
		return &d.miGrid[r*d.miW4+c], true
	}
	// add implements libvpx ADD_MV_REF_LIST(_EB): first MV fills slot 0 (and
	// finishes immediately under earlyBreak); a second, distinct MV fills slot 1.
	add := func(m mv.MV) {
		if count > 0 {
			if m != mvList[0] {
				mvList[1] = m
				count = 2
				finished = true
			}
		} else {
			mvList[0] = m
			count = 1
			if earlyBreak {
				finished = true
			}
		}
	}
	scale := func(m mv.MV, candRef int) mv.MV {
		if signBias[candRef] != signBias[refFrame] {
			return mv.MV{Row: -m.Row, Col: -m.Col}
		}
		return m
	}
	// Co-located block in the decode-order previous frame (temporal candidate).
	var prevMI *miBlock
	if d.usePrevFrameMVs {
		if idx := row4*d.prevFrameW4 + col4; idx >= 0 && idx < len(d.prevFrameMI) {
			prevMI = &d.prevFrameMI[idx]
		}
	}

	// Same-reference spatial scan: first two neighbours use the sub-block MV and
	// count modes; the rest use the whole-block MV.
	for i := 0; i < 8 && !finished; i++ {
		cand, ok := inside(int(nbrs[i][0]), int(nbrs[i][1]))
		if !ok {
			continue
		}
		differentRefFound = true
		if i < 2 {
			contextCounter += mode2counter[cand.mode]
			searchCol := int(nbrs[i][1])
			if cand.ref[0] == refFrame {
				add(getSubBlockMV(cand, 0, searchCol, block))
			} else if cand.ref[1] == refFrame {
				add(getSubBlockMV(cand, 1, searchCol, block))
			}
		} else {
			if cand.ref[0] == refFrame {
				add(cand.mv[0])
			} else if cand.ref[1] == refFrame {
				add(cand.mv[1])
			}
		}
	}
	// Same-reference temporal candidate: the co-located previous-frame MV.
	if prevMI != nil && !finished {
		if prevMI.ref[0] == refFrame {
			add(prevMI.mv[0])
		} else if prevMI.ref[1] == refFrame {
			add(prevMI.mv[1])
		}
	}

	// Different-reference scan (sign-bias scaled, whole-block MVs).
	if differentRefFound && !finished {
		for i := 0; i < 8 && !finished; i++ {
			cand, ok := inside(int(nbrs[i][0]), int(nbrs[i][1]))
			if !ok || !cand.inter {
				continue
			}
			if cand.ref[0] != refFrame {
				add(scale(cand.mv[0], cand.ref[0]))
			}
			if !finished && cand.ref[1] > ctx.RefIntra && cand.ref[1] != refFrame &&
				cand.mv[1] != cand.mv[0] {
				add(scale(cand.mv[1], cand.ref[1]))
			}
		}
	}
	// Different-reference temporal candidate (sign-bias scaled).
	if prevMI != nil && !finished {
		if prevMI.ref[0] != refFrame && prevMI.ref[0] > ctx.RefIntra {
			add(scale(prevMI.mv[0], prevMI.ref[0]))
		}
		if !finished && prevMI.ref[1] > ctx.RefIntra && prevMI.ref[1] != refFrame &&
			prevMI.mv[1] != prevMI.mv[0] {
			add(scale(prevMI.mv[1], prevMI.ref[1]))
		}
	}

	modeContext = counterToContext[contextCounter]

	// Clamp candidates to the block's allowed MV range (libvpx clamp_mv_ref).
	minCol := -col4*32 - mvBorder
	maxCol := (d.miW4-bsW4-col4)*32 + mvBorder
	minRow := -row4*32 - mvBorder
	maxRow := (d.miH4-bsH4-row4)*32 + mvBorder
	for i := range mvList {
		mvList[i].Col = clampInt16(mvList[i].Col, minCol, maxCol)
		mvList[i].Row = clampInt16(mvList[i].Row, minRow, maxRow)
		if lowerPrec {
			mvList[i] = lowerMVPrecision(mvList[i], allowHP)
		}
	}
	return
}

// lowerMVPrecision rounds an MV to 1/4-pel when high precision is not used for
// it (libvpx lower_mv_precision).
func lowerMVPrecision(m mv.MV, allowHP bool) mv.MV {
	if allowHP && mv.UseHP(m) {
		return m
	}
	if m.Row&1 != 0 {
		if m.Row > 0 {
			m.Row--
		} else {
			m.Row++
		}
	}
	if m.Col&1 != 0 {
		if m.Col > 0 {
			m.Col--
		} else {
			m.Col++
		}
	}
	return m
}

func clampInt16(v int16, lo, hi int) int16 {
	x := int(v)
	if x < lo {
		x = lo
	}
	if x > hi {
		x = hi
	}
	return int16(x)
}

// mvRefs is the blockmode.Decode whole-block callback: the inter-mode context
// plus the two best precision-lowered candidate MVs (block=-1, no early break).
func (d *Decoder) mvRefs(refFrame, row4, col4, bsW4, bsH4 int) (int, [2]mv.MV) {
	ctxV, mvList, _ := d.findMVRefs(row4, col4, bsW4, bsH4, refFrame, -1, false, true, d.signBias, d.allowHP)
	return ctxV, mvList
}

// appendSub8x8 resolves the per-sub-block reference MV for a NEAREST/NEAR
// sub-8×8 prediction unit (libvpx decoder append_sub8x8_mvs_for_idx). bmi holds
// the already-resolved MVs of earlier sub-blocks (indexed by ref). bMode is the
// sub-block's inter mode (NEARESTMV/NEARMV); ref is the reference index (0/1).
func (d *Decoder) appendSub8x8(refFrame, ref, row4, col4, bsW4, bsH4, blockIdx, bMode int, bmi [4][2]mv.MV) mv.MV {
	near := bMode == blockmode.NearMV
	switch blockIdx {
	case 0:
		_, mvList, cnt := d.findMVRefs(row4, col4, bsW4, bsH4, refFrame, 0, !near, false, d.signBias, d.allowHP)
		return mvList[cnt-1]
	case 1, 2:
		if !near { // NEARESTMV
			return bmi[0][ref]
		}
		_, mvList, _ := d.findMVRefs(row4, col4, bsW4, bsH4, refFrame, blockIdx, false, false, d.signBias, d.allowHP)
		for n := 0; n < 2; n++ {
			if bmi[0][ref] != mvList[n] {
				return mvList[n]
			}
		}
		return mv.MV{}
	default: // case 3
		if !near { // NEARESTMV
			return bmi[2][ref]
		}
		if bmi[2][ref] != bmi[1][ref] {
			return bmi[1][ref]
		}
		if bmi[2][ref] != bmi[0][ref] {
			return bmi[0][ref]
		}
		_, mvList, _ := d.findMVRefs(row4, col4, bsW4, bsH4, refFrame, 3, false, false, d.signBias, d.allowHP)
		for n := 0; n < 2; n++ {
			if bmi[2][ref] != mvList[n] {
				return mvList[n]
			}
		}
		return mv.MV{}
	}
}
