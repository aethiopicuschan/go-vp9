// Package reconstruct performs per-block pixel reconstruction for a VP9 frame
// (spec §8.4–§8.6 / libvpx vp9/decoder/vp9_decodeframe.c).
//
// For each coding block the pipeline is:
//
//  1. Prediction: fill a pred buffer with either intra (intra.Predict) or
//     inter (mc.Predict) samples.
//  2. Residual: if !skip, decode coefficient tokens (coeftoken.DecodeBlock),
//     dequantise (dequant.DequantBlock), inverse-transform
//     (transform.InvTransform2D or wht.IWHT4x4), and add to pred.
//  3. Write: copy the result into the output frame (dst).
//
// The caller is responsible for:
//   - Resolving the actual motion vector for NEAREST/NEAR/ZERO/NEW modes
//     (mvpred) and passing it in as mv0/mv1.
//   - Computing per-plane DC/AC quantiser steps (segment-adjusted Q-index →
//     dequant.DCQuant / dequant.ACQuant).
package reconstruct

import (
	"fmt"
	"io"

	"github.com/aethiopicuschan/go-vp9/blockmode"
	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/coeftoken"
	"github.com/aethiopicuschan/go-vp9/compheader"
	"github.com/aethiopicuschan/go-vp9/ctx"
	"github.com/aethiopicuschan/go-vp9/dequant"
	"github.com/aethiopicuschan/go-vp9/intra"
	"github.com/aethiopicuschan/go-vp9/mc"
	"github.com/aethiopicuschan/go-vp9/modeinfo"
	"github.com/aethiopicuschan/go-vp9/mv"
	"github.com/aethiopicuschan/go-vp9/refbuf"
	"github.com/aethiopicuschan/go-vp9/transform"
	"github.com/aethiopicuschan/go-vp9/wht"
)

// intraModeTxType maps intra Y mode → TxType for intra 4×4/8×8/16×16 luma.
// Inter blocks and chroma always use DCT_DCT.  32×32 blocks also force DCT_DCT.
// intraModeTxType follows libvpx vp9_blockd.h intra_mode_to_tx_type_lookup.
// ADST_DCT=ADST in vertical(cols)+DCT in rows; DCT_ADST=DCT in cols+ADST in rows.
var intraModeTxType = [10]transform.TxType{
	transform.DCT_DCT,   // DC
	transform.ADST_DCT,  // V    – source at top  → ADST in vertical
	transform.DCT_ADST,  // H    – source at left → ADST in horizontal
	transform.DCT_DCT,   // D45
	transform.ADST_ADST, // D135
	transform.ADST_DCT,  // D117
	transform.DCT_ADST,  // D153
	transform.DCT_ADST,  // D207 (libvpx intra_mode_to_tx_type_lookup)
	transform.ADST_DCT,  // D63  (libvpx intra_mode_to_tx_type_lookup)
	transform.ADST_ADST, // TM
}

// scanTypeFor maps a TX type to the coefficient scan pattern, matching libvpx
// vp9_scan_orders: ADST_DCT → row scan, DCT_ADST → col scan, others → diagonal.
func scanTypeFor(txType transform.TxType) int {
	switch txType {
	case transform.ADST_DCT:
		return coeftoken.ScanRow
	case transform.DCT_ADST:
		return coeftoken.ScanCol
	default:
		return coeftoken.ScanDefault
	}
}

// TxTypeFor returns the inverse-transform type for the luma plane.
// Chroma always uses DCT_DCT (call with isInter=true or any value).
//
// Callers also pass isInter=true for intra blocks of a lossless frame: there
// the transform is the WHT and the coefficient scan is the default (DCT_DCT)
// one regardless of the intra mode (libvpx: (plane || xd->lossless) ? DCT_DCT).
func TxTypeFor(yMode modeinfo.IntraMode, txSize uint8, isInter bool) transform.TxType {
	if isInter || txSize >= ctx.TX32x32 {
		return transform.DCT_DCT
	}
	return intraModeTxType[yMode]
}

// DbgW: optional debug trace writer (residual coef decode), set for one frame.
var DbgW io.Writer

// MiW4, MiH4 are the frame's MI grid size in 4×4 units, set by the decoder
// before reconstruction. A coding block at the right/bottom frame edge may
// extend past the grid; its transform blocks whose top-left lies outside the
// grid are not coded (libvpx max_blocks_wide/high), so they must be skipped in
// both the residual token decode and the reconstruction.
var MiW4, MiH4 int

// intraFrameSize returns the plane size that bounds intra edge pixels: libvpx
// build_intra_predictors replicates the last pixel only past the frame
// BUFFER size (y_width / uv_width = the width aligned up to 8, i.e. the MI
// grid), not past the visible picture, so decoded pixels between the picture
// edge and the 8-aligned edge are used as they are. ssX, ssY are the plane
// subsampling shifts (0,0 luma; the frame's chroma subsampling for U/V:
// uv_width = aligned_width >> ss_x); w, h are the fallback when no grid is set.
func intraFrameSize(w, h int, ssX, ssY uint) (int, int) {
	if MiW4 > 0 && MiH4 > 0 {
		return MiW4 * 4 >> ssX, MiH4 * 4 >> ssY
	}
	return w, h
}

// eobAcc collects the eobs of one coding block's transform blocks: last is the
// eob of the block decoded last (its entropy context is eob > 0), total their
// sum (libvpx decode_block's eobtotal). It lives on Block's stack so that
// decoders share no eob state.
type eobAcc struct{ last, total int }

func (e *eobAcc) decode(br *boolcoder.Reader, txSize, planeType, refType int,
	scan, nb []int16, initCtx int, probs *coeftoken.CoefProbs, counts *compheader.Counts) []int16 {
	out, eob := coeftoken.DecodeBlockEOB(br, txSize, planeType, refType, scan, nb, initCtx, probs, counts)
	e.last = eob
	e.total += eob
	return out
}

func (e *eobAcc) decode16(br *boolcoder.Reader, txSize, planeType, refType int,
	scan, nb []int16, initCtx int, probs *coeftoken.CoefProbs, counts *compheader.Counts) []int64 {
	out, eob := coeftoken.DecodeBlock16EOB(br, txSize, planeType, refType, scan, nb, initCtx, probs, counts)
	e.last = eob
	e.total += eob
	return out
}

// Block reconstructs one VP9 coding block and writes the result into dst.
//
// Parameters:
//   - row4, col4: top-left position in 4×4 MI units (relative to the frame)
//   - bsLog2: coding block size (0=4×4 … 4=64×64)
//   - bm: decoded mode info from blockmode.Decode
//   - lossless: true when frame BaseQIdx==0
//   - mv0, mv1: resolved motion vectors; mv0 is the primary MV (used for all
//     inter modes); mv1 is only used for compound prediction
//   - dc, ac: dequantisation step sizes per plane [Y, U, V]
//   - dst: output frame (write target)
//   - refs: reference frames indexed [Last-1=0, Golden-1=1, AltRef-1=2];
//     nil entries are safe (their plane will remain unchanged)
//   - aboveRow, leftCol: context arrays; used to gather above/left pixel
//     borders and to update NZC flags after residual decode
//
// The result is the sum of the eobs of the block's transform blocks (0 for a
// skipped block): libvpx decode_block's eobtotal.
func Block(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	bm blockmode.BlockMode, lossless bool,
	mv0, mv1 mv.MV,
	dc, ac [3]int16,
	dst *refbuf.Frame,
	refs [3]*refbuf.Frame,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsW4, bsH4 int,
) (eobTotal int) {
	var ea eobAcc
	bsW := bsW4 * 4 // coding block width in pixels (luma)
	bsH := bsH4 * 4 // coding block height in pixels (luma)

	// A skipped block decodes no residual, so its TX paths never update the
	// nonzero-context. libvpx reset_skip_context zeroes the above/left entropy
	// context across the whole block for every plane; mirror that, else the
	// stale nonzero flags leak into the next block's coef context.
	if bm.Skip {
		for p := 0; p < 3; p++ {
			aboveRow.SetNZC(col4, bsW4, p, false)
			leftCol.SetNZC(row4, bsH4, p, false)
		}
	}

	if bm.IsInter {
		reconstructInter(&ea, br, probs, counts, bm, lossless, mv0, mv1, dc, ac,
			dst, refs, aboveRow, leftCol, row4, col4, bsW, bsH, bsW4, bsH4)
	} else {
		reconstructIntra(&ea, br, probs, counts, bm, lossless, dc, ac,
			dst, aboveRow, leftCol, row4, col4, bsW, bsH, bsW4, bsH4)
	}
	return ea.total
}

// ---- intra ------------------------------------------------------------------

func reconstructIntra(ea *eobAcc, br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	bm blockmode.BlockMode, lossless bool,
	dc, ac [3]int16,
	dst *refbuf.Frame,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsW, bsH, bsW4, bsH4 int,
) {
	// Y plane — stack-allocate pred/above/left to avoid heap allocation per block.
	row := row4 * 4
	col := col4 * 4
	var predBuf [64 * 64]byte
	pred := predBuf[:bsW*bsH]

	if bm.NumSubY > 1 {
		// Sub-8×8 intra: each 4×4 sub-position has its own mode and prediction.
		// Positions: ty=0..bsH4-1, tx=0..bsW4-1 (always TX4x4 for sub-8×8).
		var subAboveBuf [8]byte // 2×4
		var subLeftBuf [4]byte
		var subPredBuf [16]byte
		for ty := 0; ty < bsH4; ty++ {
			for tx := 0; tx < bsW4; tx++ {
				subMode := bm.SubYModes[ty*bsW4+tx]
				subRow := row + ty*4
				subCol := col + tx*4
				subAbove := subAboveBuf[:]
				subLeft := subLeftBuf[:]
				subTL := gatherBordersY(dst, subRow, subCol, 4, 4, subAbove, subLeft)
				subPred := subPredBuf[:]
				intra.PredictBlock(subMode, subPred, 4, 4, subAbove, subLeft, subTL,
					subRow > TileRowStart4*4, subCol > TileColStart4*4, tx+1 < bsW4)

				if !bm.Skip {
					txType := TxTypeFor(subMode, ctx.TX4x4, lossless)
					subRow4 := row4 + ty
					subCol4 := col4 + tx
					initCtx := nzcInitCtx(aboveRow, leftCol, subCol4, subRow4, 1)
					scan, nb := coeftoken.ScanOrder(ctx.TX4x4, scanTypeFor(txType))
					dbg := DbgW != nil && row4 == 4 && col4 == 12
					if dbg {
						{
							rb, tb := br.State()
							fmt.Fprintf(DbgW, "RC go Ysub(%d,%d) initCtx=%d txType=%d bpBefore=%d rng=%d t=%d\n", ty, tx, initCtx, txType, br.BitPos(), rb, tb)
						}
					}
					coeffs := ea.decode(br, ctx.TX4x4, 0, 0, scan, nb, initCtx, &probs.CoefProbs, counts)
					if dbg {
						nzcnt := 0
						for _, v := range coeffs {
							if v != 0 {
								nzcnt++
							}
						}
						fmt.Fprintf(DbgW, "RC go Ysub(%d,%d) nz=%d bpAfter=%d\n", ty, tx, nzcnt, br.BitPos())
					}
					if coeffs != nil {
						nzc := ea.last > 0 // libvpx: eob > 0
						updateNZC(aboveRow, leftCol, subCol4, subRow4, 1, nzc, 0)
						dequant.DequantBlock(coeffs, dc[0], ac[0], 0)
						if lossless {
							wht.IWHT4x4(coeffs)
						} else {
							transform.InvTransform2D(coeffs, 4, txType)
						}
						addResidual(subPred, 4, coeffs, 4)
					} else {
						updateNZC(aboveRow, leftCol, subCol4, subRow4, 1, false, 0)
					}
				}
				// Copy 4×4 sub-prediction into the full pred buffer.
				for r := 0; r < 4; r++ {
					copy(pred[(ty*4+r)*bsW+tx*4:], subPred[r*4:(r+1)*4])
				}
				// Write to frame immediately so borders are available for next sub-block.
				writePlane(dst.Y, dst.YStride, dst.YBase(), subRow, subCol, subPred, 4, 4)
			}
		}
		// pred is already written above; skip the writePlane below.
	} else {
		// Per-TX intra: predict + reconstruct each transform block in raster
		// order, writing it to the frame before the next so its neighbours are
		// available (libvpx predict_and_reconstruct_intra_block).
		_ = pred
		txN4 := 1 << bm.TxSize
		txBs := 4 << bm.TxSize
		numX := bsW4 / txN4
		numY := bsH4 / txN4
		txType := TxTypeFor(bm.YMode, bm.TxSize, lossless)
		scan, nb := coeftoken.ScanOrder(int(bm.TxSize), scanTypeFor(txType))
		var aboveBuf [128]byte
		var leftBuf [64]byte
		var txPredBuf [32 * 32]byte
		for ty := 0; ty < numY; ty++ {
			sr4 := row4 + ty*txN4
			if MiH4 > 0 && sr4 >= MiH4 { // TX block top below the frame: not coded
				break
			}
			for tx := 0; tx < numX; tx++ {
				sc4 := col4 + tx*txN4
				if MiW4 > 0 && sc4 >= MiW4 { // TX block left past the frame
					break
				}
				txRow := row + ty*txBs
				txCol := col + tx*txBs
				above := aboveBuf[:2*txBs]
				left := leftBuf[:txBs]
				corner := gatherBordersY(dst, txRow, txCol, txBs, txBs, above, left)
				txPred := txPredBuf[:txBs*txBs]
				intra.PredictBlock(bm.YMode, txPred, txBs, txBs, above, left, corner,
					txRow > TileRowStart4*4, txCol > TileColStart4*4, (tx+1)*txN4 < bsW4)

				if !bm.Skip {
					initCtx := nzcInitCtx(aboveRow, leftCol, sc4, sr4, txN4)
					coeffs := ea.decode(br, int(bm.TxSize), 0, 0, scan, nb, initCtx, &probs.CoefProbs, counts)
					if coeffs != nil {
						updateNZC(aboveRow, leftCol, sc4, sr4, txN4, ea.last > 0, 0)
						dequant.DequantBlock(coeffs, dc[0], ac[0], dqShift(bm.TxSize))
						if lossless {
							wht.IWHT4x4(coeffs)
						} else {
							transform.InvTransform2D(coeffs, txBs, txType)
						}
						addResidual(txPred, txBs, coeffs, txBs)
					} else {
						updateNZC(aboveRow, leftCol, sc4, sr4, txN4, false, 0)
					}
				}
				writePlane(dst.Y, dst.YStride, dst.YBase(), txRow, txCol, txPred, txBs, txBs)
			}
		}
	}

	// UV planes (subsampled by dst.SubX/SubY; the coding block is at least
	// 8×8 luma, so the chroma block is at least 4×4) — shared stack buffers
	// reused for U and V.
	ssX, ssY := uint(dst.SubX), uint(dst.SubY)
	uvBsW := bsW >> ssX
	uvBsH := bsH >> ssY
	uvTxSize := uvTxSizeSS(bm.TxSize, bsW4, bsH4, dst.SubX, dst.SubY)
	uvRow := row >> ssY
	uvCol := col >> ssX

	var aboveUVBuf [64]byte // 2×32
	var leftUVBuf [32]byte
	var txPredUVBuf [32 * 32]byte
	uvTxBs := 4 << uvTxSize
	uvNumX := uvBsW / uvTxBs
	uvNumY := uvBsH / uvTxBs
	// Luma 4×4 units spanned per UV TX block along each axis.
	uvTxN4X := (1 << uvTxSize) << ssX
	uvTxN4Y := (1 << uvTxSize) << ssY
	uvScan, uvNb := coeftoken.ScanOrder(int(uvTxSize), coeftoken.ScanDefault)

	for _, plane := range []struct {
		buf      []byte
		stride   int
		base     int
		mode     modeinfo.IntraMode
		dcQ      int16
		acQ      int16
		planeIdx int
	}{
		{dst.U, dst.UVStride, dst.UBase(), bm.UVMode, dc[1], ac[1], 1},
		{dst.V, dst.UVStride, dst.VBase(), bm.UVMode, dc[2], ac[2], 2},
	} {
		// Per-TX intra for UV (libvpx order): predict + reconstruct + write each
		// UV transform block before the next.
		for ty := 0; ty < uvNumY; ty++ {
			if MiH4 > 0 && uvRow+ty*uvTxBs >= (MiH4*4)>>ssY { // UV TX below frame
				break
			}
			for tx := 0; tx < uvNumX; tx++ {
				if MiW4 > 0 && uvCol+tx*uvTxBs >= (MiW4*4)>>ssX { // UV TX past right edge
					break
				}
				uvTxRow := uvRow + ty*uvTxBs
				uvTxCol := uvCol + tx*uvTxBs
				above := aboveUVBuf[:2*uvTxBs]
				left := leftUVBuf[:uvTxBs]
				corner := gatherBordersUV(plane.buf, plane.stride, plane.base, dst.UVWidth(), dst.UVHeight(), uvTxRow, uvTxCol, uvTxBs, uvTxBs, above, left, ssX, ssY)
				txPred := txPredUVBuf[:uvTxBs*uvTxBs]
				intra.PredictBlock(plane.mode, txPred, uvTxBs, uvTxBs, above, left, corner,
					uvTxRow > (TileRowStart4*4)>>ssY, uvTxCol > (TileColStart4*4)>>ssX, (tx+1)*uvTxBs < uvBsW)

				sc4 := col4 + tx*uvTxN4X
				sr4 := row4 + ty*uvTxN4Y
				if !bm.Skip {
					initCtx := nzcInitCtxUV(aboveRow, leftCol, sc4, sr4, uvTxN4X, uvTxN4Y, plane.planeIdx)
					coeffs := ea.decode(br, int(uvTxSize), 1, 0, uvScan, uvNb, initCtx, &probs.CoefProbs, counts)
					if coeffs != nil {
						updateNZCUV(aboveRow, leftCol, sc4, sr4, uvTxN4X, uvTxN4Y, ea.last > 0, plane.planeIdx)
						dequant.DequantBlock(coeffs, plane.dcQ, plane.acQ, dqShift(uvTxSize))
						if lossless {
							wht.IWHT4x4(coeffs)
						} else {
							transform.InvTransform2D(coeffs, uvTxBs, transform.DCT_DCT)
						}
						addResidual(txPred, uvTxBs, coeffs, uvTxBs)
					} else {
						updateNZCUV(aboveRow, leftCol, sc4, sr4, uvTxN4X, uvTxN4Y, false, plane.planeIdx)
					}
				}
				writePlane(plane.buf, plane.stride, plane.base, uvTxRow, uvTxCol, txPred, uvTxBs, uvTxBs)
			}
		}
	}
}

// avgSubMV returns the q4-rounded average of the four sub-block MV components
// for reference index ref (libvpx mi_mv_pred_q4): sum the four, round toward the
// sign by 2, divide by 4.
func avgSubMV(sub [4][2]mv.MV, ref int) (col, row int) {
	sc := int(sub[0][ref].Col) + int(sub[1][ref].Col) + int(sub[2][ref].Col) + int(sub[3][ref].Col)
	sr := int(sub[0][ref].Row) + int(sub[1][ref].Row) + int(sub[2][ref].Row) + int(sub[3][ref].Row)
	return roundMVCompQ4(sc), roundMVCompQ4(sr)
}

// splitMVUV is libvpx average_split_mvs: the MV (1/8 luma pel) used to
// predict chroma 4×4 number block of a sub-8×8 coding block. 4:4:4 uses the
// luma sub-block's own MV, 4:4:0 averages blocks (block, block+2), 4:2:2
// averages (block, block+1) and 4:2:0 averages all four (avgSubMV).
func splitMVUV(sub [4][2]mv.MV, ref, block, ssX, ssY int) (col, row int) {
	switch {
	case ssX != 0 && ssY != 0:
		return avgSubMV(sub, ref)
	case ssX == 0 && ssY == 0:
		return int(sub[block][ref].Col), int(sub[block][ref].Row)
	}
	other := block + 2 // 4:4:0
	if ssX != 0 {
		other = block + 1 // 4:2:2
	}
	return roundMVCompQ2(int(sub[block][ref].Col) + int(sub[other][ref].Col)),
		roundMVCompQ2(int(sub[block][ref].Row) + int(sub[other][ref].Row))
}

// roundMVCompQ2 is libvpx round_mv_comp_q2.
func roundMVCompQ2(v int) int {
	if v < 0 {
		return (v - 1) / 2
	}
	return (v + 1) / 2
}

func roundMVCompQ4(v int) int {
	if v < 0 {
		return (v - 2) / 4
	}
	return (v + 2) / 4
}

// ---- inter ------------------------------------------------------------------

func reconstructInter(ea *eobAcc, br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	bm blockmode.BlockMode, lossless bool,
	mv0, mv1 mv.MV,
	dc, ac [3]int16,
	dst *refbuf.Frame,
	refs [3]*refbuf.Frame,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsW, bsH, bsW4, bsH4 int,
) {
	row := row4 * 4
	col := col4 * 4
	filt := mc.Filter(bm.InterpFilter)

	col0, row0 := int(mv0.Col), int(mv0.Row)
	col1, row1 := int(mv1.Col), int(mv1.Row)

	// Y prediction — stack-allocate pred to avoid heap allocation per block.
	ref0 := refFrame(refs, bm.RefFrame[0])
	var predBuf [64 * 64]byte
	pred := predBuf[:bsW*bsH]
	compound := bm.RefFrame[1] != ctx.RefIntra
	ref1 := refFrame(refs, bm.RefFrame[1])

	// libvpx clamp_mv_to_umv_border_sb: MVs are converted to 1/16-pel (q4) and
	// clamped so the reference block stays within the border the reference
	// frame is extended by; the clamp can change the sub-pel phase too.
	col0, row0 = clampMVQ4(col0, col, bsW, bsW, MiW4, 0), clampMVQ4(row0, row, bsH, bsH, MiH4, 0)
	col1, row1 = clampMVQ4(col1, col, bsW, bsW, MiW4, 0), clampMVQ4(row1, row, bsH, bsH, MiH4, 0)
	// Luma sub-pel: VP9 filters are 1/16-pel (16 phases) and the 1/8-pel MV was
	// doubled to q4 above, so the integer offset is q4>>4 and the phase q4&15.
	if bm.NumSubMV > 0 && ref0 != nil {
		// Sub-8×8 inter: per-4×4 motion compensation — each 4×4 prediction unit
		// uses its own MV (SubMV is duplicated for 4×8/8×4, so a flat 4×4 loop is
		// correct). libvpx dec_build_inter_predictors per bmi sub-block.
		for sy := 0; sy < 2; sy++ {
			for sx := 0; sx < 2; sx++ {
				smv := bm.SubMV[sy*2+sx]
				sc0, sr0 := clampMVQ4(int(smv[0].Col), col, bsW, bsW, MiW4, 0), clampMVQ4(int(smv[0].Row), row, bsH, bsH, MiH4, 0)
				sRow, sCol := row+sy*4, col+sx*4
				po := sy*4*bsW + sx*4
				if compound && ref1 != nil {
					sc1, sr1 := clampMVQ4(int(smv[1].Col), col, bsW, bsW, MiW4, 0), clampMVQ4(int(smv[1].Row), row, bsH, bsH, MiH4, 0)
					b0 := ref0.YBase() + (sRow+(sr0>>4))*ref0.YStride + (sCol + (sc0 >> 4))
					b1 := ref1.YBase() + (sRow+(sr1>>4))*ref1.YStride + (sCol + (sc1 >> 4))
					mc.PredictComp(ref0.Y, b0, ref0.YStride, sc0&15, sr0&15, filt,
						ref1.Y, b1, ref1.YStride, sc1&15, sr1&15, filt, 4, 4, pred[po:], bsW)
				} else {
					b0 := ref0.YBase() + (sRow+(sr0>>4))*ref0.YStride + (sCol + (sc0 >> 4))
					mc.Predict(ref0.Y, b0, ref0.YStride, 4, 4, sc0&15, sr0&15, filt, pred[po:], bsW)
				}
			}
		}
	} else if ref0 != nil && compound && ref1 != nil {
		base0 := ref0.YBase() + (row+(row0>>4))*ref0.YStride + (col + (col0 >> 4))
		base1 := ref1.YBase() + (row+(row1>>4))*ref1.YStride + (col + (col1 >> 4))
		mc.PredictComp(
			ref0.Y, base0, ref0.YStride, col0&15, row0&15, filt,
			ref1.Y, base1, ref1.YStride, col1&15, row1&15, filt,
			bsW, bsH, pred, bsW,
		)
	} else if ref0 != nil {
		base0 := ref0.YBase() + (row+(row0>>4))*ref0.YStride + (col + (col0 >> 4))
		mc.Predict(ref0.Y, base0, ref0.YStride, bsW, bsH, col0&15, row0&15, filt, pred, bsW)
	}

	if !bm.Skip {
		addResidualY(ea, br, probs, counts, bm.TxSize, transform.DCT_DCT, lossless, dc[0], ac[0],
			pred, bsW, bsH, aboveRow, leftCol, row4, col4)
	}
	writePlane(dst.Y, dst.YStride, dst.YBase(), row, col, pred, bsW, bsH)

	// UV prediction. The plane block is (bsW>>ssX)×(bsH>>ssY) (at least 4×4:
	// the coding block is at least 8×8 luma).
	ssX, ssY := dst.SubX, dst.SubY
	uvBsW := bsW >> uint(ssX)
	uvBsH := bsH >> uint(ssY)
	uvTxSize := uvTxSizeSS(bm.TxSize, bsW4, bsH4, ssX, ssY)
	uvRow := row >> uint(ssY)
	uvCol := col >> uint(ssX)
	// Chroma sub-pel: clampMVQ4 scales the 1/8-luma-pel MV to the plane's
	// 1/16-pel units (×2 on a non-subsampled axis; on a subsampled axis the
	// value is already 1/16 chroma pel), so the integer offset is q4>>4 and the
	// filter phase q4&15 (libvpx clamp_mv_to_umv_border_sb). Taken from the
	// unclamped block MVs: the clamp is per plane.
	col0UV, row0UV := clampMVQ4(int(mv0.Col), col, bsW, uvBsW, MiW4, ssX), clampMVQ4(int(mv0.Row), row, bsH, uvBsH, MiH4, ssY)
	col1UV, row1UV := clampMVQ4(int(mv1.Col), col, bsW, uvBsW, MiW4, ssX), clampMVQ4(int(mv1.Row), row, bsH, uvBsH, MiH4, ssY)
	var predUVBuf [64 * 64]byte
	for pi := 0; pi < 2; pi++ {
		var dstBuf []byte
		var dstStride, dstBase int
		var dcQ, acQ int16
		switch pi {
		case 0:
			dstBuf, dstStride, dstBase = dst.U, dst.UVStride, dst.UBase()
			dcQ, acQ = dc[1], ac[1]
		case 1:
			dstBuf, dstStride, dstBase = dst.V, dst.UVStride, dst.VBase()
			dcQ, acQ = dc[2], ac[2]
		}

		predUV := predUVBuf[:uvBsW*uvBsH]
		if bm.NumSubMV > 0 && ref0 != nil {
			// Sub-8×8: one 4×4 prediction per chroma 4×4, each with the MV
			// libvpx average_split_mvs derives for block index i (raster order
			// over the plane's 4×4s): the luma sub-MV itself for 4:4:4, a pair
			// average for 4:2:2 / 4:4:0, the average of all four for 4:2:0.
			i := 0
			for y := 0; y < uvBsH; y += 4 {
				for x := 0; x < uvBsW; x += 4 {
					c0, r0 := splitMVUV(bm.SubMV, 0, i, ssX, ssY)
					c0, r0 = clampMVQ4(c0, col, bsW, uvBsW, MiW4, ssX), clampMVQ4(r0, row, bsH, uvBsH, MiH4, ssY)
					b0 := uOrVBase(ref0, pi) + (uvRow+y+(r0>>4))*ref0.UVStride + (uvCol + x + (c0 >> 4))
					po := y*uvBsW + x
					if compound && ref1 != nil {
						c1, r1 := splitMVUV(bm.SubMV, 1, i, ssX, ssY)
						c1, r1 = clampMVQ4(c1, col, bsW, uvBsW, MiW4, ssX), clampMVQ4(r1, row, bsH, uvBsH, MiH4, ssY)
						b1 := uOrVBase(ref1, pi) + (uvRow+y+(r1>>4))*ref1.UVStride + (uvCol + x + (c1 >> 4))
						mc.PredictComp(uOrVSlice(ref0, pi), b0, ref0.UVStride, c0&15, r0&15, filt,
							uOrVSlice(ref1, pi), b1, ref1.UVStride, c1&15, r1&15, filt, 4, 4, predUV[po:], uvBsW)
					} else {
						mc.Predict(uOrVSlice(ref0, pi), b0, ref0.UVStride, 4, 4, c0&15, r0&15, filt, predUV[po:], uvBsW)
					}
					i++
				}
			}
		} else if ref0 != nil && compound && ref1 != nil {
			base0 := uOrVBase(ref0, pi) + (uvRow+(row0UV>>4))*ref0.UVStride + (uvCol + (col0UV >> 4))
			base1 := uOrVBase(ref1, pi) + (uvRow+(row1UV>>4))*ref1.UVStride + (uvCol + (col1UV >> 4))
			mc.PredictComp(
				uOrVSlice(ref0, pi), base0, ref0.UVStride, col0UV&15, row0UV&15, filt,
				uOrVSlice(ref1, pi), base1, ref1.UVStride, col1UV&15, row1UV&15, filt,
				uvBsW, uvBsH, predUV, uvBsW,
			)
		} else if ref0 != nil {
			base0 := uOrVBase(ref0, pi) + (uvRow+(row0UV>>4))*ref0.UVStride + (uvCol + (col0UV >> 4))
			mc.Predict(uOrVSlice(ref0, pi), base0, ref0.UVStride, uvBsW, uvBsH,
				col0UV&15, row0UV&15, filt, predUV, uvBsW)
		}

		if !bm.Skip {
			addResidualUV(ea, br, probs, counts, uvTxSize, lossless, dcQ, acQ, predUV, uvBsW, uvBsH,
				aboveRow, leftCol, col4, row4, pi+1, ssX, ssY) // plane 1=U, 2=V
		}
		writePlane(dstBuf, dstStride, dstBase, uvRow, uvCol, predUV, uvBsW, uvBsH)
	}
}

// ---- residual ---------------------------------------------------------------

// addResidualY decodes coefficients for all TX-sized sub-blocks of the Y plane
// and adds the inverse-transformed residual to pred in-place.
// bsW/bsH are the luma block dimensions in pixels (may be non-square for HORZ/VERT).
func addResidualY(ea *eobAcc, br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	txSize uint8, txType transform.TxType, lossless bool, dcQ, acQ int16,
	pred []byte, bsW, bsH int,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4 int,
) {
	txBs := 4 << txSize // TX block size in pixels (square)
	numX := bsW / txBs  // TX sub-blocks per row
	numY := bsH / txBs  // TX sub-blocks per column
	txN4 := 1 << txSize // TX size in 4×4 units

	scan, nb := coeftoken.ScanOrder(int(txSize), scanTypeFor(txType))
	refType := 1 // inter block (coef-prob refType=1)

	for ty := 0; ty < numY; ty++ {
		if MiH4 > 0 && row4+ty*txN4 >= MiH4 { // TX block top below the frame
			break
		}
		for tx := 0; tx < numX; tx++ {
			if MiW4 > 0 && col4+tx*txN4 >= MiW4 { // TX block left past the frame
				break
			}
			subRow4 := row4 + ty*txN4
			subCol4 := col4 + tx*txN4
			initCtx := nzcInitCtx(aboveRow, leftCol, subCol4, subRow4, txN4)

			coeffs := ea.decode(br, int(txSize), 0, refType, scan, nb, initCtx, &probs.CoefProbs, counts)
			if coeffs == nil {
				updateNZC(aboveRow, leftCol, subCol4, subRow4, txN4, false, 0)
				continue
			}
			// Combined scan: check DC-only and nzc in one pass over AC coefficients.
			dcOnly := true
			for _, v := range coeffs[1:] {
				if v != 0 {
					dcOnly = false
					break
				}
			}
			nzc := ea.last > 0 // libvpx dec_set_contexts(eob > 0)
			updateNZC(aboveRow, leftCol, subCol4, subRow4, txN4, nzc, 0)

			// DC-only fast path: skip DequantBlock, full InvTransform2D, and indexed addResidual.
			if !lossless && txType == transform.DCT_DCT && dcOnly {
				if coeffs[0] != 0 {
					dqDC := dequantDC(coeffs[0], dcQ, txSize)
					predOff := ty*txBs*bsW + tx*txBs
					addResidualUniform(pred[predOff:], bsW, txBs, transform.DCOnlyValue(int16(dqDC), transform.FinalShift(txBs)))
				}
				continue
			}

			dequant.DequantBlock(coeffs, dcQ, acQ, dqShift(txSize))

			if lossless {
				wht.IWHT4x4(coeffs)
			} else {
				transform.InvTransform2D(coeffs, txBs, txType)
			}

			// Add residual to prediction sub-block (pred stride = bsW).
			predOff := ty*txBs*bsW + tx*txBs
			addResidual(pred[predOff:], bsW, coeffs, txBs)
		}
	}
}

// addResidualUV decodes and applies residual for one UV plane.
// bsW/bsH are the UV block dimensions in pixels (may be non-square).
// col4, row4 are the luma 4×4 position of the coding block; plane is 1 (U) or 2 (V).
// uvTxN4 = txN4 in luma 4×4 units for one UV TX block = 2 * (1 << uvTxSize).
func addResidualUV(ea *eobAcc, br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	txSize uint8, lossless bool, dcQ, acQ int16,
	pred []byte, bsW, bsH int,
	aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4, plane, ssX, ssY int,
) {
	txBs := 4 << txSize
	numX := bsW / txBs
	numY := bsH / txBs
	scan, nb := coeftoken.ScanOrder(int(txSize), coeftoken.ScanDefault)
	// Luma 4×4 units covered by one UV TX block along each axis.
	uvTxN4X := (1 << txSize) << uint(ssX)
	uvTxN4Y := (1 << txSize) << uint(ssY)

	for ty := 0; ty < numY; ty++ {
		if MiH4 > 0 && row4+ty*uvTxN4Y >= MiH4 { // UV TX block below the frame
			break
		}
		for tx := 0; tx < numX; tx++ {
			if MiW4 > 0 && col4+tx*uvTxN4X >= MiW4 { // UV TX block past the frame
				break
			}
			subCol4 := col4 + tx*uvTxN4X
			subRow4 := row4 + ty*uvTxN4Y
			initCtx := nzcInitCtxUV(aboveRow, leftCol, subCol4, subRow4, uvTxN4X, uvTxN4Y, plane)
			coeffs := ea.decode(br, int(txSize), 1, 1, scan, nb, initCtx, &probs.CoefProbs, counts)
			if coeffs == nil {
				updateNZCUV(aboveRow, leftCol, subCol4, subRow4, uvTxN4X, uvTxN4Y, false, plane)
				continue
			}
			// Combined scan: check DC-only and nzc in one pass over AC coefficients.
			dcOnly := true
			for _, v := range coeffs[1:] {
				if v != 0 {
					dcOnly = false
					break
				}
			}
			nzc := ea.last > 0 // libvpx dec_set_contexts(eob > 0)
			updateNZCUV(aboveRow, leftCol, subCol4, subRow4, uvTxN4X, uvTxN4Y, nzc, plane)

			// DC-only fast path: UV always uses DCT_DCT.
			if !lossless && dcOnly {
				if coeffs[0] != 0 {
					dqDC := dequantDC(coeffs[0], dcQ, txSize)
					predOff := ty*txBs*bsW + tx*txBs
					addResidualUniform(pred[predOff:], bsW, txBs, transform.DCOnlyValue(int16(dqDC), transform.FinalShift(txBs)))
				}
				continue
			}

			dequant.DequantBlock(coeffs, dcQ, acQ, dqShift(txSize))
			if lossless {
				wht.IWHT4x4(coeffs)
			} else {
				transform.InvTransform2D(coeffs, txBs, transform.DCT_DCT)
			}
			predOff := ty*txBs*bsW + tx*txBs
			addResidual(pred[predOff:], bsW, coeffs, txBs)
		}
	}
}

// addResidualUniform adds the same bias value to every pixel in a txBs×txBs block.
// Used by the DC-only fast path to avoid indexing into the coefficients array.
func addResidualUniform(pred []byte, predStride, txBs int, bias int16) {
	b := int(bias)
	for r := 0; r < txBs; r++ {
		row := pred[r*predStride : r*predStride+txBs]
		for i, px := range row {
			v := int(px) + b
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			row[i] = byte(v)
		}
	}
}

// addResidual adds txBs×txBs residual coefficients to pred (stride=predStride).
func addResidual(pred []byte, predStride int, coeffs []int16, txBs int) {
	for r := 0; r < txBs; r++ {
		for c := 0; c < txBs; c++ {
			v := int(pred[r*predStride+c]) + int(coeffs[r*txBs+c])
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			pred[r*predStride+c] = byte(v)
		}
	}
}

// ---- border gathering -------------------------------------------------------

// TileColStart4, TileRowStart4 hold the current tile's top-left in 4×4 units
// (set by the decoder before each tile). Intra prediction treats a left
// neighbour as available only when it lies inside the current tile column
// (libvpx left_mi: mi_col > tile->mi_col_start). The above neighbour is
// frame-relative in libvpx (above_mi: mi_row != 0) — tile rows do not cut intra
// prediction — so the decoder keeps TileRowStart4 at 0 for every tile row.
var (
	TileColStart4 int
	TileRowStart4 int
)

// gatherBordersY reads above-row (2*bsW pixels) and left-column (bsH pixels)
// pixel borders for intra prediction of a Y block at (row, col).
// above is always 2*bsW long to satisfy D45/D63 extended requirements;
// pixels beyond the frame edge are filled with the last available value.
// Caller must provide above[:2*bsW] and left[:bsH]; returns topLeft.
func gatherBordersY(f *refbuf.Frame, row, col, bsW, bsH int, above, left []byte) byte {
	w, h := intraFrameSize(f.Width, f.Height, 0, 0)
	return gatherBorders(f.Y, f.YStride, f.YBase(), w, h, row, col, bsW, bsH, above, left,
		TileColStart4*4, TileRowStart4*4)
}

// gatherBorders reads the above-row (2*bsW samples) and left-column (bsH
// samples) reference pixels for an intra block at (row,col), replicating the
// frame edge for samples outside the w×h plane (libvpx
// build_intra_predictors slower path: out-of-frame above/left samples use the
// last in-frame pixel, NOT the decoded out-of-frame neighbour). w×h is the
// 8-aligned buffer size, see intraFrameSize. Unavailable
// sides are filled with 128 (the caller's PredictBlock re-pads per availability).
func gatherBorders(plane []byte, stride, base, w, h, row, col, bsW, bsH int, above, left []byte, leftBound, aboveBound int) byte {
	var topLeft byte
	// Clamp the reference row/col to the visible frame. For valid blocks this is
	// a no-op; it only guards against a block placed past the frame edge (which
	// indicates an upstream desync — e.g. an unfinished inter-frame path).
	if row > h {
		row = h
	}
	if col > w {
		col = w
	}
	// A neighbour is available only inside the current tile (leftBound/aboveBound
	// are the tile's top-left in this plane's pixels; 0 for a single tile).
	if row > aboveBound {
		ar := base + (row-1)*stride
		for i := 0; i < 2*bsW; i++ {
			px := col + i
			if px >= w {
				px = w - 1
			}
			above[i] = plane[ar+px]
		}
		if col > leftBound {
			topLeft = plane[ar+col-1]
		}
	} else {
		for i := range above {
			above[i] = 128
		}
		topLeft = 128
	}
	if col > leftBound {
		for i := 0; i < bsH; i++ {
			py := row + i
			if py >= h {
				py = h - 1
			}
			left[i] = plane[base+py*stride+col-1]
		}
	} else {
		for i := range left {
			left[i] = 128
		}
	}
	return topLeft
}

// gatherBordersUV reads borders for one UV plane (2*bsW above, bsH left).
// above is 2*bsW to satisfy D45/D63 extended requirements.
// Caller must provide above[:2*bsW] and left[:bsH]; returns topLeft.
func gatherBordersUV(buf []byte, stride, base, w, h, row, col, bsW, bsH int, above, left []byte, ssX, ssY uint) byte {
	// The tile boundary in chroma pixels is the luma one scaled by the subsampling.
	w, h = intraFrameSize(w, h, ssX, ssY)
	return gatherBorders(buf, stride, base, w, h, row, col, bsW, bsH, above, left,
		(TileColStart4*4)>>ssX, (TileRowStart4*4)>>ssY)
}

// ---- plane write ------------------------------------------------------------

func writePlane(buf []byte, stride, base, row, col int, pred []byte, bsW, bsH int) {
	for r := 0; r < bsH; r++ {
		startIdx := base + (row+r)*stride + col
		if startIdx >= len(buf) {
			break
		}
		copy(buf[startIdx:], pred[r*bsW:(r+1)*bsW])
	}
}

// ---- helper utilities -------------------------------------------------------

// uvTxSizeFor returns the 4:2:0 UV TX size for a given Y TX size and block
// dimensions (UVTxSize with ssX = ssY = 1).
func uvTxSizeFor(yTxSize uint8, bsW4, bsH4 int) uint8 {
	return uvTxSizeSS(yTxSize, bsW4, bsH4, 1, 1)
}

// UVTxSize returns the chroma transform size for a coding block of bsW4×bsH4
// luma 4×4 units with luma transform size yTxSize, for chroma subsampling
// shifts ssX, ssY (libvpx uv_txsize_lookup[bsize][tx][ss_x][ss_y]): the luma
// size capped to the largest transform that fits the chroma block.
func UVTxSize(yTxSize uint8, bsW4, bsH4, ssX, ssY int) uint8 {
	return uvTxSizeSS(yTxSize, bsW4, bsH4, ssX, ssY)
}

func uvTxSizeSS(yTxSize uint8, bsW4, bsH4, ssX, ssY int) uint8 {
	uvW4 := bsW4 >> uint(ssX)
	if uvW4 < 1 {
		uvW4 = 1
	}
	uvH4 := bsH4 >> uint(ssY)
	if uvH4 < 1 {
		uvH4 = 1
	}
	min4 := uvW4
	if uvH4 < min4 {
		min4 = uvH4
	}
	maxUV := uint8(log2n4(min4))
	if yTxSize < maxUV {
		return yTxSize
	}
	return maxUV
}

func log2n4(n int) int {
	l := 0
	for n > 1 {
		l++
		n >>= 1
	}
	return l
}

// refFrame returns the reference frame for the given RefFrame constant.
// ctx.RefLast=1, RefGolden=2, RefAltRef=3 → refs[0..2].
func refFrame(refs [3]*refbuf.Frame, rf int) *refbuf.Frame {
	if rf < 1 || rf > 3 {
		return nil
	}
	return refs[rf-1]
}

// mvSubPelUV sub-samples a luma MV (units: 1/8 pixel) for 4:2:0 chroma.
// Returns the chroma MV components in the same 1/8-pixel units.
// chroma_mv = round(luma_mv / 2) with sign preservation.
func mvSubPelUV(m mv.MV) (colMV, rowMV int) {
	c := int(m.Col)
	r := int(m.Row)
	if c < 0 {
		c = -((-c + 1) >> 1)
	} else {
		c = (c + 1) >> 1
	}
	if r < 0 {
		r = -((-r + 1) >> 1)
	} else {
		r = (r + 1) >> 1
	}
	return c, r
}

// nzcInitCtx returns the NZC init context (0..2) for a Y TX sub-block (plane 0).
// nzcInitCtx aggregates the nonzero-context across the TX block's span (txN4
// 4×4 units), matching libvpx get_entropy_context which ORs all above/left
// entropy-context bytes the TX block covers (TX8x8 reads 2, TX16x16 4, …).
func nzcInitCtx(aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4, txN4 int) int {
	var an, ln bool
	for i := 0; i < txN4; i++ {
		if col4+i < aboveRow.N4 && aboveRow.NZC[0][col4+i] {
			an = true
		}
		if row4+i < leftCol.N4 && leftCol.NZC[0][row4+i] {
			ln = true
		}
	}
	return ctx.NZCInitCtx(an, ln)
}

// nzcInitCtxUV returns the NZC init context for a UV TX sub-block, aggregating
// across the UV TX block's span (in luma 4×4 units: spanX columns of the above
// context, spanY rows of the left context — they differ for 4:2:2 / 4:4:0).
// col4, row4 are luma 4×4 positions; plane is 1 (U) or 2 (V).
func nzcInitCtxUV(aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4, spanX, spanY, plane int) int {
	var an, ln bool
	for i := 0; i < spanX; i++ {
		if col4+i < aboveRow.N4 && aboveRow.NZC[plane][col4+i] {
			an = true
		}
	}
	for i := 0; i < spanY; i++ {
		if row4+i < leftCol.N4 && leftCol.NZC[plane][row4+i] {
			ln = true
		}
	}
	return ctx.NZCInitCtx(an, ln)
}

// updateNZCUV is updateNZC for a UV TX block spanning spanX above-context
// columns and spanY left-context rows (luma 4×4 units).
func updateNZCUV(aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4, spanX, spanY int, nzc bool, plane int) {
	aboveRow.SetNZC(col4, spanX, plane, nzc)
	leftCol.SetNZC(row4, spanY, plane, nzc)
}

// updateNZC writes the NZC flag for the given plane to above/left context.
func updateNZC(aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4, txN4 int, nzc bool, plane int) {
	aboveRow.SetNZC(col4, txN4, plane, nzc)
	leftCol.SetNZC(row4, txN4, plane, nzc)
}

// uOrVBase returns the pixel base offset for plane pi (0=U, 1=V).
func uOrVBase(f *refbuf.Frame, pi int) int {
	if pi == 0 {
		return f.UBase()
	}
	return f.VBase()
}

// uOrVSlice returns the pixel slice for plane pi (0=U, 1=V).
func uOrVSlice(f *refbuf.Frame, pi int) []byte {
	if pi == 0 {
		return f.U
	}
	return f.V
}

// dqShift is libvpx dq_shift: 1 for TX_32X32, else 0.
func dqShift(txSize uint8) int {
	if txSize == ctx.TX32x32 {
		return 1
	}
	return 0
}

// dequantDC dequantises a DC coefficient exactly like dequant.dequant1 (apply
// the TX_32X32 >>1 dq_shift toward zero, then clamp to int16) — used by the
// DC-only fast paths, which must match the full DequantBlock path.
func dequantDC(coef, q int16, txSize uint8) int32 {
	p := int32(coef) * int32(q)
	if dqShift(txSize) > 0 {
		if p < 0 {
			p = -((-p) >> 1)
		} else {
			p >>= 1
		}
	}
	if p > 32767 {
		p = 32767
	} else if p < -32768 {
		p = -32768
	}
	return p
}

// anyNonzero reports whether any coefficient is non-zero.
func anyNonzero(c []int16) bool {
	for _, v := range c {
		if v != 0 {
			return true
		}
	}
	return false
}

// clampMVQ4 is one component of libvpx clamp_mv_to_umv_border_sb
// (vp9_reconinter.h), as used by dec_build_inter_predictors: it converts a
// 1/8-pel luma MV component to the plane's 1/16-pel units and clamps it to
// [mb_to_left_edge - spel_left, mb_to_right_edge + spel_right].
//
// mv: MV component in 1/8 luma pel (for 4:2:0 chroma that is already 1/16
// chroma pel). pos, blk: the prediction block's luma position and size in
// pixels along this axis (the whole 8x8 for sub-8x8 blocks). planeBW: the
// block size in this plane's pixels. mi4: frame size in 4x4 units along this
// axis (MiW4 / MiH4). ss: plane subsampling (0 luma, 1 chroma).
func clampMVQ4(mv, pos, blk, planeBW, mi4, ss int) int {
	mul := 1 << uint(1-ss)
	q4 := mv * mul
	if mi4 <= 0 {
		return q4
	}
	// libvpx works on the 8x8 MI grid: mi_cols = ceil(width / 8).
	frame := (mi4 + 1) / 2 * 8
	toLo := -pos * 8                // mb_to_left_edge (1/8 pel)
	toHi := (frame - blk - pos) * 8 // mb_to_right_edge
	spel := (4 + planeBW) << 4      // (VP9_INTERP_EXTEND + bw) << SUBPEL_BITS
	lo, hi := toLo*mul-spel, toHi*mul+spel-16
	if q4 < lo {
		return lo
	}
	if q4 > hi {
		return hi
	}
	return q4
}
