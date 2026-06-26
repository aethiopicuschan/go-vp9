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

	"github.com/mgvs/go-vp9/blockmode"
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/coeftoken"
	"github.com/mgvs/go-vp9/compheader"
	"github.com/mgvs/go-vp9/ctx"
	"github.com/mgvs/go-vp9/dequant"
	"github.com/mgvs/go-vp9/intra"
	"github.com/mgvs/go-vp9/mc"
	"github.com/mgvs/go-vp9/modeinfo"
	"github.com/mgvs/go-vp9/mv"
	"github.com/mgvs/go-vp9/refbuf"
	"github.com/mgvs/go-vp9/transform"
	"github.com/mgvs/go-vp9/wht"
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
func Block(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	bm blockmode.BlockMode, lossless bool,
	mv0, mv1 mv.MV,
	dc, ac [3]int16,
	dst *refbuf.Frame,
	refs [3]*refbuf.Frame,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsW4, bsH4 int,
) {
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
		reconstructInter(br, probs, counts, bm, lossless, mv0, mv1, dc, ac,
			dst, refs, aboveRow, leftCol, row4, col4, bsW, bsH, bsW4, bsH4)
	} else {
		reconstructIntra(br, probs, counts, bm, lossless, dc, ac,
			dst, aboveRow, leftCol, row4, col4, bsW, bsH, bsW4, bsH4)
	}
}

// ---- intra ------------------------------------------------------------------

func reconstructIntra(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
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
					txType := TxTypeFor(subMode, ctx.TX4x4, false)
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
					coeffs := coeftoken.DecodeBlock(br, ctx.TX4x4, 0, 0, scan, nb, initCtx, &probs.CoefProbs, counts)
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
						nzc := false
						for _, v := range coeffs {
							if v != 0 {
								nzc = true
								break
							}
						}
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
		txType := TxTypeFor(bm.YMode, bm.TxSize, false)
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
					coeffs := coeftoken.DecodeBlock(br, int(bm.TxSize), 0, 0, scan, nb, initCtx, &probs.CoefProbs, counts)
					if coeffs != nil {
						updateNZC(aboveRow, leftCol, sc4, sr4, txN4, anyNonzero(coeffs), 0)
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

	// UV planes (4:2:0: half resolution) — shared stack buffers reused for U and V.
	uvBsW := bsW / 2
	if uvBsW < 4 {
		uvBsW = 4
	}
	uvBsH := bsH / 2
	if uvBsH < 4 {
		uvBsH = 4
	}
	uvTxSize := uvTxSizeFor(bm.TxSize, bsW4, bsH4)
	uvRow := row / 2
	uvCol := col / 2

	var aboveUVBuf [64]byte // 2×32
	var leftUVBuf [32]byte
	var txPredUVBuf [32 * 32]byte
	uvTxBs := 4 << uvTxSize
	uvNumX := uvBsW / uvTxBs
	uvNumY := uvBsH / uvTxBs
	uvTxN4 := 2 << uvTxSize // luma 4×4 units spanned per UV TX block (4:2:0)
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
			if MiH4 > 0 && uvRow+ty*uvTxBs >= MiH4*2 { // UV TX below frame (4:2:0)
				break
			}
			for tx := 0; tx < uvNumX; tx++ {
				if MiW4 > 0 && uvCol+tx*uvTxBs >= MiW4*2 { // UV TX past right edge
					break
				}
				uvTxRow := uvRow + ty*uvTxBs
				uvTxCol := uvCol + tx*uvTxBs
				above := aboveUVBuf[:2*uvTxBs]
				left := leftUVBuf[:uvTxBs]
				corner := gatherBordersUV(plane.buf, plane.stride, plane.base, dst.Width/2, dst.Height/2, uvTxRow, uvTxCol, uvTxBs, uvTxBs, above, left)
				txPred := txPredUVBuf[:uvTxBs*uvTxBs]
				intra.PredictBlock(plane.mode, txPred, uvTxBs, uvTxBs, above, left, corner,
					uvTxRow > TileRowStart4*2, uvTxCol > TileColStart4*2, (tx+1)*uvTxBs < uvBsW)

				sc4 := col4 + tx*uvTxN4
				sr4 := row4 + ty*uvTxN4
				if !bm.Skip {
					initCtx := nzcInitCtxUV(aboveRow, leftCol, sc4, sr4, uvTxN4, plane.planeIdx)
					coeffs := coeftoken.DecodeBlock(br, int(uvTxSize), 1, 0, uvScan, uvNb, initCtx, &probs.CoefProbs, counts)
					if coeffs != nil {
						updateNZC(aboveRow, leftCol, sc4, sr4, uvTxN4, anyNonzero(coeffs), plane.planeIdx)
						dequant.DequantBlock(coeffs, plane.dcQ, plane.acQ, dqShift(uvTxSize))
						if lossless {
							wht.IWHT4x4(coeffs)
						} else {
							transform.InvTransform2D(coeffs, uvTxBs, transform.DCT_DCT)
						}
						addResidual(txPred, uvTxBs, coeffs, uvTxBs)
					} else {
						updateNZC(aboveRow, leftCol, sc4, sr4, uvTxN4, false, plane.planeIdx)
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

func roundMVCompQ4(v int) int {
	if v < 0 {
		return (v - 2) / 4
	}
	return (v + 2) / 4
}

// ---- inter ------------------------------------------------------------------

func reconstructInter(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
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

	// Luma sub-pel: VP9 filters are 1/16-pel (16 phases) but the MV is 1/8-pel,
	// so the integer offset is mv>>3 and the filter phase is (mv&7)<<1 (the even
	// 1/16-pel phases). See libvpx clamp_mv_to_umv_border_sb (mv_q4 = mv<<1).
	if bm.NumSubMV > 0 && ref0 != nil {
		// Sub-8×8 inter: per-4×4 motion compensation — each 4×4 prediction unit
		// uses its own MV (SubMV is duplicated for 4×8/8×4, so a flat 4×4 loop is
		// correct). libvpx dec_build_inter_predictors per bmi sub-block.
		for sy := 0; sy < 2; sy++ {
			for sx := 0; sx < 2; sx++ {
				smv := bm.SubMV[sy*2+sx]
				sc0, sr0 := int(smv[0].Col), int(smv[0].Row)
				sRow, sCol := row+sy*4, col+sx*4
				po := sy*4*bsW + sx*4
				if compound && ref1 != nil {
					sc1, sr1 := int(smv[1].Col), int(smv[1].Row)
					b0 := ref0.YBase() + (sRow+(sr0>>3))*ref0.YStride + (sCol + (sc0 >> 3))
					b1 := ref1.YBase() + (sRow+(sr1>>3))*ref1.YStride + (sCol + (sc1 >> 3))
					mc.PredictComp(ref0.Y, b0, ref0.YStride, (sc0&7)<<1, (sr0&7)<<1, filt,
						ref1.Y, b1, ref1.YStride, (sc1&7)<<1, (sr1&7)<<1, filt, 4, 4, pred[po:], bsW)
				} else {
					b0 := ref0.YBase() + (sRow+(sr0>>3))*ref0.YStride + (sCol + (sc0 >> 3))
					mc.Predict(ref0.Y, b0, ref0.YStride, 4, 4, (sc0&7)<<1, (sr0&7)<<1, filt, pred[po:], bsW)
				}
			}
		}
	} else if ref0 != nil && compound && ref1 != nil {
		base0 := ref0.YBase() + (row+(row0>>3))*ref0.YStride + (col + (col0 >> 3))
		base1 := ref1.YBase() + (row+(row1>>3))*ref1.YStride + (col + (col1 >> 3))
		mc.PredictComp(
			ref0.Y, base0, ref0.YStride, (col0&7)<<1, (row0&7)<<1, filt,
			ref1.Y, base1, ref1.YStride, (col1&7)<<1, (row1&7)<<1, filt,
			bsW, bsH, pred, bsW,
		)
	} else if ref0 != nil {
		base0 := ref0.YBase() + (row+(row0>>3))*ref0.YStride + (col + (col0 >> 3))
		mc.Predict(ref0.Y, base0, ref0.YStride, bsW, bsH, (col0&7)<<1, (row0&7)<<1, filt, pred, bsW)
	}

	if !bm.Skip {
		addResidualY(br, probs, counts, bm.TxSize, transform.DCT_DCT, lossless, dc[0], ac[0],
			pred, bsW, bsH, aboveRow, leftCol, row4, col4)
	}
	writePlane(dst.Y, dst.YStride, dst.YBase(), row, col, pred, bsW, bsH)

	// UV prediction (4:2:0)
	uvBsW := bsW / 2
	if uvBsW < 4 {
		uvBsW = 4
	}
	uvBsH := bsH / 2
	if uvBsH < 4 {
		uvBsH = 4
	}
	uvTxSize := uvTxSizeFor(bm.TxSize, bsW4, bsH4)
	uvRow := row / 2
	uvCol := col / 2
	// Chroma sub-pel (4:2:0): the luma 1/8-pel MV equals a 1/16-pel chroma MV
	// (1 chroma px = 2 luma px), so the integer offset is mv>>4 and the filter
	// phase is mv&15 — all 16 phases (libvpx clamp_mv_to_umv_border_sb, ss=1).
	col0UV, row0UV := int(mv0.Col), int(mv0.Row)
	col1UV, row1UV := int(mv1.Col), int(mv1.Row)
	if bm.NumSubMV > 0 {
		// Sub-8×8 chroma uses the q4-rounded average of the four luma sub-MVs
		// (libvpx mi_mv_pred_q4 / round_mv_comp_q4).
		col0UV, row0UV = avgSubMV(bm.SubMV, 0)
		col1UV, row1UV = avgSubMV(bm.SubMV, 1)
	}

	var predUVBuf [32 * 32]byte
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
		if ref0 != nil && compound && ref1 != nil {
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
			addResidualUV(br, probs, counts, uvTxSize, lossless, dcQ, acQ, predUV, uvBsW, uvBsH,
				aboveRow, leftCol, col4, row4, pi+1) // plane 1=U, 2=V
		}
		writePlane(dstBuf, dstStride, dstBase, uvRow, uvCol, predUV, uvBsW, uvBsH)
	}
}

// ---- residual ---------------------------------------------------------------

// addResidualY decodes coefficients for all TX-sized sub-blocks of the Y plane
// and adds the inverse-transformed residual to pred in-place.
// bsW/bsH are the luma block dimensions in pixels (may be non-square for HORZ/VERT).
func addResidualY(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
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

			coeffs := coeftoken.DecodeBlock(br, int(txSize), 0, refType, scan, nb, initCtx, &probs.CoefProbs, counts)
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
			nzc := !dcOnly || (coeffs[0] != 0)
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
func addResidualUV(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	txSize uint8, lossless bool, dcQ, acQ int16,
	pred []byte, bsW, bsH int,
	aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4, plane int,
) {
	txBs := 4 << txSize
	numX := bsW / txBs
	numY := bsH / txBs
	scan, nb := coeftoken.ScanOrder(int(txSize), coeftoken.ScanDefault)
	// Each UV TX block covers 2 * txN4 luma 4×4 units per dimension (4:2:0).
	uvTxN4 := 2 << txSize

	for ty := 0; ty < numY; ty++ {
		if MiH4 > 0 && row4+ty*uvTxN4 >= MiH4 { // UV TX block below the frame
			break
		}
		for tx := 0; tx < numX; tx++ {
			if MiW4 > 0 && col4+tx*uvTxN4 >= MiW4 { // UV TX block past the frame
				break
			}
			subCol4 := col4 + tx*uvTxN4
			subRow4 := row4 + ty*uvTxN4
			initCtx := nzcInitCtxUV(aboveRow, leftCol, subCol4, subRow4, uvTxN4, plane)
			coeffs := coeftoken.DecodeBlock(br, int(txSize), 1, 1, scan, nb, initCtx, &probs.CoefProbs, counts)
			if coeffs == nil {
				updateNZC(aboveRow, leftCol, subCol4, subRow4, uvTxN4, false, plane)
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
			nzc := !dcOnly || (coeffs[0] != 0)
			updateNZC(aboveRow, leftCol, subCol4, subRow4, uvTxN4, nzc, plane)

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
// (set by the decoder before each tile). Intra prediction treats an above/left
// neighbour as available only when it lies inside the current tile — libvpx
// AvailL = MiCol > MiColStart / AvailU = MiRow > MiRowStart. For a single tile
// these are 0 and the behaviour is identical to a plain frame-edge check.
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
	return gatherBorders(f.Y, f.YStride, f.YBase(), f.Width, f.Height, row, col, bsW, bsH, above, left,
		TileColStart4*4, TileRowStart4*4)
}

// gatherBorders reads the above-row (2*bsW samples) and left-column (bsH
// samples) reference pixels for an intra block at (row,col), replicating the
// frame edge for samples outside the visible w×h picture (libvpx
// build_intra_predictors slower path: out-of-frame above/left samples use the
// last in-frame pixel, NOT the decoded out-of-frame neighbour). Unavailable
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
func gatherBordersUV(buf []byte, stride, base, w, h, row, col, bsW, bsH int, above, left []byte) byte {
	// Chroma is 4:2:0, so the tile boundary in chroma pixels is half the luma one.
	return gatherBorders(buf, stride, base, w, h, row, col, bsW, bsH, above, left,
		TileColStart4*2, TileRowStart4*2)
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

// uvTxSizeFor returns the UV TX size for a given Y TX size and block dimensions.
// For 4:2:0: UV block is (bsW4/2)×(bsH4/2) in 4×4 units; max UV TX = log2(min(uvW4,uvH4)).
func uvTxSizeFor(yTxSize uint8, bsW4, bsH4 int) uint8 {
	uvW4 := bsW4 / 2
	if uvW4 < 1 {
		uvW4 = 1
	}
	uvH4 := bsH4 / 2
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
// across the UV TX block's span (in luma 4×4 units). col4, row4 are luma 4×4
// positions; plane is 1 (U) or 2 (V).
func nzcInitCtxUV(aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4, span, plane int) int {
	var an, ln bool
	for i := 0; i < span; i++ {
		if col4+i < aboveRow.N4 && aboveRow.NZC[plane][col4+i] {
			an = true
		}
		if row4+i < leftCol.N4 && leftCol.NZC[plane][row4+i] {
			ln = true
		}
	}
	return ctx.NZCInitCtx(an, ln)
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
