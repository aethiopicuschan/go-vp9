package reconstruct

import (
	"fmt"

	"github.com/aethiopicuschan/go-vp9/blockmode"
	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/coeftoken"
	"github.com/aethiopicuschan/go-vp9/compheader"
	"github.com/aethiopicuschan/go-vp9/ctx"
	"github.com/aethiopicuschan/go-vp9/dequant"
	"github.com/aethiopicuschan/go-vp9/intra16"
	"github.com/aethiopicuschan/go-vp9/mc16"
	"github.com/aethiopicuschan/go-vp9/modeinfo"
	"github.com/aethiopicuschan/go-vp9/mv"
	"github.com/aethiopicuschan/go-vp9/refbuf"
	"github.com/aethiopicuschan/go-vp9/transform"
	"github.com/aethiopicuschan/go-vp9/wht"
)

var MaxVal16 = 255

func Block16(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	bm blockmode.BlockMode, lossless bool,
	mv0, mv1 mv.MV,
	dc, ac [3]int16,
	dst *refbuf.Frame16,
	refs [3]*refbuf.Frame16,
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
		reconstructInter_16(br, probs, counts, bm, lossless, mv0, mv1, dc, ac,
			dst, refs, aboveRow, leftCol, row4, col4, bsW, bsH, bsW4, bsH4)
	} else {
		reconstructIntra_16(br, probs, counts, bm, lossless, dc, ac,
			dst, aboveRow, leftCol, row4, col4, bsW, bsH, bsW4, bsH4)
	}
}

func reconstructIntra_16(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	bm blockmode.BlockMode, lossless bool,
	dc, ac [3]int16,
	dst *refbuf.Frame16,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsW, bsH, bsW4, bsH4 int,
) {
	// Y plane — stack-allocate pred/above/left to avoid heap allocation per block.
	row := row4 * 4
	col := col4 * 4
	var predBuf [64 * 64]uint16
	pred := predBuf[:bsW*bsH]

	if bm.NumSubY > 1 {
		// Sub-8×8 intra: each 4×4 sub-position has its own mode and prediction.
		// Positions: ty=0..bsH4-1, tx=0..bsW4-1 (always TX4x4 for sub-8×8).
		var subAboveBuf [8]uint16 // 2×4
		var subLeftBuf [4]uint16
		var subPredBuf [16]uint16
		for ty := 0; ty < bsH4; ty++ {
			for tx := 0; tx < bsW4; tx++ {
				subMode := bm.SubYModes[ty*bsW4+tx]
				subRow := row + ty*4
				subCol := col + tx*4
				subAbove := subAboveBuf[:]
				subLeft := subLeftBuf[:]
				subTL := gatherBordersY_16(dst, subRow, subCol, 4, 4, subAbove, subLeft)
				subPred := subPredBuf[:]
				intra16.PredictBlock(subMode, subPred, 4, 4, subAbove, subLeft, subTL,
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
					coeffs := coeftoken.DecodeBlock16(br, ctx.TX4x4, 0, 0, scan, nb, initCtx, &probs.CoefProbs, counts)
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
						dequant.DequantBlock16(coeffs, dc[0], ac[0], 0)
						if lossless {
							wht.IWHT4x4_16(coeffs)
						} else {
							transform.InvTransform2D16(coeffs, 4, txType)
						}
						addResidual_16(subPred, 4, coeffs, 4)
					} else {
						updateNZC(aboveRow, leftCol, subCol4, subRow4, 1, false, 0)
					}
				}
				// Copy 4×4 sub-prediction into the full pred buffer.
				for r := 0; r < 4; r++ {
					copy(pred[(ty*4+r)*bsW+tx*4:], subPred[r*4:(r+1)*4])
				}
				// Write to frame immediately so borders are available for next sub-block.
				writePlane_16(dst.Y, dst.YStride, dst.YBase(), subRow, subCol, subPred, 4, 4)
			}
		}
		// pred is already written above; skip the writePlane_16 below.
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
		var aboveBuf [128]uint16
		var leftBuf [64]uint16
		var txPredBuf [32 * 32]uint16
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
				corner := gatherBordersY_16(dst, txRow, txCol, txBs, txBs, above, left)
				txPred := txPredBuf[:txBs*txBs]
				intra16.PredictBlock(bm.YMode, txPred, txBs, txBs, above, left, corner,
					txRow > TileRowStart4*4, txCol > TileColStart4*4, (tx+1)*txN4 < bsW4)

				if !bm.Skip {
					initCtx := nzcInitCtx(aboveRow, leftCol, sc4, sr4, txN4)
					coeffs := coeftoken.DecodeBlock16(br, int(bm.TxSize), 0, 0, scan, nb, initCtx, &probs.CoefProbs, counts)
					if coeffs != nil {
						updateNZC(aboveRow, leftCol, sc4, sr4, txN4, anyNonzero_16(coeffs), 0)
						dequant.DequantBlock16(coeffs, dc[0], ac[0], dqShift(bm.TxSize))
						if lossless {
							wht.IWHT4x4_16(coeffs)
						} else {
							transform.InvTransform2D16(coeffs, txBs, txType)
						}
						addResidual_16(txPred, txBs, coeffs, txBs)
					} else {
						updateNZC(aboveRow, leftCol, sc4, sr4, txN4, false, 0)
					}
				}
				writePlane_16(dst.Y, dst.YStride, dst.YBase(), txRow, txCol, txPred, txBs, txBs)
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

	var aboveUVBuf [64]uint16 // 2×32
	var leftUVBuf [32]uint16
	var txPredUVBuf [32 * 32]uint16
	uvTxBs := 4 << uvTxSize
	uvNumX := uvBsW / uvTxBs
	uvNumY := uvBsH / uvTxBs
	uvTxN4 := 2 << uvTxSize // luma 4×4 units spanned per UV TX block (4:2:0)
	uvScan, uvNb := coeftoken.ScanOrder(int(uvTxSize), coeftoken.ScanDefault)

	for _, plane := range []struct {
		buf      []uint16
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
				corner := gatherBordersUV_16(plane.buf, plane.stride, plane.base, dst.Width/2, dst.Height/2, uvTxRow, uvTxCol, uvTxBs, uvTxBs, above, left)
				txPred := txPredUVBuf[:uvTxBs*uvTxBs]
				intra16.PredictBlock(plane.mode, txPred, uvTxBs, uvTxBs, above, left, corner,
					uvTxRow > TileRowStart4*2, uvTxCol > TileColStart4*2, (tx+1)*uvTxBs < uvBsW)

				sc4 := col4 + tx*uvTxN4
				sr4 := row4 + ty*uvTxN4
				if !bm.Skip {
					initCtx := nzcInitCtxUV(aboveRow, leftCol, sc4, sr4, uvTxN4, plane.planeIdx)
					coeffs := coeftoken.DecodeBlock16(br, int(uvTxSize), 1, 0, uvScan, uvNb, initCtx, &probs.CoefProbs, counts)
					if coeffs != nil {
						updateNZC(aboveRow, leftCol, sc4, sr4, uvTxN4, anyNonzero_16(coeffs), plane.planeIdx)
						dequant.DequantBlock16(coeffs, plane.dcQ, plane.acQ, dqShift(uvTxSize))
						if lossless {
							wht.IWHT4x4_16(coeffs)
						} else {
							transform.InvTransform2D16(coeffs, uvTxBs, transform.DCT_DCT)
						}
						addResidual_16(txPred, uvTxBs, coeffs, uvTxBs)
					} else {
						updateNZC(aboveRow, leftCol, sc4, sr4, uvTxN4, false, plane.planeIdx)
					}
				}
				writePlane_16(plane.buf, plane.stride, plane.base, uvTxRow, uvTxCol, txPred, uvTxBs, uvTxBs)
			}
		}
	}
}

func reconstructInter_16(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	bm blockmode.BlockMode, lossless bool,
	mv0, mv1 mv.MV,
	dc, ac [3]int16,
	dst *refbuf.Frame16,
	refs [3]*refbuf.Frame16,
	aboveRow *ctx.Row, leftCol *ctx.Col,
	row4, col4, bsW, bsH, bsW4, bsH4 int,
) {
	row := row4 * 4
	col := col4 * 4
	filt := mc16.Filter(bm.InterpFilter)

	col0, row0 := int(mv0.Col), int(mv0.Row)
	col1, row1 := int(mv1.Col), int(mv1.Row)

	// Y prediction — stack-allocate pred to avoid heap allocation per block.
	ref0 := refFrame_16(refs, bm.RefFrame[0])
	var predBuf [64 * 64]uint16
	pred := predBuf[:bsW*bsH]
	compound := bm.RefFrame[1] != ctx.RefIntra
	ref1 := refFrame_16(refs, bm.RefFrame[1])

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
					mc16.PredictComp(ref0.Y, b0, ref0.YStride, (sc0&7)<<1, (sr0&7)<<1, filt,
						ref1.Y, b1, ref1.YStride, (sc1&7)<<1, (sr1&7)<<1, filt, 4, 4, pred[po:], bsW)
				} else {
					b0 := ref0.YBase() + (sRow+(sr0>>3))*ref0.YStride + (sCol + (sc0 >> 3))
					mc16.Predict(ref0.Y, b0, ref0.YStride, 4, 4, (sc0&7)<<1, (sr0&7)<<1, filt, pred[po:], bsW)
				}
			}
		}
	} else if ref0 != nil && compound && ref1 != nil {
		base0 := ref0.YBase() + (row+(row0>>3))*ref0.YStride + (col + (col0 >> 3))
		base1 := ref1.YBase() + (row+(row1>>3))*ref1.YStride + (col + (col1 >> 3))
		mc16.PredictComp(
			ref0.Y, base0, ref0.YStride, (col0&7)<<1, (row0&7)<<1, filt,
			ref1.Y, base1, ref1.YStride, (col1&7)<<1, (row1&7)<<1, filt,
			bsW, bsH, pred, bsW,
		)
	} else if ref0 != nil {
		base0 := ref0.YBase() + (row+(row0>>3))*ref0.YStride + (col + (col0 >> 3))
		mc16.Predict(ref0.Y, base0, ref0.YStride, bsW, bsH, (col0&7)<<1, (row0&7)<<1, filt, pred, bsW)
	}

	if !bm.Skip {
		addResidualY_16(br, probs, counts, bm.TxSize, transform.DCT_DCT, lossless, dc[0], ac[0],
			pred, bsW, bsH, aboveRow, leftCol, row4, col4)
	}
	writePlane_16(dst.Y, dst.YStride, dst.YBase(), row, col, pred, bsW, bsH)

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

	var predUVBuf [32 * 32]uint16
	for pi := 0; pi < 2; pi++ {
		var dstBuf []uint16
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
			base0 := uOrVBase_16(ref0, pi) + (uvRow+(row0UV>>4))*ref0.UVStride + (uvCol + (col0UV >> 4))
			base1 := uOrVBase_16(ref1, pi) + (uvRow+(row1UV>>4))*ref1.UVStride + (uvCol + (col1UV >> 4))
			mc16.PredictComp(
				uOrVSlice_16(ref0, pi), base0, ref0.UVStride, col0UV&15, row0UV&15, filt,
				uOrVSlice_16(ref1, pi), base1, ref1.UVStride, col1UV&15, row1UV&15, filt,
				uvBsW, uvBsH, predUV, uvBsW,
			)
		} else if ref0 != nil {
			base0 := uOrVBase_16(ref0, pi) + (uvRow+(row0UV>>4))*ref0.UVStride + (uvCol + (col0UV >> 4))
			mc16.Predict(uOrVSlice_16(ref0, pi), base0, ref0.UVStride, uvBsW, uvBsH,
				col0UV&15, row0UV&15, filt, predUV, uvBsW)
		}

		if !bm.Skip {
			addResidualUV_16(br, probs, counts, uvTxSize, lossless, dcQ, acQ, predUV, uvBsW, uvBsH,
				aboveRow, leftCol, col4, row4, pi+1) // plane 1=U, 2=V
		}
		writePlane_16(dstBuf, dstStride, dstBase, uvRow, uvCol, predUV, uvBsW, uvBsH)
	}
}

func addResidualUniform_16(pred []uint16, predStride, txBs int, bias int64) {
	b := int(bias)
	for r := 0; r < txBs; r++ {
		row := pred[r*predStride : r*predStride+txBs]
		for i, px := range row {
			v := int(px) + b
			if v < 0 {
				v = 0
			} else if v > MaxVal16 {
				v = MaxVal16
			}
			row[i] = uint16(v)
		}
	}
}

func addResidualY_16(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	txSize uint8, txType transform.TxType, lossless bool, dcQ, acQ int16,
	pred []uint16, bsW, bsH int,
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

			coeffs := coeftoken.DecodeBlock16(br, int(txSize), 0, refType, scan, nb, initCtx, &probs.CoefProbs, counts)
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

			// DC-only fast path: skip DequantBlock, full InvTransform2D, and indexed addResidual_16.
			if !lossless && txType == transform.DCT_DCT && dcOnly {
				if coeffs[0] != 0 {
					dqDC := dequantDC_16(coeffs[0], dcQ, txSize)
					predOff := ty*txBs*bsW + tx*txBs
					addResidualUniform_16(pred[predOff:], bsW, txBs, transform.DCOnlyValue16(dqDC, transform.FinalShift(txBs)))
				}
				continue
			}

			dequant.DequantBlock16(coeffs, dcQ, acQ, dqShift(txSize))

			if lossless {
				wht.IWHT4x4_16(coeffs)
			} else {
				transform.InvTransform2D16(coeffs, txBs, txType)
			}

			// Add residual to prediction sub-block (pred stride = bsW).
			predOff := ty*txBs*bsW + tx*txBs
			addResidual_16(pred[predOff:], bsW, coeffs, txBs)
		}
	}
}

func addResidualUV_16(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts,
	txSize uint8, lossless bool, dcQ, acQ int16,
	pred []uint16, bsW, bsH int,
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
			coeffs := coeftoken.DecodeBlock16(br, int(txSize), 1, 1, scan, nb, initCtx, &probs.CoefProbs, counts)
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
					dqDC := dequantDC_16(coeffs[0], dcQ, txSize)
					predOff := ty*txBs*bsW + tx*txBs
					addResidualUniform_16(pred[predOff:], bsW, txBs, transform.DCOnlyValue16(dqDC, transform.FinalShift(txBs)))
				}
				continue
			}

			dequant.DequantBlock16(coeffs, dcQ, acQ, dqShift(txSize))
			if lossless {
				wht.IWHT4x4_16(coeffs)
			} else {
				transform.InvTransform2D16(coeffs, txBs, transform.DCT_DCT)
			}
			predOff := ty*txBs*bsW + tx*txBs
			addResidual_16(pred[predOff:], bsW, coeffs, txBs)
		}
	}
}

func addResidual_16(pred []uint16, predStride int, coeffs []int64, txBs int) {
	for r := 0; r < txBs; r++ {
		for c := 0; c < txBs; c++ {
			v := int(pred[r*predStride+c]) + int(coeffs[r*txBs+c])
			if v < 0 {
				v = 0
			} else if v > MaxVal16 {
				v = MaxVal16
			}
			pred[r*predStride+c] = uint16(v)
		}
	}
}

func gatherBordersY_16(f *refbuf.Frame16, row, col, bsW, bsH int, above, left []uint16) uint16 {
	return gatherBorders_16(f.Y, f.YStride, f.YBase(), f.Width, f.Height, row, col, bsW, bsH, above, left,
		TileColStart4*4, TileRowStart4*4)
}

func gatherBordersUV_16(buf []uint16, stride, base, w, h, row, col, bsW, bsH int, above, left []uint16) uint16 {
	// Chroma is 4:2:0, so the tile boundary in chroma pixels is half the luma one.
	return gatherBorders_16(buf, stride, base, w, h, row, col, bsW, bsH, above, left,
		TileColStart4*2, TileRowStart4*2)
}

func gatherBorders_16(plane []uint16, stride, base, w, h, row, col, bsW, bsH int, above, left []uint16, leftBound, aboveBound int) uint16 {
	var topLeft uint16
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

func writePlane_16(buf []uint16, stride, base, row, col int, pred []uint16, bsW, bsH int) {
	for r := 0; r < bsH; r++ {
		startIdx := base + (row+r)*stride + col
		if startIdx >= len(buf) {
			break
		}
		copy(buf[startIdx:], pred[r*bsW:(r+1)*bsW])
	}
}

func uOrVBase_16(f *refbuf.Frame16, pi int) int {
	if pi == 0 {
		return f.UBase()
	}
	return f.VBase()
}

func uOrVSlice_16(f *refbuf.Frame16, pi int) []uint16 {
	if pi == 0 {
		return f.U
	}
	return f.V
}

func refFrame_16(refs [3]*refbuf.Frame16, rf int) *refbuf.Frame16 {
	if rf < 1 || rf > 3 {
		return nil
	}
	return refs[rf-1]
}

func dequantDC_16(coef int64, q int16, txSize uint8) int64 {
	p := coef * int64(q)
	if dqShift(txSize) > 0 {
		if p < 0 {
			p = -((-p) >> 1)
		} else {
			p >>= 1
		}
	}
	return p // no int16 clamp for high bit depth (libvpx highbd_check_range no-op)
}

func anyNonzero_16(c []int64) bool {
	for _, v := range c {
		if v != 0 {
			return true
		}
	}
	return false
}
