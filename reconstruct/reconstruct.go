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
	transform.DCT_DCT,   // D207
	transform.DCT_DCT,   // D63
	transform.ADST_ADST, // TM
}

// TxTypeFor returns the inverse-transform type for the luma plane.
// Chroma always uses DCT_DCT (call with isInter=true or any value).
func TxTypeFor(yMode modeinfo.IntraMode, txSize uint8, isInter bool) transform.TxType {
	if isInter || txSize >= ctx.TX32x32 {
		return transform.DCT_DCT
	}
	return intraModeTxType[yMode]
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
		var subAboveBuf [8]byte  // 2×4
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
				intra.Predict(subMode, subPred, 4, 4, subAbove, subLeft, subTL)

				if !bm.Skip {
					txType := TxTypeFor(subMode, ctx.TX4x4, false)
					subRow4 := row4 + ty
					subCol4 := col4 + tx
					initCtx := nzcInitCtx(aboveRow, leftCol, subCol4, subRow4)
					scan := ScanFor(ctx.TX4x4, txType)
					coeffs := coeftoken.DecodeBlock(br, ctx.TX4x4, 0, 0, scan, initCtx, &probs.CoefProbs, counts)
					if coeffs != nil {
						nzc := false
						for _, v := range coeffs {
							if v != 0 {
								nzc = true
								break
							}
						}
						updateNZC(aboveRow, leftCol, subCol4, subRow4, 1, nzc, 0)
						dequant.DequantBlock(coeffs, dc[0], ac[0])
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
		var aboveBuf [128]byte // 2×64
		var leftBuf [64]byte
		above := aboveBuf[:2*bsW]
		left := leftBuf[:bsH]
		tl := gatherBordersY(dst, row, col, bsW, bsH, above, left)
		intra.Predict(bm.YMode, pred, bsW, bsH, above, left, tl)

		if !bm.Skip {
			txType := TxTypeFor(bm.YMode, bm.TxSize, false)
			addResidualY(br, probs, counts, bm.TxSize, txType, lossless, dc[0], ac[0],
				pred, bsW, bsH, aboveRow, leftCol, row4, col4)
		}
		writePlane(dst.Y, dst.YStride, dst.YBase(), row, col, pred, bsW, bsH)
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

	var predUVBuf [32 * 32]byte
	var aboveUVBuf [64]byte // 2×32
	var leftUVBuf [32]byte
	predUV := predUVBuf[:uvBsW*uvBsH]
	aboveUV := aboveUVBuf[:2*uvBsW]
	leftUV := leftUVBuf[:uvBsH]

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
		tlUV := gatherBordersUV(plane.buf, plane.stride, plane.base, uvRow, uvCol, uvBsW, uvBsH, aboveUV, leftUV)
		intra.Predict(plane.mode, predUV, uvBsW, uvBsH, aboveUV, leftUV, tlUV)

		if !bm.Skip {
			addResidualUV(br, probs, counts, uvTxSize, lossless, plane.dcQ, plane.acQ,
				predUV, uvBsW, uvBsH, aboveRow, leftCol, col4, row4, plane.planeIdx)
		}
		writePlane(plane.buf, plane.stride, plane.base, uvRow, uvCol, predUV, uvBsW, uvBsH)
	}
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

	if ref0 != nil && compound && ref1 != nil {
		base0 := ref0.YBase() + (row+row0/8)*ref0.YStride + (col + col0/8)
		base1 := ref1.YBase() + (row+row1/8)*ref1.YStride + (col + col1/8)
		mc.PredictComp(
			ref0.Y, base0, ref0.YStride, col0&7, row0&7, filt,
			ref1.Y, base1, ref1.YStride, col1&7, row1&7, filt,
			bsW, bsH, pred, bsW,
		)
	} else if ref0 != nil {
		base0 := ref0.YBase() + (row+row0/8)*ref0.YStride + (col + col0/8)
		mc.Predict(ref0.Y, base0, ref0.YStride, bsW, bsH, col0&7, row0&7, filt, pred, bsW)
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
	col0UV, row0UV := mvSubPelUV(mv0)
	col1UV, row1UV := mvSubPelUV(mv1)

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
			base0 := uOrVBase(ref0, pi) + (uvRow+row0UV/8)*ref0.UVStride + (uvCol + col0UV/8)
			base1 := uOrVBase(ref1, pi) + (uvRow+row1UV/8)*ref1.UVStride + (uvCol + col1UV/8)
			mc.PredictComp(
				uOrVSlice(ref0, pi), base0, ref0.UVStride, col0UV&7, row0UV&7, filt,
				uOrVSlice(ref1, pi), base1, ref1.UVStride, col1UV&7, row1UV&7, filt,
				uvBsW, uvBsH, predUV, uvBsW,
			)
		} else if ref0 != nil {
			base0 := uOrVBase(ref0, pi) + (uvRow+row0UV/8)*ref0.UVStride + (uvCol + col0UV/8)
			mc.Predict(uOrVSlice(ref0, pi), base0, ref0.UVStride, uvBsW, uvBsH,
				col0UV&7, row0UV&7, filt, predUV, uvBsW)
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
	txBs := 4 << txSize  // TX block size in pixels (square)
	numX := bsW / txBs   // TX sub-blocks per row
	numY := bsH / txBs   // TX sub-blocks per column
	txN4 := 1 << txSize  // TX size in 4×4 units

	scan := ScanFor(txSize, txType)
	refType := 0 // intra

	for ty := 0; ty < numY; ty++ {
		for tx := 0; tx < numX; tx++ {
			subRow4 := row4 + ty*txN4
			subCol4 := col4 + tx*txN4
			initCtx := nzcInitCtx(aboveRow, leftCol, subCol4, subRow4)

			coeffs := coeftoken.DecodeBlock(br, int(txSize), 0, refType, scan, initCtx, &probs.CoefProbs, counts)
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
					dqDC := int32(coeffs[0]) * int32(dcQ)
					if dqDC > 32767 {
						dqDC = 32767
					} else if dqDC < -32768 {
						dqDC = -32768
					}
					predOff := ty*txBs*bsW + tx*txBs
					addResidualUniform(pred[predOff:], bsW, txBs, transform.DCOnlyValue(int16(dqDC)))
				}
				continue
			}

			dequant.DequantBlock(coeffs, dcQ, acQ)

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
	scan := DefaultScanFor(txSize)
	// Each UV TX block covers 2 * txN4 luma 4×4 units per dimension (4:2:0).
	uvTxN4 := 2 << txSize

	for ty := 0; ty < numY; ty++ {
		for tx := 0; tx < numX; tx++ {
			subCol4 := col4 + tx*uvTxN4
			subRow4 := row4 + ty*uvTxN4
			initCtx := nzcInitCtxUV(aboveRow, leftCol, subCol4, subRow4, plane)
			coeffs := coeftoken.DecodeBlock(br, int(txSize), 1, 0, scan, initCtx, &probs.CoefProbs, counts)
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
					dqDC := int32(coeffs[0]) * int32(dcQ)
					if dqDC > 32767 {
						dqDC = 32767
					} else if dqDC < -32768 {
						dqDC = -32768
					}
					predOff := ty*txBs*bsW + tx*txBs
					addResidualUniform(pred[predOff:], bsW, txBs, transform.DCOnlyValue(int16(dqDC)))
				}
				continue
			}

			dequant.DequantBlock(coeffs, dcQ, acQ)
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

// gatherBordersY reads above-row (2*bsW pixels) and left-column (bsH pixels)
// pixel borders for intra prediction of a Y block at (row, col).
// above is always 2*bsW long to satisfy D45/D63 extended requirements;
// pixels beyond the frame edge are filled with the last available value.
// Caller must provide above[:2*bsW] and left[:bsH]; returns topLeft.
func gatherBordersY(f *refbuf.Frame, row, col, bsW, bsH int, above, left []byte) byte {
	var topLeft byte
	base := f.YBase()
	s := f.YStride
	if row > 0 {
		frameW := f.YStride
		for i := 0; i < 2*bsW; i++ {
			px := col + i
			if px >= frameW {
				px = frameW - 1
			}
			above[i] = f.Y[base+(row-1)*s+px]
		}
		if col > 0 {
			topLeft = f.Y[base+(row-1)*s+col-1]
		}
	} else {
		for i := range above {
			above[i] = 128
		}
		topLeft = 128
	}
	if col > 0 {
		for i := 0; i < bsH; i++ {
			idx := base + (row+i)*s + col - 1
			if idx >= len(f.Y) {
				left[i] = left[i-1]
			} else {
				left[i] = f.Y[idx]
			}
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
func gatherBordersUV(buf []byte, stride, base, row, col, bsW, bsH int, above, left []byte) byte {
	var topLeft byte
	if row > 0 {
		for i := 0; i < 2*bsW; i++ {
			px := col + i
			if px >= stride {
				px = stride - 1
			}
			above[i] = buf[base+(row-1)*stride+px]
		}
		if col > 0 {
			topLeft = buf[base+(row-1)*stride+col-1]
		}
	} else {
		for i := range above {
			above[i] = 128
		}
		topLeft = 128
	}
	if col > 0 {
		for i := 0; i < bsH; i++ {
			idx := base + (row+i)*stride + col - 1
			if idx >= len(buf) {
				left[i] = left[i-1]
			} else {
				left[i] = buf[idx]
			}
		}
	} else {
		for i := range left {
			left[i] = 128
		}
	}
	return topLeft
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
func nzcInitCtx(aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4 int) int {
	var an, ln bool
	if col4 < aboveRow.N4 {
		an = aboveRow.NZC[0][col4]
	}
	if row4 < leftCol.N4 {
		ln = leftCol.NZC[0][row4]
	}
	return ctx.NZCInitCtx(an, ln)
}

// nzcInitCtxUV returns the NZC init context for a UV TX sub-block.
// col4, row4 are luma 4×4 positions; plane is 1 (U) or 2 (V).
func nzcInitCtxUV(aboveRow *ctx.Row, leftCol *ctx.Col, col4, row4, plane int) int {
	var an, ln bool
	if col4 < aboveRow.N4 {
		an = aboveRow.NZC[plane][col4]
	}
	if row4 < leftCol.N4 {
		ln = leftCol.NZC[plane][row4]
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

