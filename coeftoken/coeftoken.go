// Package coeftoken implements VP9 coefficient token decoding (spec §8.6.1).
// Reads one transform block's worth of quantized coefficients from a
// boolcoder.Reader, applying the band/context probability model.
package coeftoken

import (
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
)

// ScanType selects the coefficient scan pattern for a TX block.
const (
	ScanDefault = 0 // diagonal (DCT_DCT, ADST_ADST)
	ScanRow     = 1 // row scan (DCT_ADST)
	ScanCol     = 2 // col scan (ADST_DCT)
)

// ScanOrder returns the scan-position→raster table and the matching neighbor
// table for a given TX size (0=4×4 … 3=32×32) and scan pattern. TX32x32 always
// uses the diagonal scan.
func ScanOrder(txSize, scanType int) (scan, nb []int16) {
	switch txSize {
	case 0:
		switch scanType {
		case ScanRow:
			return rowScan4x4, rowScanNb4x4
		case ScanCol:
			return colScan4x4, colScanNb4x4
		default:
			return defaultScan4x4, defaultScanNb4x4
		}
	case 1:
		switch scanType {
		case ScanRow:
			return rowScan8x8, rowScanNb8x8
		case ScanCol:
			return colScan8x8, colScanNb8x8
		default:
			return defaultScan8x8, defaultScanNb8x8
		}
	case 2:
		switch scanType {
		case ScanRow:
			return rowScan16x16, rowScanNb16x16
		case ScanCol:
			return colScan16x16, colScanNb16x16
		default:
			return defaultScan16x16, defaultScanNb16x16
		}
	default:
		return defaultScan32x32, defaultScanNb32x32
	}
}

// TX size indices (matching compheader.txSizes).
const (
	TX4x4   = 0
	TX8x8   = 1
	TX16x16 = 2
	TX32x32 = 3
)

// nCoefs returns the number of coefficient positions for a given TX size.
// Side = 4<<txSize, so coefs = side² = 16<<(2*txSize).
func nCoefs(txSize int) int {
	return 16 << (uint(txSize) * 2) // 16, 64, 256, 1024
}

// ---- coefficient value decoding (spec §8.6.1.3) ----------------------------

// Fixed extra-bit probabilities for DCT_VAL_CATEGORY1..6 (from libvpx).
var catProbs = [6][]uint8{
	{159},
	{165, 145},
	{173, 148, 140},
	{176, 155, 140, 135},
	{180, 157, 141, 134, 130},
	{254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129},
}

// catBase[k] is the base value for category k (0-indexed, k=0 → CAT1).
var catBase = [6]int{5, 7, 11, 19, 35, 67}

// readMoreThanOne reads a coefficient value ≥ 2 from the bitstream.
// The category is selected via a fixed tree (spec §8.6.1.3 / libvpx cat_prob).
func readMoreThanOne(br *boolcoder.Reader) int {
	var cat int
	if br.ReadBool(128) == 1 {
		if br.ReadBool(128) == 1 {
			// cat 4..5 (0-indexed)
			cat = 4
			if br.ReadBool(128) == 1 {
				cat = 5
			}
		} else {
			// cat 2..3
			cat = 2
			if br.ReadBool(128) == 1 {
				cat = 3
			}
		}
	} else {
		// cat 0..1 (CAT1..CAT2)
		cat = 0
		if br.ReadBool(128) == 1 {
			cat = 1
		}
	}
	// Read extra bits MSB-first with category probabilities, then add the
	// category base. The extra bits form an unsigned offset; catBase[cat] is the
	// smallest value in the category (spec §8.6.1.3 / libvpx read_coeff).
	extra := 0
	for _, p := range catProbs[cat] {
		extra = (extra << 1) | br.ReadBool(p)
	}
	return catBase[cat] + extra
}

// ---- band tables (spec §8.6.1.1) -------------------------------------------

// coefband4 maps a 4×4 scan position to its coefficient band (libvpx
// vp9_coefband_trans_4x4).
var coefband4 = [16]uint8{
	0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5,
}

// coefband8plus maps an 8×8/16×16/32×32 scan position to its band (libvpx
// vp9_coefband_trans_8x8plus). Indices beyond 1024 clamp to band 5.
var coefband8plus = [1024]uint8{
	0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
}

// band returns the coef band for coefficient scan position c in txSize.
func band(txSize, c int) uint8 {
	if txSize == TX4x4 {
		if c >= 16 {
			return 5
		}
		return coefband4[c]
	}
	if c >= 1024 {
		return 5
	}
	return coefband8plus[c]
}

// ---- context calculation (spec §8.6.1.2) ------------------------------------

// prevCtx returns the coefficient probability context for position c > 0
// based on the previous coefficient's token category.
// 0=zero, 1=one, 2=more-than-one.
func prevCtx(prevVal int) int {
	if prevVal == 0 {
		return 0
	}
	if prevVal == 1 {
		return 1
	}
	return 2
}

// ---- DecodeBlock ------------------------------------------------------------

// CoefProbs is the coefficient probability array shape from compheader.Probs.
// [txSize][planeType][refType][band][ctx][3 nodes]
type CoefProbs = [4][2][2][6][6][3]uint8

// getCoefContext computes the coefficient probability context for position c>0
// from the two scan neighbors' token energy classes (libvpx get_coef_context):
// (1 + tokenCache[nb0] + tokenCache[nb1]) >> 1.
func getCoefContext(nb []int16, tokenCache []uint8, c int) int {
	return (1 + int(tokenCache[nb[2*c]]) + int(tokenCache[nb[2*c+1]])) >> 1
}

// readCoeffExtra reads n extra bits MSB-first with the given category probs and
// returns the unsigned offset (libvpx read_coeff).
func readCoeffExtra(br *boolcoder.Reader, probs []uint8) int {
	val := 0
	for _, p := range probs {
		val = (val << 1) | br.ReadBool(p)
	}
	return val
}

// DecodeBlock reads quantized coefficients for one transform block, faithfully
// porting libvpx decode_coefs (spec §8.6.1). Returns coefficients in raster order.
//
//	scan      – scan-position → raster-index table for this txSize/txType
//	nb        – matching neighbor table (2 entries per scan position + pad)
//	initCtx   – context for position 0 (0/1/2 from above/left nonzero context)
//
// All-zero return means EOB at position 0.
func DecodeBlock(br *boolcoder.Reader, txSize, planeType, refType int,
	scan, nb []int16, initCtx int, probs *CoefProbs, counts *compheader.Counts) []int16 {

	n := nCoefs(txSize)
	out := make([]int16, n)
	tokenCache := make([]uint8, n)
	pt := planeType
	rt := refType
	ctx := initCtx

	c := 0
	for c < n {
		bd := int(band(txSize, c))
		if bd > 5 {
			bd = 5
		}
		ctxIdx := ctx
		if ctxIdx > 5 {
			ctxIdx = 5
		}
		p := &probs[txSize][pt][rt][bd][ctxIdx]

		// EOB node: 0 → end of block.
		if br.ReadBool(p[0]) == 0 {
			break
		}

		// ZERO loop: consume runs of zero tokens; each skips re-reading the EOB
		// node for the next position (libvpx inner while).
		for br.ReadBool(p[1]) == 0 {
			tokenCache[scan[c]] = 0
			c++
			if c >= n {
				return out
			}
			ctx = getCoefContext(nb, tokenCache, c)
			bd = int(band(txSize, c))
			if bd > 5 {
				bd = 5
			}
			ctxIdx = ctx
			if ctxIdx > 5 {
				ctxIdx = 5
			}
			p = &probs[txSize][pt][rt][bd][ctxIdx]
		}

		// Non-zero token: decode value + energy class.
		var val int
		var energy uint8
		if br.ReadBool(p[2]) == 0 {
			val = 1
			energy = 1
		} else {
			pp := &pareto8Full[p[2]-1]
			if br.ReadBool(pp[0]) == 1 {
				// CAT tokens (val ≥ 5).
				if br.ReadBool(pp[3]) == 1 {
					energy = 5
					if br.ReadBool(pp[5]) == 1 {
						if br.ReadBool(pp[7]) == 1 {
							val = catBase[5] + readCoeffExtra(br, catProbs[5]) // CAT6
						} else {
							val = catBase[4] + readCoeffExtra(br, catProbs[4]) // CAT5
						}
					} else if br.ReadBool(pp[6]) == 1 {
						val = catBase[3] + readCoeffExtra(br, catProbs[3]) // CAT4
					} else {
						val = catBase[2] + readCoeffExtra(br, catProbs[2]) // CAT3
					}
				} else {
					energy = 4
					if br.ReadBool(pp[4]) == 1 {
						val = catBase[1] + readCoeffExtra(br, catProbs[1]) // CAT2
					} else {
						val = catBase[0] + readCoeffExtra(br, catProbs[0]) // CAT1
					}
				}
			} else {
				// Values 2, 3, 4.
				if br.ReadBool(pp[1]) == 1 {
					energy = 3
					val = 3 + br.ReadBool(pp[2]) // 3 or 4
				} else {
					energy = 2
					val = 2
				}
			}
		}
		// Sign bit.
		if br.ReadBit() == 1 {
			val = -val
		}
		out[scan[c]] = int16(val)
		tokenCache[scan[c]] = energy
		c++
		if c < n {
			ctx = getCoefContext(nb, tokenCache, c)
		}
	}
	return out
}
