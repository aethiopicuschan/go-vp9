// Package coeftoken implements VP9 coefficient token decoding (spec §8.6.1).
// Reads one transform block's worth of quantized coefficients from a
// boolcoder.Reader, applying the band/context probability model.
package coeftoken

import (
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
)

// TX size indices (matching compheader.txSizes).
const (
	TX4x4  = 0
	TX8x8  = 1
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
	// Read extra bits MSB-first with category probabilities.
	val := catBase[cat]
	for _, p := range catProbs[cat] {
		val = (val << 1) | br.ReadBool(p)
	}
	return val
}

// ---- band tables (spec §8.6.1.1) -------------------------------------------

// Band assignment for 4×4 (indexed by scan position 0..15).
var band4 = [16]uint8{0, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3}

// Band assignment for 8×8 and larger (indexed by scan position).
// VP9 uses the same table for 8×8, 16×16, 32×32 (only first 64/256/1024 entries used).
var bandLarge [1025]uint8

func init() {
	// band k covers 5*2^(k-1) positions for k>0; band 0 = DC only.
	// For N=64/256/1024: band[0]=0, bands 1-5 distributed.
	breaks := [7]int{0, 1, 2, 3, 6, 11, 1025}
	b := uint8(0)
	for i := 0; i < 1025; i++ {
		if int(b) < 6 && i >= breaks[b+1] {
			b++
		}
		bandLarge[i] = b
	}
}

// band returns the coef band for coefficient scan position c in txSize.
func band(txSize, c int) uint8 {
	if txSize == TX4x4 {
		if c >= 16 {
			return 5
		}
		return band4[c]
	}
	if c >= 1025 {
		return 5
	}
	return bandLarge[c]
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

// DecodeBlock reads quantized coefficients for one transform block.
// Parameters:
//
//	br        – positioned at the first bit of the coefficient data
//	txSize    – TX4x4..TX32x32
//	planeType – 0=luma, 1=chroma
//	refType   – 0=intra, 1=inter
//	scan      – scan order table (len ≥ nCoefs(txSize))
//	initCtx   – initial context for position 0 (0, 1, or 2; from above/left blocks)
//	probs     – coefficient probability tables
//
// Returns the decoded coefficient array in raster order (scan de-applied).
// All-zero return means "skip" (every coef was zero or EOB at position 0).
func DecodeBlock(br *boolcoder.Reader, txSize, planeType, refType int,
	scan []int, initCtx int, probs *CoefProbs, counts *compheader.Counts) []int16 {

	n := nCoefs(txSize)
	out := make([]int16, n)
	ctx := initCtx
	pt := planeType
	rt := refType

	for c := 0; c < n; c++ {
		bd := int(band(txSize, c))
		if bd > 5 {
			bd = 5
		}
		ctxIdx := ctx
		if ctxIdx > 5 {
			ctxIdx = 5
		}
		p := probs[txSize][pt][rt][bd][ctxIdx]

		// Node 0: EOB?
		eob := br.ReadBool(p[0])
		if counts != nil {
			counts.CoefProbs[txSize][pt][rt][bd][ctxIdx][0][eob]++
		}
		if eob == 0 {
			break
		}
		// Node 1: ZERO?
		zero := br.ReadBool(p[1])
		if counts != nil {
			counts.CoefProbs[txSize][pt][rt][bd][ctxIdx][1][zero]++
		}
		if zero == 0 {
			ctx = prevCtx(0)
			continue
		}
		// Node 2: ONE vs MORE?
		var val int
		one := br.ReadBool(p[2])
		if counts != nil {
			counts.CoefProbs[txSize][pt][rt][bd][ctxIdx][2][one]++
		}
		if one == 0 {
			val = 1
		} else {
			val = readMoreThanOne(br)
		}
		// Sign bit (0=positive, 1=negative)
		if br.ReadBit() == 1 {
			val = -val
		}
		rasterIdx := scan[c]
		out[rasterIdx] = int16(val)
		ctx = prevCtx(val)
	}
	return out
}
