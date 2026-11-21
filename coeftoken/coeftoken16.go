package coeftoken

import (
	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/compheader"
)

// High-bit-depth coefficient token decode: identical to DecodeBlock but emits
// int32 coefficients (CAT6 for 10/12-bit exceeds int16) and uses the
// bit-depth-extended CAT6 probability table. Set BD to the frame bit depth.

// BD is the current bit depth (8/10/12) for highbd coefficient decode.
var BD = 8

// CAT6 probs extended with leading 255-prob bits for 10/12-bit (libvpx
// vp9_cat6_prob_high10 / _high12: 2 / 4 extra high-probability MSBs).
var cat6Probs10 = []uint8{255, 255, 254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129}
var cat6Probs12 = []uint8{255, 255, 255, 255, 254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129}

func cat6ProbsBD() []uint8 {
	switch BD {
	case 10:
		return cat6Probs10
	case 12:
		return cat6Probs12
	default:
		return catProbs[5]
	}
}

func DecodeBlock16EOB(br *boolcoder.Reader, txSize, planeType, refType int,
	scan, nb []int16, initCtx int, probs *CoefProbs, counts *compheader.Counts) (coeffs []int64, eob int) {

	n := nCoefs(txSize)
	out := make([]int64, n)
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

		// EOB node: 0 → end of block. Counted for backward adaptation: node 0
		// is {EOB_MODEL_TOKEN, eob_branch - EOB_MODEL_TOKEN} (libvpx
		// adapt_coef_probs), i.e. only positions where the EOB check is coded.
		var cnt *[3][2]uint32
		if counts != nil {
			cnt = &counts.CoefProbs[txSize][pt][rt][bd][ctxIdx]
		}
		moreCoefs := br.ReadBool(p[0])
		if cnt != nil {
			cnt[0][moreCoefs]++
		}
		if moreCoefs == 0 {
			break
		}

		// ZERO loop: consume runs of zero tokens; each skips re-reading the EOB
		// node for the next position (libvpx inner while).
		for br.ReadBool(p[1]) == 0 {
			if cnt != nil {
				cnt[1][0]++ // ZERO_TOKEN
			}
			tokenCache[scan[c]] = 0
			c++
			if c >= n {
				return out, c
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
			if counts != nil {
				cnt = &counts.CoefProbs[txSize][pt][rt][bd][ctxIdx]
			}
		}
		if cnt != nil {
			cnt[1][1]++ // non-zero token
		}

		// Non-zero token: decode value + energy class.
		var val int
		var energy uint8
		if br.ReadBool(p[2]) == 0 {
			if cnt != nil {
				cnt[2][0]++ // ONE_TOKEN
			}
			val = 1
			energy = 1
		} else {
			if cnt != nil {
				cnt[2][1]++ // TWO_TOKEN (two or more)
			}
			pp := &pareto8Full[p[2]-1]
			if br.ReadBool(pp[0]) == 1 {
				// CAT tokens (val ≥ 5).
				if br.ReadBool(pp[3]) == 1 {
					energy = 5
					if br.ReadBool(pp[5]) == 1 {
						if br.ReadBool(pp[7]) == 1 {
							val = catBase[5] + readCoeffExtra(br, cat6ProbsBD()) // CAT6 (bd-extended)
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
		out[scan[c]] = int64(val)
		tokenCache[scan[c]] = energy
		c++
		if c < n {
			ctx = getCoefContext(nb, tokenCache, c)
		}
	}
	return out, c
}

// DecodeBlock16 is DecodeBlock16EOB without the eob.
func DecodeBlock16(br *boolcoder.Reader, txSize, planeType, refType int,
	scan, nb []int16, initCtx int, probs *CoefProbs, counts *compheader.Counts) []int64 {
	out, _ := DecodeBlock16EOB(br, txSize, planeType, refType, scan, nb, initCtx, probs, counts)
	return out
}
