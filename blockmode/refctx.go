package blockmode

import "github.com/mgvs/go-vp9/ctx"

// ---- Reference frame context functions (spec §8.3.3 / libvpx vp9_pred_common.c) ----

// CompModeCtx is a faithful port of libvpx vp9_get_reference_mode_context — the
// context (0..4) for the per-block comp_mode (single vs compound) flag, used
// only when the frame reference_mode is SELECT.
func CompModeCtx(fi FrameInfo,
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) int {

	fixedRef := fi.CompFixedRef
	aboveComp := hasAbove && aboveRef[1] != ctx.RefIntra // has_second_ref(above)
	leftComp := hasLeft && leftRef[1] != ctx.RefIntra

	switch {
	case hasAbove && hasLeft: // both edges available
		switch {
		case !aboveComp && !leftComp: // neither edge uses comp pred (0/1)
			return boolToInt(aboveRef[0] == fixedRef) ^ boolToInt(leftRef[0] == fixedRef)
		case !aboveComp: // above single, left comp (2/3)
			return 2 + boolToInt(aboveRef[0] == fixedRef || !aboveInter)
		case !leftComp: // left single, above comp (2/3)
			return 2 + boolToInt(leftRef[0] == fixedRef || !leftInter)
		default: // both edges use comp pred (4)
			return 4
		}
	case hasAbove || hasLeft: // one edge available
		edgeRef, edgeComp := aboveRef, aboveComp
		if hasLeft {
			edgeRef, edgeComp = leftRef, leftComp
		}
		if !edgeComp { // edge does not use comp pred (0/1)
			return boolToInt(edgeRef[0] == fixedRef)
		}
		return 3 // edge uses comp pred
	default: // no edges available
		return 1
	}
}

// SingleRefCtxP1 returns context (0..4) for the first bit of single-reference
// decode: LAST (0) vs GOLDEN/ALTREF (1).
// Indexes SingleRefProbs[ctx][0].
func SingleRefCtxP1(
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) int {

	// libvpx vp9_get_pred_context_single_ref_p1 — all comparisons against
	// LAST_FRAME; has_second_ref ⇔ ref_frame[1] > INTRA_FRAME.
	isLast := func(r int) bool { return r == ctx.RefLast }
	aboveSecond := aboveRef[1] > ctx.RefIntra
	leftSecond := leftRef[1] > ctx.RefIntra

	if hasAbove && hasLeft {
		aboveIntra, leftIntra := !aboveInter, !leftInter
		if aboveIntra && leftIntra {
			return 2
		}
		if aboveIntra || leftIntra {
			er, es := aboveRef, aboveSecond
			if aboveIntra {
				er, es = leftRef, leftSecond
			}
			if !es {
				return 4 * boolToInt(isLast(er[0]))
			}
			return 1 + boolToInt(isLast(er[0]) || isLast(er[1]))
		}
		if aboveSecond && leftSecond {
			return 1 + boolToInt(isLast(aboveRef[0]) || isLast(aboveRef[1]) ||
				isLast(leftRef[0]) || isLast(leftRef[1]))
		}
		if aboveSecond || leftSecond {
			rfs := aboveRef[0]
			if aboveSecond {
				rfs = leftRef[0]
			}
			crf1, crf2 := leftRef[0], leftRef[1]
			if aboveSecond {
				crf1, crf2 = aboveRef[0], aboveRef[1]
			}
			if isLast(rfs) {
				return 3 + boolToInt(isLast(crf1) || isLast(crf2))
			}
			return boolToInt(isLast(crf1) || isLast(crf2))
		}
		return 2*boolToInt(isLast(aboveRef[0])) + 2*boolToInt(isLast(leftRef[0]))
	}

	if hasAbove || hasLeft {
		er, es, inter := aboveRef, aboveSecond, aboveInter
		if hasLeft {
			er, es, inter = leftRef, leftSecond, leftInter
		}
		if !inter {
			return 2
		}
		if !es {
			return 4 * boolToInt(isLast(er[0]))
		}
		return 1 + boolToInt(isLast(er[0]) || isLast(er[1]))
	}

	return 2
}

// SingleRefCtxP2 returns context (0..4) for the second bit of single-reference
// decode (given bit0=1): GOLDEN (0) vs ALTREF (1).
// Indexes SingleRefProbs[ctx][1].
func SingleRefCtxP2(
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) int {

	// Faithful port of libvpx vp9_get_pred_context_single_ref_p2.
	isLast := func(r int) bool { return r == ctx.RefLast }
	isGold := func(r int) bool { return r == ctx.RefGolden }
	isAlt := func(r int) bool { return r == ctx.RefAltRef }
	aboveSecond := aboveRef[1] > ctx.RefIntra
	leftSecond := leftRef[1] > ctx.RefIntra

	if hasAbove && hasLeft {
		aboveIntra, leftIntra := !aboveInter, !leftInter
		if aboveIntra && leftIntra {
			return 2
		}
		if aboveIntra || leftIntra {
			er, es := aboveRef, aboveSecond
			if aboveIntra {
				er, es = leftRef, leftSecond
			}
			if !es {
				if isLast(er[0]) {
					return 3
				}
				return 4 * boolToInt(isGold(er[0]))
			}
			return 1 + 2*boolToInt(isGold(er[0]) || isGold(er[1]))
		}
		// inter/inter
		if aboveSecond && leftSecond {
			if aboveRef[0] == leftRef[0] && aboveRef[1] == leftRef[1] {
				return 3 * boolToInt(isGold(aboveRef[0]) || isGold(aboveRef[1]) ||
					isGold(leftRef[0]) || isGold(leftRef[1]))
			}
			return 2
		}
		if aboveSecond || leftSecond {
			rfs := aboveRef[0]
			if aboveSecond {
				rfs = leftRef[0]
			}
			crf1, crf2 := leftRef[0], leftRef[1]
			if aboveSecond {
				crf1, crf2 = aboveRef[0], aboveRef[1]
			}
			if isGold(rfs) {
				return 3 + boolToInt(isGold(crf1) || isGold(crf2))
			}
			if isAlt(rfs) {
				return boolToInt(isGold(crf1) || isGold(crf2))
			}
			return 1 + 2*boolToInt(isGold(crf1) || isGold(crf2))
		}
		if isLast(aboveRef[0]) && isLast(leftRef[0]) {
			return 3
		}
		if isLast(aboveRef[0]) || isLast(leftRef[0]) {
			edge0 := aboveRef[0]
			if isLast(aboveRef[0]) {
				edge0 = leftRef[0]
			}
			return 4 * boolToInt(isGold(edge0))
		}
		return 2*boolToInt(isGold(aboveRef[0])) + 2*boolToInt(isGold(leftRef[0]))
	}

	if hasAbove || hasLeft {
		er, es, inter := aboveRef, aboveSecond, aboveInter
		if hasLeft {
			er, es, inter = leftRef, leftSecond, leftInter
		}
		if !inter || (isLast(er[0]) && !es) {
			return 2
		}
		if !es {
			return 4 * boolToInt(isGold(er[0]))
		}
		return 3 * boolToInt(isGold(er[0]) || isGold(er[1]))
	}

	return 2
}

// CompRefCtx returns context (0..4) for the compound-reference variable-ref
// bit (compVarRef[0] vs compVarRef[1]). Faithful port of libvpx
// vp9_get_pred_context_comp_ref_p. var_ref_idx is the neighbour ref_frame slot
// holding the variable ref (= !sign_bias[fixed]).
func CompRefCtx(fi FrameInfo,
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) int {

	fixedRef := fi.CompFixedRef
	varRef0 := fi.CompVarRef[0]
	varRef1 := fi.CompVarRef[1]
	varIdx := 1 - fi.CompFixedIdx

	aboveComp := hasAbove && aboveRef[1] != ctx.RefIntra
	leftComp := hasLeft && leftRef[1] != ctx.RefIntra

	switch {
	case hasAbove && hasLeft: // both edges available
		switch {
		case !aboveInter && !leftInter: // intra/intra
			return 2
		case !aboveInter || !leftInter: // intra/inter
			edgeRef, edgeComp := aboveRef, aboveComp
			if !aboveInter {
				edgeRef, edgeComp = leftRef, leftComp
			}
			if !edgeComp { // single pred
				return 1 + 2*boolToInt(edgeRef[0] != varRef1)
			}
			return 1 + 2*boolToInt(edgeRef[varIdx] != varRef1)
		default: // inter/inter
			aSg := !aboveComp
			lSg := !leftComp
			vrfa := aboveRef[varIdx]
			if aSg {
				vrfa = aboveRef[0]
			}
			vrfl := leftRef[varIdx]
			if lSg {
				vrfl = leftRef[0]
			}
			switch {
			case vrfa == vrfl && varRef1 == vrfa:
				return 0
			case lSg && aSg: // single/single
				if (vrfa == fixedRef && vrfl == varRef0) || (vrfl == fixedRef && vrfa == varRef0) {
					return 4
				} else if vrfa == vrfl {
					return 3
				}
				return 1
			case lSg || aSg: // single/comp
				vrfc := vrfl // the compound edge's var ref
				if lSg {
					vrfc = vrfa
				}
				rfs := vrfl // the single edge's ref
				if aSg {
					rfs = vrfa
				}
				switch {
				case vrfc == varRef1 && rfs != varRef1:
					return 1
				case rfs == varRef1 && vrfc != varRef1:
					return 2
				default:
					return 4
				}
			case vrfa == vrfl: // comp/comp
				return 4
			default:
				return 2
			}
		}
	case hasAbove || hasLeft: // one edge available
		edgeRef, edgeComp, edgeInter := aboveRef, aboveComp, aboveInter
		if hasLeft {
			edgeRef, edgeComp, edgeInter = leftRef, leftComp, leftInter
		}
		if !edgeInter {
			return 2
		}
		if edgeComp {
			return 4 * boolToInt(edgeRef[varIdx] != varRef1)
		}
		return 3 * boolToInt(edgeRef[0] != varRef1)
	default: // no edges available
		return 2
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
