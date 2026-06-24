package blockmode

import "github.com/mgvs/go-vp9/ctx"

// ---- Reference frame context functions (spec §8.3.3 / libvpx vp9_pred_common.c) ----

// CompModeCtx returns the compound-vs-single reference mode context (0..4)
// for CompInterProbs (compInterContexts=5).
//
// compFixedRef is the "fixed" reference frame for compound prediction.
func CompModeCtx(fi FrameInfo,
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) int {

	fixedRef := fi.CompFixedRef

	aboveHasSecond := hasAbove && aboveRef[1] != ctx.RefIntra
	leftHasSecond := hasLeft && leftRef[1] != ctx.RefIntra

	if hasAbove && hasLeft {
		if !aboveHasSecond && !leftHasSecond {
			// both single
			sameFixed := func(r int) bool { return r == fixedRef }
			return boolToInt(sameFixed(aboveRef[0])) ^ boolToInt(sameFixed(leftRef[0]))
		}
		if aboveHasSecond && leftHasSecond {
			// both compound
			return 4
		}
		// one compound, one single
		singleRef := aboveRef[0]
		if leftHasSecond {
			singleRef = aboveRef[0]
		} else {
			singleRef = leftRef[0]
		}
		_ = singleRef
		compRef0, compRef1 := aboveRef[0], aboveRef[1]
		if leftHasSecond {
			compRef0, compRef1 = leftRef[0], leftRef[1]
		}
		otherSingle := leftRef[0]
		if leftHasSecond {
			otherSingle = aboveRef[0]
		}
		hasFixed := compRef0 == fixedRef || compRef1 == fixedRef
		return 2 + boolToInt(hasFixed || otherSingle == fixedRef || otherSingle == ctx.RefIntra)
	}

	if hasAbove || hasLeft {
		ref := aboveRef
		hasSecond := aboveHasSecond
		if hasLeft {
			ref = leftRef
			hasSecond = leftHasSecond
		}
		if !aboveInter && !leftInter {
			return 2
		}
		if hasSecond {
			return 3
		}
		return boolToInt(ref[0] == fixedRef)
	}

	return 0
}

// SingleRefCtxP1 returns context (0..4) for the first bit of single-reference
// decode: LAST (0) vs GOLDEN/ALTREF (1).
// Indexes SingleRefProbs[ctx][0].
func SingleRefCtxP1(
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) int {

	aboveHasSecond := hasAbove && aboveRef[1] != ctx.RefIntra
	leftHasSecond := hasLeft && leftRef[1] != ctx.RefIntra

	isAltRef := func(r int) bool { return r == ctx.RefAltRef }

	if hasAbove && hasLeft {
		if aboveInter && leftInter {
			if aboveHasSecond && leftHasSecond {
				a := isAltRef(aboveRef[0]) || isAltRef(aboveRef[1])
				l := isAltRef(leftRef[0]) || isAltRef(leftRef[1])
				return boolToInt(a && l)
			}
			if aboveHasSecond {
				a := isAltRef(aboveRef[0]) || isAltRef(aboveRef[1])
				return 2 + boolToInt(a || isAltRef(leftRef[0]))
			}
			if leftHasSecond {
				l := isAltRef(leftRef[0]) || isAltRef(leftRef[1])
				return 2 + boolToInt(l || isAltRef(aboveRef[0]))
			}
			return boolToInt(isAltRef(aboveRef[0])) ^ boolToInt(isAltRef(leftRef[0]))
		}
		if aboveInter || leftInter {
			mbmiRef := aboveRef
			hasSecond := aboveHasSecond
			if leftInter {
				mbmiRef = leftRef
				hasSecond = leftHasSecond
			}
			if hasSecond {
				return 4
			}
			return 2 + boolToInt(isAltRef(mbmiRef[0]))
		}
		return 2
	}

	if hasAbove || hasLeft {
		inter := (hasAbove && aboveInter) || (hasLeft && leftInter)
		if !inter {
			return 2
		}
		ref := aboveRef
		hasSecond := aboveHasSecond
		if hasLeft {
			ref = leftRef
			hasSecond = leftHasSecond
		}
		if hasSecond {
			return 4
		}
		return 2 * boolToInt(isAltRef(ref[0]))
	}

	return 2
}

// SingleRefCtxP2 returns context (0..4) for the second bit of single-reference
// decode (given bit0=1): GOLDEN (0) vs ALTREF (1).
// Indexes SingleRefProbs[ctx][1].
func SingleRefCtxP2(
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) int {

	aboveHasSecond := hasAbove && aboveRef[1] != ctx.RefIntra
	leftHasSecond := hasLeft && leftRef[1] != ctx.RefIntra

	isGolden := func(r int) bool { return r == ctx.RefGolden }

	if hasAbove && hasLeft {
		if aboveInter && leftInter {
			if aboveHasSecond && leftHasSecond {
				a := isGolden(aboveRef[0]) || isGolden(aboveRef[1])
				l := isGolden(leftRef[0]) || isGolden(leftRef[1])
				return boolToInt(a && l)
			}
			if aboveHasSecond {
				a := isGolden(aboveRef[0]) || isGolden(aboveRef[1])
				return 2 + boolToInt(a || isGolden(leftRef[0]))
			}
			if leftHasSecond {
				l := isGolden(leftRef[0]) || isGolden(leftRef[1])
				return 2 + boolToInt(l || isGolden(aboveRef[0]))
			}
			return boolToInt(isGolden(aboveRef[0])) ^ boolToInt(isGolden(leftRef[0]))
		}
		if aboveInter || leftInter {
			mbmiRef := aboveRef
			hasSecond := aboveHasSecond
			if leftInter {
				mbmiRef = leftRef
				hasSecond = leftHasSecond
			}
			if hasSecond {
				return 4
			}
			return 2 + boolToInt(isGolden(mbmiRef[0]))
		}
		return 2
	}

	if hasAbove || hasLeft {
		inter := (hasAbove && aboveInter) || (hasLeft && leftInter)
		if !inter {
			return 2
		}
		ref := aboveRef
		hasSecond := aboveHasSecond
		if hasLeft {
			ref = leftRef
			hasSecond = leftHasSecond
		}
		if hasSecond {
			return 4
		}
		return 2 * boolToInt(isGolden(ref[0]))
	}

	return 2
}

// CompRefCtx returns context (0..4) for the compound-reference variable-ref
// bit: compVarRef[0] (0) vs compVarRef[1] (1).
// Indexes CompRefProbs[ctx].
func CompRefCtx(fi FrameInfo,
	hasAbove, hasLeft bool, aboveRef, leftRef [2]int, aboveInter, leftInter bool) int {

	varRef0 := fi.CompVarRef[0]
	varRef1 := fi.CompVarRef[1]

	isVarRef0 := func(r int) bool { return r == varRef0 }
	isVarRef1 := func(r int) bool { return r == varRef1 }

	aboveHasSecond := hasAbove && aboveRef[1] != ctx.RefIntra
	leftHasSecond := hasLeft && leftRef[1] != ctx.RefIntra

	if hasAbove && hasLeft {
		if aboveInter && leftInter {
			if aboveHasSecond && leftHasSecond {
				a := isVarRef0(aboveRef[0]) || isVarRef0(aboveRef[1])
				l := isVarRef0(leftRef[0]) || isVarRef0(leftRef[1])
				return boolToInt(a && l)
			}
			if aboveHasSecond {
				a := isVarRef0(aboveRef[0]) || isVarRef0(aboveRef[1])
				return 2 + boolToInt(a || isVarRef1(leftRef[0]))
			}
			if leftHasSecond {
				l := isVarRef0(leftRef[0]) || isVarRef0(leftRef[1])
				return 2 + boolToInt(l || isVarRef1(aboveRef[0]))
			}
			return boolToInt(isVarRef0(aboveRef[0])) ^ boolToInt(isVarRef0(leftRef[0]))
		}
		if aboveInter || leftInter {
			mbmiRef := aboveRef
			hasSecond := aboveHasSecond
			if leftInter {
				mbmiRef = leftRef
				hasSecond = leftHasSecond
			}
			if hasSecond {
				return 4
			}
			return 2 + boolToInt(isVarRef0(mbmiRef[0]) || isVarRef1(mbmiRef[0]))
		}
		return 2
	}

	if hasAbove || hasLeft {
		inter := (hasAbove && aboveInter) || (hasLeft && leftInter)
		if !inter {
			return 2
		}
		ref := aboveRef
		hasSecond := aboveHasSecond
		if hasLeft {
			ref = leftRef
			hasSecond = leftHasSecond
		}
		if hasSecond {
			return 4
		}
		return 4 * boolToInt(isVarRef0(ref[0]))
	}

	return 2
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
