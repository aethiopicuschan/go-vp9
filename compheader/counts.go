package compheader

// Counts accumulates per-symbol binary counts for all VP9 adaptive entropy
// tables during one frame decode.  After the frame, AdaptProbs uses these
// counts to update the probability tables (spec §8.9 / libvpx vp9_adapt_probs).
//
// Each leaf is [2]uint32{count_zero, count_one}: how many times the binary
// decision resolved to 0 or 1 respectively.
type Counts struct {
	TxProbs8          [txSizeContexts][2]uint32
	TxProbs16         [txSizeContexts][2][2]uint32
	TxProbs32         [txSizeContexts][3][2]uint32
	CoefProbs         [txSizes][planeTypes][refTypes][coefBands][prevCoefContexts][uncostrainedNodes][2]uint32
	SkipProbs         [skipContexts][2]uint32
	InterModeProbs    [interModeContexts][interModeDelta][2]uint32
	InterpFilterProbs [switchFilterCtx][switchFilterDelta][2]uint32
	IsInterProbs      [isInterContexts][2]uint32
	CompInterProbs    [compInterContexts][2]uint32
	SingleRefProbs    [refContexts][2][2]uint32
	CompRefProbs      [refContexts][2]uint32
	YModeProbs        [blockSizeGroups][intraModeDelta][2]uint32
	UVModeProbs       [intraModes][intraModeDelta][2]uint32
	PartitionProbs    [partitionContexts][partitionDelta][2]uint32
	MVJointProbs      [mvJointDelta][2]uint32
	MVSignProbs       [mvComponents][2]uint32
	MVClassProbs      [mvComponents][mvClassDelta][2]uint32
	MVClass0BitProbs  [mvComponents][2]uint32
	MVBitsProbs       [mvComponents][mvOffsetBits][2]uint32
	MVClass0FrProbs   [mvComponents][class0FrContexts][mvFrDelta][2]uint32
	MVFrProbs         [mvComponents][mvFrDelta][2]uint32
	MVClass0HPProbs   [mvComponents][2]uint32
	MVHPProbs         [mvComponents][2]uint32
}

// AdaptProbs updates all non-nil probability entries using the accumulated
// counts from one inter frame decode (spec §8.9.2 / libvpx merge_probs).
//
// full selects libvpx's vp9_adapt_mode_probs + vp9_adapt_mv_probs (tx, skip,
// modes, partition, mv) which run only for inter frames; coefficient probs
// (vp9_adapt_coef_probs) are always adapted. Intra (key/intra-only) frames pass
// full=false so they adapt only the coef probs and leave mode/mv at the loaded
// context (libvpx gates adapt_mode/mv_probs on !frame_is_intra_only).
func AdaptProbs(probs *Probs, counts *Counts, full bool, coefCountSat, coefUpdateFactor uint32) {
	if counts == nil {
		return
	}

	if full {
		// TX size probs.
		for i := range probs.TxProbs8 {
			probs.TxProbs8[i] = adaptProb(probs.TxProbs8[i], counts.TxProbs8[i])
		}
		for i := range probs.TxProbs16 {
			for j := range probs.TxProbs16[i] {
				probs.TxProbs16[i][j] = adaptProb(probs.TxProbs16[i][j], counts.TxProbs16[i][j])
			}
		}
		for i := range probs.TxProbs32 {
			for j := range probs.TxProbs32[i] {
				probs.TxProbs32[i][j] = adaptProb(probs.TxProbs32[i][j], counts.TxProbs32[i][j])
			}
		}
	}

	// Coefficient probs (adapted for all frame types).
	for i := range probs.CoefProbs {
		for j := range probs.CoefProbs[i] {
			for k := range probs.CoefProbs[i][j] {
				for l := range probs.CoefProbs[i][j][k] {
					for m := range probs.CoefProbs[i][j][k][l] {
						for n := range probs.CoefProbs[i][j][k][l][m] {
							probs.CoefProbs[i][j][k][l][m][n] = adaptProbParam(
								probs.CoefProbs[i][j][k][l][m][n],
								counts.CoefProbs[i][j][k][l][m][n],
								coefCountSat, coefUpdateFactor,
							)
						}
					}
				}
			}
		}
	}

	if !full {
		return
	}

	// Skip probs.
	for i := range probs.SkipProbs {
		probs.SkipProbs[i] = adaptProb(probs.SkipProbs[i], counts.SkipProbs[i])
	}

	// Inter-mode probs.
	for i := range probs.InterModeProbs {
		for j := range probs.InterModeProbs[i] {
			probs.InterModeProbs[i][j] = adaptProb(probs.InterModeProbs[i][j], counts.InterModeProbs[i][j])
		}
	}

	// Interpolation filter probs.
	for i := range probs.InterpFilterProbs {
		for j := range probs.InterpFilterProbs[i] {
			probs.InterpFilterProbs[i][j] = adaptProb(probs.InterpFilterProbs[i][j], counts.InterpFilterProbs[i][j])
		}
	}

	// is_inter probs.
	for i := range probs.IsInterProbs {
		probs.IsInterProbs[i] = adaptProb(probs.IsInterProbs[i], counts.IsInterProbs[i])
	}

	// Compound inter probs.
	for i := range probs.CompInterProbs {
		probs.CompInterProbs[i] = adaptProb(probs.CompInterProbs[i], counts.CompInterProbs[i])
	}

	// Single ref probs.
	for i := range probs.SingleRefProbs {
		for j := range probs.SingleRefProbs[i] {
			probs.SingleRefProbs[i][j] = adaptProb(probs.SingleRefProbs[i][j], counts.SingleRefProbs[i][j])
		}
	}

	// Comp ref probs.
	for i := range probs.CompRefProbs {
		probs.CompRefProbs[i] = adaptProb(probs.CompRefProbs[i], counts.CompRefProbs[i])
	}

	// Y mode probs.
	for i := range probs.YModeProbs {
		for j := range probs.YModeProbs[i] {
			probs.YModeProbs[i][j] = adaptProb(probs.YModeProbs[i][j], counts.YModeProbs[i][j])
		}
	}

	// UV mode probs.
	for i := range probs.UVModeProbs {
		for j := range probs.UVModeProbs[i] {
			probs.UVModeProbs[i][j] = adaptProb(probs.UVModeProbs[i][j], counts.UVModeProbs[i][j])
		}
	}

	// Partition probs.
	for i := range probs.PartitionProbs {
		for j := range probs.PartitionProbs[i] {
			probs.PartitionProbs[i][j] = adaptProb(probs.PartitionProbs[i][j], counts.PartitionProbs[i][j])
		}
	}

	// MV probs.
	for i := range probs.MVJointProbs {
		probs.MVJointProbs[i] = adaptProb(probs.MVJointProbs[i], counts.MVJointProbs[i])
	}
	for i := range probs.MVSignProbs {
		probs.MVSignProbs[i] = adaptProb(probs.MVSignProbs[i], counts.MVSignProbs[i])
	}
	for i := range probs.MVClassProbs {
		for j := range probs.MVClassProbs[i] {
			probs.MVClassProbs[i][j] = adaptProb(probs.MVClassProbs[i][j], counts.MVClassProbs[i][j])
		}
	}
	for i := range probs.MVClass0BitProbs {
		probs.MVClass0BitProbs[i] = adaptProb(probs.MVClass0BitProbs[i], counts.MVClass0BitProbs[i])
	}
	for i := range probs.MVBitsProbs {
		for j := range probs.MVBitsProbs[i] {
			probs.MVBitsProbs[i][j] = adaptProb(probs.MVBitsProbs[i][j], counts.MVBitsProbs[i][j])
		}
	}
	for i := range probs.MVClass0FrProbs {
		for j := range probs.MVClass0FrProbs[i] {
			for k := range probs.MVClass0FrProbs[i][j] {
				probs.MVClass0FrProbs[i][j][k] = adaptProb(probs.MVClass0FrProbs[i][j][k], counts.MVClass0FrProbs[i][j][k])
			}
		}
	}
	for i := range probs.MVFrProbs {
		for j := range probs.MVFrProbs[i] {
			probs.MVFrProbs[i][j] = adaptProb(probs.MVFrProbs[i][j], counts.MVFrProbs[i][j])
		}
	}
	for i := range probs.MVClass0HPProbs {
		probs.MVClass0HPProbs[i] = adaptProb(probs.MVClass0HPProbs[i], counts.MVClass0HPProbs[i])
	}
	for i := range probs.MVHPProbs {
		probs.MVHPProbs[i] = adaptProb(probs.MVHPProbs[i], counts.MVHPProbs[i])
	}
}

// adaptProb implements VP9 mode/mv merge_probs (spec §8.9.2 / libvpx
// mode_mv_merge_probs): MODE_MV_COUNT_SAT=20, MODE_MV_MAX_UPDATE_FACTOR=128.
func adaptProb(prob uint8, count [2]uint32) uint8 {
	return adaptProbParam(prob, count, 20, 128)
}

// adaptProbParam is VP9 merge_probs with explicit count_sat / max_update_factor
// (coefficient probs use COEF_COUNT_SAT=24 and 112 or 128 — see AdaptProbs).
func adaptProbParam(prob uint8, count [2]uint32, countSat, maxUpdateFactor uint32) uint8 {
	total := count[0] + count[1]
	if total == 0 {
		return prob
	}
	sat := total
	if sat > countSat {
		sat = countSat
	}
	factor := maxUpdateFactor * sat / countSat
	// dominated_prob = clip(round(count[0]*256/total), 1, 255)
	dom := (uint64(count[0])*256 + uint64(total)>>1) / uint64(total)
	if dom < 1 {
		dom = 1
	}
	if dom > 255 {
		dom = 255
	}
	// new_prob = round_power_of_two(pre*(256-factor) + dom*factor, 8)
	result := (uint32(prob)*uint32(256-factor) + uint32(dom)*factor + 128) >> 8
	if result < 1 {
		return 1
	}
	if result > 255 {
		return 255
	}
	return uint8(result)
}
