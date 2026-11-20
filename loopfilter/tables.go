package loopfilter

// Params holds the per-edge filter thresholds derived from a filter level.
type Params struct {
	Limit  uint8 // interior flatness threshold (lim_lut)
	BLimit uint8 // block-edge threshold (2*limit + level)
	Thresh uint8 // HEV threshold (thresh_lut)
}

// ParamsForLevel returns filter thresholds for level ∈ [0,63]
// and sharpness ∈ [0,7] (spec §8.8.3).
func ParamsForLevel(level, sharpness int) Params {
	if level < 0 {
		level = 0
	}
	if level > 63 {
		level = 63
	}
	if sharpness < 0 {
		sharpness = 0
	}
	if sharpness > 7 {
		sharpness = 7
	}
	// libvpx update_sharpness: limit = level >> ((sharp>0)+(sharp>4)), capped at
	// 9-sharpness and ≥1; blimit = 2*(level+2)+limit; thresh = level>>4.
	shift := 0
	if sharpness > 0 {
		shift++
	}
	if sharpness > 4 {
		shift++
	}
	lim := level >> shift
	if sharpness > 0 && lim > 9-sharpness {
		lim = 9 - sharpness
	}
	if lim < 1 {
		lim = 1
	}
	return Params{
		Limit:  uint8(lim),
		BLimit: uint8(2*(level+2) + lim),
		Thresh: uint8(level >> 4),
	}
}

// LevelForBlock computes the per-block loop filter level (spec §8.8.1 / libvpx vp9_loop_filter.c).
//
// segLFActive: SEG_LVL_ALT_LF is enabled for this block's segment.
// segLFDelta: the segment's LF delta (always additive, never absolute).
// deltaEnabled: frame-level mode_ref_delta_enabled flag.
// refFrame: 0=INTRA, 1=LAST, 2=GOLDEN, 3=ALTREF.
// isZeroMV: inter block with zero MV (uses modeDeltas[0], else modeDeltas[1]).
func LevelForBlock(
	baseLevel int,
	segLFActive bool, segLFDelta int16,
	deltaEnabled bool,
	refFrame int, isInter bool, isZeroMV bool,
	refDeltas [4]int, modeDeltas [2]int,
) int {
	level := baseLevel
	if segLFActive {
		level += int(segLFDelta)
		if level < 0 {
			level = 0
		} else if level > 63 {
			level = 63
		}
	}
	if deltaEnabled && level > 0 {
		r := refFrame
		if r < 0 || r >= 4 {
			r = 0
		}
		scale := 1 << uint(baseLevel>>5)
		level += refDeltas[r] * scale
		if isInter {
			if isZeroMV {
				level += modeDeltas[0] * scale
			} else {
				level += modeDeltas[1] * scale
			}
		}
		if level < 0 {
			level = 0
		} else if level > 63 {
			level = 63
		}
	}
	return level
}
