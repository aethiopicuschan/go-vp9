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
	lim := int(limLUT[level])
	if sharpness > 0 {
		// Reduce limit for higher sharpness (§8.8.3).
		if sharpness > 4 {
			lim >>= 2
		} else {
			lim >>= 1
		}
		max := 9 - sharpness
		if lim > max {
			lim = max
		}
	}
	if lim < 1 {
		lim = 1
	}
	return Params{
		Limit:  uint8(lim),
		BLimit: uint8(2*level + lim),
		Thresh: threshLUT[level],
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
		level += refDeltas[r]
		if isInter {
			if isZeroMV {
				level += modeDeltas[0]
			} else {
				level += modeDeltas[1]
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

// limLUT[level] — inner flatness limit per filter level (libvpx loop_filter_info_n).
var limLUT = [64]uint8{
	0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
	3, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6,
	8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 12, 12, 14, 16,
}

// threshLUT[level] — HEV threshold per filter level.
var threshLUT = [64]uint8{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
}
