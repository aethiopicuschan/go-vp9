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

// LevelForBlock computes the per-block loop filter level (libvpx
// vp9_loop_filter_frame_init + get_filter_level).
//
// segLFActive: SEG_LVL_ALT_LF is enabled for this block's segment (and
// segmentation is enabled on the frame).
// segLFData / segAbs: the segment's feature value, applied as a delta to
// baseLevel or, with segmentation_abs_or_delta_update set, as the level itself.
// deltaEnabled: frame-level mode_ref_delta_enabled flag.
// refFrame: 0=INTRA, 1=LAST, 2=GOLDEN, 3=ALTREF.
// zeroMVMode: the inter prediction mode is ZEROMV (mode_lf_lut index 0 →
// modeDeltas[0]); every other inter mode uses modeDeltas[1], whatever the MV
// value. Intra blocks get the INTRA ref delta only.
//
// The ref/mode deltas are scaled by 1 << (baseLevel >> 5) and applied whenever
// deltaEnabled is set — also when the segment level is 0.
func LevelForBlock(
	baseLevel int,
	segLFActive bool, segLFData int16, segAbs bool,
	deltaEnabled bool,
	refFrame int, isInter bool, zeroMVMode bool,
	refDeltas [4]int, modeDeltas [2]int,
) int {
	clamp := func(v int) int {
		if v < 0 {
			return 0
		}
		if v > 63 {
			return 63
		}
		return v
	}
	level := baseLevel
	if segLFActive {
		if segAbs {
			level = clamp(int(segLFData))
		} else {
			level = clamp(baseLevel + int(segLFData))
		}
	}
	if !deltaEnabled {
		return level
	}
	scale := 1 << uint(baseLevel>>5)
	if !isInter || refFrame <= 0 || refFrame >= 4 {
		return clamp(level + refDeltas[0]*scale)
	}
	mode := 1
	if zeroMVMode {
		mode = 0
	}
	return clamp(level + refDeltas[refFrame]*scale + modeDeltas[mode]*scale)
}
