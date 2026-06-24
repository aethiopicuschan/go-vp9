package compheader

// DefaultProbs returns Probs initialised to the VP9 spec Appendix B default
// values. These are the starting probabilities for the first frame in a
// sequence (or after a context reset).
//
// Coefficient probs are set to 128 (placeholder) — the spec Appendix B tables
// are large and the values are not consumed until M5 (coefficient token decode).
// All other tables match the normative defaults.
func DefaultProbs() *Probs {
	p := &Probs{}

	// TX mode probs (spec Appendix B.1, default_tx_probs [2][3]).
	// Layout: each row is [p(choose 4x4), p(choose 8x8), p(choose 16x16)]
	// for a given TX size context.
	p.TxProbs8 = [2]uint8{3, 5}
	p.TxProbs16 = [2][2]uint8{{3, 136}, {5, 52}}
	p.TxProbs32 = [2][3]uint8{{3, 136, 37}, {5, 52, 13}}

	// Coefficient probabilities (spec Appendix B, default_coef_probs).
	// Layout: [txSize][planeType][refType][band][ctx][3 nodes].
	// Sourced from libvpx vp9_default_coef_probs (BSD-3).
	p.CoefProbs = defaultCoefProbs

	// Skip probs (spec Appendix B, default_skip_probs).
	p.SkipProbs = [3]uint8{192, 128, 64}

	// Inter-mode probs [7][3] (default_inter_mode_probs).
	p.InterModeProbs = [7][3]uint8{
		{2, 173, 34},
		{7, 145, 85},
		{7, 166, 63},
		{7, 94, 66},
		{8, 64, 46},
		{17, 81, 31},
		{25, 29, 30},
	}

	// Switchable interpolation filter probs [4][2] (default_switchable_interp_prob).
	p.InterpFilterProbs = [4][2]uint8{
		{235, 162},
		{36, 255},
		{34, 3},
		{149, 144},
	}

	// Is-inter probs [4] (default_intra_inter_prob).
	p.IsInterProbs = [4]uint8{9, 102, 187, 225}

	// Compound-inter probs [5] (default_comp_inter_prob).
	p.CompInterProbs = [5]uint8{239, 189, 93, 100, 178}

	// Single-ref probs [5][2] (default_single_ref_prob).
	p.SingleRefProbs = [5][2]uint8{
		{33, 16},
		{77, 74},
		{142, 142},
		{172, 170},
		{238, 247},
	}

	// Compound-ref probs [5] (default_comp_ref_prob).
	p.CompRefProbs = [5]uint8{50, 126, 123, 221, 226}

	// Y intra mode probs [4][9] (default_if_y_probs).
	p.YModeProbs = [4][9]uint8{
		{65, 32, 18, 144, 162, 194, 41, 51, 98},
		{132, 68, 18, 165, 217, 196, 45, 40, 112},
		{103, 26, 11, 165, 213, 206, 39, 77, 130},
		{132, 126, 18, 188, 175, 194, 50, 50, 163},
	}

	// UV mode probs [10][9] (default_uv_probs, indexed by Y intra mode).
	p.UVModeProbs = [10][9]uint8{
		{120, 7, 76, 176, 208, 126, 28, 54, 103},  // DC
		{48, 12, 154, 155, 139, 90, 34, 117, 119}, // V
		{67, 6, 25, 204, 243, 158, 13, 21, 96},    // H
		{97, 5, 44, 131, 176, 139, 48, 68, 97},    // D45
		{83, 5, 42, 156, 111, 152, 26, 49, 152},   // D135
		{180, 7, 149, 143, 151, 159, 91, 47, 124}, // D117
		{91, 17, 34, 196, 140, 195, 12, 15, 70},   // D153
		{26, 12, 131, 201, 230, 223, 26, 16, 57},  // D207
		{57, 12, 85, 236, 202, 198, 130, 26, 102}, // D63
		{91, 6, 18, 228, 252, 196, 12, 10, 55},    // TM
	}

	// Partition probs [16][3] (default_partition_probs).
	// 4 block-size levels × 4 above/left combinations.
	p.PartitionProbs = [16][3]uint8{
		// 8x8 → 4x4
		{199, 122, 141},
		{147, 63, 159},
		{148, 133, 118},
		{2, 32, 68},
		// 16x16 → 8x8
		{174, 146, 170},
		{90, 74, 141},
		{127, 119, 132},
		{28, 46, 107},
		// 32x32 → 16x16
		{132, 110, 168},
		{74, 71, 134},
		{110, 105, 127},
		{12, 24, 72},
		// 64x64 → 32x32
		{224, 144, 178},
		{107, 69, 132},
		{148, 108, 122},
		{42, 44, 72},
	}

	// MV joint probs [3] (default_nmv_context.joints).
	p.MVJointProbs = [3]uint8{32, 64, 96}

	// MV sign probs [2].
	p.MVSignProbs = [2]uint8{128, 128}

	// MV class probs [2][10].
	p.MVClassProbs = [2][10]uint8{
		{224, 144, 192, 168, 192, 176, 192, 198, 198, 245},
		{216, 128, 176, 160, 176, 176, 192, 198, 198, 245},
	}

	// MV class0 bit probs [2].
	p.MVClass0BitProbs = [2]uint8{216, 216}

	// MV bits probs [2][10].
	p.MVBitsProbs = [2][10]uint8{
		{136, 140, 148, 160, 176, 192, 224, 234, 234, 240},
		{136, 140, 148, 160, 176, 192, 224, 234, 234, 240},
	}

	// MV class0 fractional probs [2][2][3].
	p.MVClass0FrProbs = [2][2][3]uint8{
		{{128, 128, 64}, {96, 112, 64}},
		{{128, 128, 64}, {96, 112, 64}},
	}

	// MV fractional probs [2][3].
	p.MVFrProbs = [2][3]uint8{
		{64, 96, 64},
		{64, 96, 64},
	}

	// MV high-precision probs [2].
	p.MVClass0HPProbs = [2]uint8{160, 160}
	p.MVHPProbs = [2]uint8{128, 128}

	return p
}
