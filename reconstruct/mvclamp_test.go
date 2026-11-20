package reconstruct

import "testing"

// clampMVQ4 against values worked out by hand from libvpx
// clamp_mv_to_umv_border_sb (vp9_reconinter.h).
func TestClampMVQ4(t *testing.T) {
	cases := []struct {
		name                           string
		mv, pos, blk, planeBW, mi4, ss int
		want                           int
	}{
		// 3840 wide, 32x32 block in the last column: mb_to_right_edge = 0,
		// spel_right = ((4+32)<<4) - 16 = 560.
		{"luma right clamp", 300, 3808, 32, 32, 960, 0, 560},
		{"luma right at limit", 280, 3808, 32, 32, 960, 0, 560},
		{"luma right inside", 100, 3808, 32, 32, 960, 0, 200},
		// First column: mb_to_left_edge = 0, spel_left = (4+32)<<4 = 576.
		{"luma left clamp", -400, 0, 32, 32, 960, 0, -576},
		{"luma left inside", -287, 0, 32, 32, 960, 0, -574},
		// Chroma of the same blocks: 16 px plane block, MV not doubled.
		{"chroma right clamp", 310, 3808, 32, 16, 960, 1, 304},
		{"chroma left clamp", -330, 0, 32, 16, 960, 1, -320},
		{"chroma inside", 37, 64, 32, 16, 960, 1, 37},
		// Interior block: far from both limits.
		{"interior", -1000, 1920, 64, 64, 960, 0, -2000},
		// Width 100 (25 4x4 units): libvpx rounds up to 13 MI = 104 px.
		{"odd width", 100, 96, 8, 8, 25, 0, 176},
		// Frame size unknown: conversion only.
		{"no frame size", 5, 0, 8, 8, 0, 0, 10},
	}
	for _, c := range cases {
		if got := clampMVQ4(c.mv, c.pos, c.blk, c.planeBW, c.mi4, c.ss); got != c.want {
			t.Errorf("%s: got %d, want %d", c.name, got, c.want)
		}
	}
}
