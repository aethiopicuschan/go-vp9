// Package mv implements VP9 motion vector (MV) decoding (spec §8.3.3).
//
// MV components are returned in VP9-native units: 8 units = 1 pixel,
// i.e. the integer pixel offset is component>>3 and the sub-pixel
// offset (for 8-tap filter selection) is (component&7)<<1 (1/16-pel).
//
// Zero displacement is encoded via the MV_JOINT_ZERO joint type, not via
// this function — ReadMV returns MV{0,0} only when the joint is ZERO.
//
// Probability tables are taken directly from the caller's *compheader.Probs
// so that probability updates (diff_update_prob) are automatically reflected.
package mv

import (
	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
)

// MV is a motion vector: Row is the vertical component, Col is horizontal.
// Values are in VP9-native units where 8 units = 1 pixel.
type MV struct {
	Row, Col int16
}

// Joint type constants (spec Table 8-9).
const (
	jointZero  = 0 // MV_JOINT_ZERO — both components zero
	jointHNZVZ = 1 // MV_JOINT_HNZVZ — row non-zero, col zero
	jointHZVNZ = 2 // MV_JOINT_HZVNZ — row zero, col non-zero
	jointHNVNZ = 3 // MV_JOINT_HNVNZ — both non-zero
)

// ReadMV decodes one motion vector from the boolean bitstream.
// probs must be a non-nil Probs updated for the current frame.
// counts is optional (nil disables count accumulation for probability adaptation).
// allowHP enables 1/8-pel (high-precision) motion vectors.
func ReadMV(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts, allowHP bool) MV {
	joint := readJoint(br, probs, counts)
	var mv MV
	if joint == jointHNZVZ || joint == jointHNVNZ {
		mv.Row = int16(readComponent(br, probs, counts, 0, allowHP))
	}
	if joint == jointHZVNZ || joint == jointHNVNZ {
		mv.Col = int16(readComponent(br, probs, counts, 1, allowHP))
	}
	return mv
}

// readJoint decodes the MV joint type (4-way tree, 3 probs).
func readJoint(br *boolcoder.Reader, p *compheader.Probs, counts *compheader.Counts) int {
	b0 := br.ReadBool(p.MVJointProbs[0])
	if counts != nil {
		counts.MVJointProbs[0][b0]++
	}
	if b0 == 0 {
		return jointZero
	}
	b1 := br.ReadBool(p.MVJointProbs[1])
	if counts != nil {
		counts.MVJointProbs[1][b1]++
	}
	if b1 == 0 {
		return jointHNZVZ
	}
	b2 := br.ReadBool(p.MVJointProbs[2])
	if counts != nil {
		counts.MVJointProbs[2][b2]++
	}
	if b2 == 0 {
		return jointHZVNZ
	}
	return jointHNVNZ
}

// readClass decodes the MV class (CLASS_0..CLASS_10) via an 11-way tree (10 probs).
func readClass(br *boolcoder.Reader, p *compheader.Probs, counts *compheader.Counts, c int) int {
	readBoolCount := func(prob uint8, idx int) int {
		bit := br.ReadBool(prob)
		if counts != nil {
			counts.MVClassProbs[c][idx][bit]++
		}
		return bit
	}
	if readBoolCount(p.MVClassProbs[c][0], 0) == 0 {
		return 0
	}
	if readBoolCount(p.MVClassProbs[c][1], 1) == 0 {
		return 1
	}
	if readBoolCount(p.MVClassProbs[c][2], 2) == 0 {
		if readBoolCount(p.MVClassProbs[c][3], 3) == 0 {
			return 2
		}
		return 3
	}
	if readBoolCount(p.MVClassProbs[c][4], 4) == 0 {
		if readBoolCount(p.MVClassProbs[c][5], 5) == 0 {
			return 4
		}
		return 5
	}
	if readBoolCount(p.MVClassProbs[c][6], 6) == 0 {
		if readBoolCount(p.MVClassProbs[c][7], 7) == 0 {
			return 6
		}
		return 7
	}
	if readBoolCount(p.MVClassProbs[c][8], 8) == 0 {
		return 8
	}
	if readBoolCount(p.MVClassProbs[c][9], 9) == 0 {
		return 9
	}
	return 10
}

// readFP decodes the 2-bit fractional-pixel component (0..3) via a 3-node tree.
// frCounts is optional slice of 3 [2]uint32 entries for bit counting.
func readFP(br *boolcoder.Reader, probs [3]uint8, frCounts [][2]uint32) int {
	b0 := br.ReadBool(probs[0])
	if frCounts != nil {
		frCounts[0][b0]++
	}
	if b0 == 0 {
		b1 := br.ReadBool(probs[1])
		if frCounts != nil {
			frCounts[1][b1]++
		}
		return b1
	}
	b2 := br.ReadBool(probs[2])
	if frCounts != nil {
		frCounts[2][b2]++
	}
	return 2 | b2
}

// readComponent decodes one signed MV component (row or col).
// c is 0 for row, 1 for col.
func readComponent(br *boolcoder.Reader, p *compheader.Probs, counts *compheader.Counts, c int, allowHP bool) int {
	signBit := br.ReadBool(p.MVSignProbs[c])
	if counts != nil {
		counts.MVSignProbs[c][signBit]++
	}
	cls := readClass(br, p, counts, c)

	var d, fr, hp int
	if cls == 0 {
		// CLASS_0: read 1 integer bit, then fractional bits from class0 sub-context.
		d = br.ReadBool(p.MVClass0BitProbs[c])
		if counts != nil {
			counts.MVClass0BitProbs[c][d]++
		}
		var frCounts [][2]uint32
		if counts != nil {
			frCounts = counts.MVClass0FrProbs[c][d][:]
		}
		fr = readFP(br, p.MVClass0FrProbs[c][d], frCounts)
		if allowHP {
			hp = br.ReadBool(p.MVClass0HPProbs[c])
			if counts != nil {
				counts.MVClass0HPProbs[c][hp]++
			}
		} else {
			hp = 1
		}
	} else {
		// CLASS_k (k≥1): read k integer bits; the class base contributes the high bit.
		for i := 0; i < cls; i++ {
			bit := br.ReadBool(p.MVBitsProbs[c][i])
			if counts != nil {
				counts.MVBitsProbs[c][i][bit]++
			}
			d |= bit << i
		}
		var frCounts [][2]uint32
		if counts != nil {
			frCounts = counts.MVFrProbs[c][:]
		}
		fr = readFP(br, p.MVFrProbs[c], frCounts)
		if allowHP {
			hp = br.ReadBool(p.MVHPProbs[c])
			if counts != nil {
				counts.MVHPProbs[c][hp]++
			}
		} else {
			hp = 1
		}
	}

	// Magnitude in VP9-native units (8 units = 1 pixel).
	// Formula from libvpx: mag = CLASS0_SIZE<<(cls+2) + (d<<3|fr<<1|hp).
	// CLASS0_SIZE=2, so base = 2<<(cls+2) = 2^(cls+3).
	base := 2 << (uint(cls) + 2)
	mag := base + (d<<3)|(fr<<1)|hp

	if signBit != 0 {
		return -mag
	}
	return mag
}
