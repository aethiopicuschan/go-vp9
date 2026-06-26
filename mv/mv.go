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
	"fmt"
	"io"

	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
)

// DbgW: when non-nil, ReadMV traces joint + per-step range.
var DbgW io.Writer

// MV is a motion vector: Row is the vertical component, Col is horizontal.
// Values are in VP9-native units where 8 units = 1 pixel.
type MV struct {
	Row, Col int16
}

// Joint type constants (spec Table 8-9).
const (
	jointZero  = 0 // MV_JOINT_ZERO  — both components zero
	jointHNZVZ = 1 // MV_JOINT_HNZVZ — vertical (row) zero, horizontal (col) non-zero
	jointHZVNZ = 2 // MV_JOINT_HZVNZ — horizontal (col) zero, vertical (row) non-zero
	jointHNVNZ = 3 // MV_JOINT_HNZVNZ — both non-zero
)

// ReadMV decodes one motion vector from the boolean bitstream.
// probs must be a non-nil Probs updated for the current frame.
// counts is optional (nil disables count accumulation for probability adaptation).
// allowHP enables 1/8-pel (high-precision) motion vectors.
func ReadMV(br *boolcoder.Reader, probs *compheader.Probs, counts *compheader.Counts, allowHP bool) MV {
	joint := readJoint(br, probs, counts)
	if DbgW != nil {
		rng, top := br.State()
		fmt.Fprintf(DbgW, "    MV joint=%d afterJoint rng=%d t=%d bp=%d\n", joint, rng, top, br.BitPos())
	}
	var mv MV
	// Vertical (row) component is coded when the joint is HZVNZ or HNZVNZ;
	// horizontal (col) when HNZVZ or HNZVNZ (libvpx mv_joint_vertical/horizontal).
	// Row is read before col.
	if joint == jointHZVNZ || joint == jointHNVNZ {
		mv.Row = int16(readComponent(br, probs, counts, 0, allowHP))
	}
	if joint == jointHNZVZ || joint == jointHNVNZ {
		mv.Col = int16(readComponent(br, probs, counts, 1, allowHP))
	}
	if DbgW != nil {
		rng, top := br.State()
		fmt.Fprintf(DbgW, "    MV result=(%d,%d) afterMV rng=%d t=%d allowHP=%v\n", mv.Row, mv.Col, rng, top, allowHP)
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
	// vp9_mv_class_tree: prob[6]=0 → CLASS_6 (leaf); else prob[7] splits
	// {7,8} (via prob[8]) from {9,10} (via prob[9]).
	if readBoolCount(p.MVClassProbs[c][6], 6) == 0 {
		return 6
	}
	if readBoolCount(p.MVClassProbs[c][7], 7) == 0 {
		if readBoolCount(p.MVClassProbs[c][8], 8) == 0 {
			return 7
		}
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
	// vp9_mv_fp_tree {-0, 2, -1, 4, -2, -3}: prob[0]=0 → 0 (leaf);
	// prob[0]=1,prob[1]=0 → 1; prob[0]=1,prob[1]=1 → 2|prob[2].
	b0 := br.ReadBool(probs[0])
	if frCounts != nil {
		frCounts[0][b0]++
	}
	if b0 == 0 {
		return 0
	}
	b1 := br.ReadBool(probs[1])
	if frCounts != nil {
		frCounts[1][b1]++
	}
	if b1 == 0 {
		return 1
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
	if DbgW != nil {
		fmt.Fprintf(DbgW, "      comp c=%d sign=%d signProb=%d afterSign bp=%d\n", c, signBit, p.MVSignProbs[c], br.BitPos())
	}
	cls := readClass(br, p, counts, c)
	if DbgW != nil {
		fmt.Fprintf(DbgW, "      comp cls=%d classProb0=%d afterClass bp=%d\n", cls, p.MVClassProbs[c][0], br.BitPos())
	}

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

	// Magnitude in VP9-native units (8 units = 1 pixel). libvpx read_mv_component:
	//   mag = vp9_get_mv_mag(cls, (d<<3)|(fr<<1)|hp) + 1
	// where mv_class_base(cls) = cls ? CLASS0_SIZE<<(cls+2) : 0  (CLASS0_SIZE=2).
	base := 0
	if cls > 0 {
		base = 2 << (uint(cls) + 2)
	}
	mag := base + ((d << 3) | (fr << 1) | hp) + 1

	if signBit != 0 {
		return -mag
	}
	return mag
}

// UseHP reports whether high-precision (1/8-pel) MV bits are coded for a
// component, given the reference (best) MV (libvpx use_mv_hp): true only when
// the reference MV is within ±64 (1/8-pel) on both axes.
func UseHP(ref MV) bool {
	r, c := ref.Row, ref.Col
	if r < 0 {
		r = -r
	}
	if c < 0 {
		c = -c
	}
	return r < 64 && c < 64
}
