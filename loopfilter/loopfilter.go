// Package loopfilter implements the VP9 deblocking loop filter (spec §8.8).
// All filter kernels operate on byte slices addressed via a step parameter,
// which allows the same code to handle both horizontal and vertical edges:
//   - horizontal edge: step = 1  (consecutive bytes in a row)
//   - vertical edge:   step = stride (bytes to the next row)
//
// The four filter widths:
//   filter4  – 4-tap narrow (inner 4×4 sub-block edges)
//   filter8  – 8-tap wide  (TX-block edges, luma/chroma)
//   filter16 – 14-tap      (super-block / large TX edges)
package loopfilter

// ---- helpers ----------------------------------------------------------------

func abs8(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func clampByte(v int) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}

// sclamp clamps v to [-128, 127].
func sclamp(v int) int {
	if v < -128 {
		return -128
	}
	if v > 127 {
		return 127
	}
	return v
}

// ---- mask predicates --------------------------------------------------------

// needsFilter returns true when the 8-sample neighbourhood at step-offset p
// satisfies the main filter mask (spec §8.8.4). p[0] is the first "inner"
// sample to the left/above of the edge; negative offsets address farther-left.
// Caller passes p as a slice starting at q0 (right/below-edge sample):
//
//	buf[edgeIdx - 4*step] = p3, ..., buf[edgeIdx] = q0, buf[edgeIdx+step] = q1, ...
//
// To avoid allocations the function takes raw indices into buf.
func needsFilter(buf []byte, q0 int, step int, blim, lim uint8) bool {
	p3 := int(buf[q0-4*step])
	p2 := int(buf[q0-3*step])
	p1 := int(buf[q0-2*step])
	p0 := int(buf[q0-1*step])
	q0v := int(buf[q0])
	q1 := int(buf[q0+1*step])
	q2 := int(buf[q0+2*step])
	q3 := int(buf[q0+3*step])

	blimI := int(blim)
	limI := int(lim)

	return !(abs8(p3-p2) > limI || abs8(p2-p1) > limI || abs8(p1-p0) > limI ||
		abs8(q0v-q1) > limI || abs8(q1-q2) > limI || abs8(q2-q3) > limI ||
		abs8(p0-q0v)*2+abs8(p1-q1)/2 > blimI)
}

// isFlat2 returns true for the extra-wide 14-tap filter condition
// (6 samples on each side are all within ±1 of p0 and q0).
func isFlat2(buf []byte, q0, step int) bool {
	p0 := int(buf[q0-1*step])
	q0v := int(buf[q0])
	return abs8(int(buf[q0-4*step])-p0) <= 1 &&
		abs8(int(buf[q0-5*step])-p0) <= 1 &&
		abs8(int(buf[q0-6*step])-p0) <= 1 &&
		abs8(int(buf[q0+3*step])-q0v) <= 1 &&
		abs8(int(buf[q0+4*step])-q0v) <= 1 &&
		abs8(int(buf[q0+5*step])-q0v) <= 1
}

// ---- 4-tap narrow filter (spec §8.8.5) -------------------------------------

// applyFilter4 modifies p1,p0,q0,q1 in-place (signed arithmetic, XOR 0x80).
func applyFilter4(buf []byte, q0, step int, highEdge bool) {
	i1 := q0 - 2*step
	i0 := q0 - 1*step
	j0 := q0
	j1 := q0 + 1*step

	ps1 := int(int8(buf[i1] ^ 0x80))
	ps0 := int(int8(buf[i0] ^ 0x80))
	qs0 := int(int8(buf[j0] ^ 0x80))
	qs1 := int(int8(buf[j1] ^ 0x80))

	// Signed clamp to [-128,127]: sclamp
	var f int
	if highEdge {
		f = sclamp(ps1 - qs1)
	}
	f = sclamp(f + 3*(qs0-ps0))

	f1 := sclamp(f+4) >> 3
	f2 := sclamp(f+3) >> 3

	buf[j0] = byte(sclamp(qs0-f1) ^ 0x80)
	buf[i0] = byte(sclamp(ps0+f2) ^ 0x80)

	if !highEdge {
		fh := (sclamp(f1+1) >> 1)
		buf[j1] = byte(sclamp(qs1-fh) ^ 0x80)
		buf[i1] = byte(sclamp(ps1+fh) ^ 0x80)
	}
}

// ---- 8-tap wide filter (spec §8.8.6) ----------------------------------------

// round3 = (v + 4) >> 3
func round3(v int) int { return (v + 4) >> 3 }

func applyFilter8(buf []byte, q0, step int) {
	p3 := int(buf[q0-4*step])
	p2 := int(buf[q0-3*step])
	p1 := int(buf[q0-2*step])
	p0 := int(buf[q0-1*step])
	q0v := int(buf[q0])
	q1 := int(buf[q0+1*step])
	q2 := int(buf[q0+2*step])
	q3 := int(buf[q0+3*step])

	buf[q0-3*step] = clampByte(round3(3*p3 + 2*p2 + p1 + p0 + q0v))
	buf[q0-2*step] = clampByte(round3(2*p3 + p2 + 2*p1 + p0 + q0v + q1))
	buf[q0-1*step] = clampByte(round3(p3 + p2 + p1 + 2*p0 + q0v + q1 + q2))
	buf[q0] = clampByte(round3(p2 + p1 + p0 + 2*q0v + q1 + q2 + q3))
	buf[q0+1*step] = clampByte(round3(p1 + p0 + q0v + 2*q1 + q2 + q3 + q3))
	buf[q0+2*step] = clampByte(round3(p0 + q0v + q1 + 2*q2 + q3 + q3 + q3))
}

// applyFilter8Vals applies the 8-tap wide filter with p3..q3 already loaded,
// avoiding the redundant 8-sample re-read that applyFilter8 would perform.
func applyFilter8Vals(buf []byte, q0, step, p3, p2, p1, p0, q0v, q1, q2, q3 int) {
	buf[q0-3*step] = clampByte(round3(3*p3 + 2*p2 + p1 + p0 + q0v))
	buf[q0-2*step] = clampByte(round3(2*p3 + p2 + 2*p1 + p0 + q0v + q1))
	buf[q0-1*step] = clampByte(round3(p3 + p2 + p1 + 2*p0 + q0v + q1 + q2))
	buf[q0] = clampByte(round3(p2 + p1 + p0 + 2*q0v + q1 + q2 + q3))
	buf[q0+1*step] = clampByte(round3(p1 + p0 + q0v + 2*q1 + q2 + q3 + q3))
	buf[q0+2*step] = clampByte(round3(p0 + q0v + q1 + 2*q2 + q3 + q3 + q3))
}

// applyFilter16Vals applies the 14-tap filter with p3..q3 pre-loaded.
// Loads only the 6 extra values (p4..p6, q4..q6) from buf.
func applyFilter16Vals(buf []byte, q0, step, p3, p2, p1, p0, q0v, q1, q2, q3 int) {
	p6 := int(buf[q0-7*step])
	p5 := int(buf[q0-6*step])
	p4 := int(buf[q0-5*step])
	q4 := int(buf[q0+4*step])
	q5 := int(buf[q0+5*step])
	q6 := int(buf[q0+6*step])

	buf[q0-6*step] = clampByte(round3(p6*7 + p5*2 + p4 + p3 + p2 + p1 + p0 + q0v))
	buf[q0-5*step] = clampByte(round3(p6*5 + p5*2 + p4*2 + p3 + p2 + p1 + p0 + q0v + q1))
	buf[q0-4*step] = clampByte(round3(p6*4 + p5 + p4*2 + p3*2 + p2 + p1 + p0 + q0v + q1 + q2))
	buf[q0-3*step] = clampByte(round3(p6*3 + p5 + p4 + p3*2 + p2*2 + p1 + p0 + q0v + q1 + q2 + q3))
	buf[q0-2*step] = clampByte(round3(p6*2 + p5 + p4 + p3 + p2*2 + p1*2 + p0 + q0v + q1 + q2 + q3 + q4))
	buf[q0-1*step] = clampByte(round3(p6 + p5 + p4 + p3 + p2 + p1*2 + p0*2 + q0v + q1 + q2 + q3 + q4 + q5))
	buf[q0] = clampByte(round3(p5 + p4 + p3 + p2 + p1 + p0*2 + q0v*2 + q1 + q2 + q3 + q4 + q5 + q6))
	buf[q0+1*step] = clampByte(round3(p4 + p3 + p2 + p1 + p0 + q0v*2 + q1*2 + q2 + q3 + q4 + q5 + q6 + q6))
	buf[q0+2*step] = clampByte(round3(p3 + p2 + p1 + p0 + q0v + q1*2 + q2*2 + q3 + q4 + q5 + q6*2))
	buf[q0+3*step] = clampByte(round3(p2 + p1 + p0 + q0v + q1 + q2*2 + q3*2 + q4 + q5 + q6*3))
	buf[q0+4*step] = clampByte(round3(p1 + p0 + q0v + q1 + q2 + q3*2 + q4*2 + q5 + q6*4))
	buf[q0+5*step] = clampByte(round3(p0 + q0v + q1 + q2 + q3 + q4*2 + q5*2 + q6*5))
}

// ---- Public edge filter entry points ----------------------------------------

// FilterEdge4 filters a single 4-sample edge in buf at position q0 (index of
// the first sample on the "right/below" side of the edge). step is the address
// step between consecutive samples (1 for horizontal, stride for vertical).
// Requires buf[q0-4*step..q0+3*step] to be valid.
// width (flat filter) and wide (flat2 filter) select the kernel:
//
//	width=4:  4-tap narrow only
//	width=8:  8-tap if flat, else 4-tap narrow
//	width=16: 14-tap if flat2, 8-tap if flat, else 4-tap narrow
func FilterEdge4(buf []byte, q0, step, width int, p Params) {
	// Load the 8 core samples once — shared by needsFilter, hev, isFlat, and apply8.
	p3 := int(buf[q0-4*step])
	p2 := int(buf[q0-3*step])
	p1 := int(buf[q0-2*step])
	p0 := int(buf[q0-1*step])
	q0v := int(buf[q0])
	q1 := int(buf[q0+1*step])
	q2 := int(buf[q0+2*step])
	q3 := int(buf[q0+3*step])

	// Inline needsFilter — most calls exit here.
	blimI := int(p.BLimit)
	limI := int(p.Limit)
	if abs8(p3-p2) > limI || abs8(p2-p1) > limI || abs8(p1-p0) > limI ||
		abs8(q0v-q1) > limI || abs8(q1-q2) > limI || abs8(q2-q3) > limI ||
		abs8(p0-q0v)*2+abs8(p1-q1)/2 > blimI {
		return
	}

	// Inline hev — uses already-loaded values.
	t := int(p.Thresh)
	highEdge := abs8(p1-p0) > t || abs8(q0v-q1) > t

	// Inline isFlat — all 7 inner deltas ≤ 1 relative to p0.
	flat := abs8(p3-p0) <= 1 && abs8(p2-p0) <= 1 && abs8(p1-p0) <= 1 &&
		abs8(q0v-p0) <= 1 && abs8(q1-p0) <= 1 && abs8(q2-p0) <= 1 && abs8(q3-p0) <= 1

	switch width {
	case 16:
		if flat && isFlat2(buf, q0, step) {
			applyFilter16Vals(buf, q0, step, p3, p2, p1, p0, q0v, q1, q2, q3)
			return
		}
		fallthrough
	case 8:
		if flat {
			applyFilter8Vals(buf, q0, step, p3, p2, p1, p0, q0v, q1, q2, q3)
			return
		}
		fallthrough
	default:
		applyFilter4(buf, q0, step, highEdge)
	}
}
