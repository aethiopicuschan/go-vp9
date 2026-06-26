package loopfilter

// High-bit-depth (10/12-bit) deblocking — a parallel uint16 port of the kernels
// in loopfilter.go. The bit-depth differences vs the 8-bit path (libvpx
// vpx_highbd_lpf_*): the narrow filter's signed transform subtracts the midpoint
// 1<<(bd-1) instead of XOR 0x80; the signed clamp widens to [-128<<s, 128<<s-1]
// (s = bd-8); the mask limit/blimit/thresh and the flatness threshold all scale
// by <<s; and the flat-filter results clamp to [0, (1<<bd)-1]. For bd=8 every
// parameter collapses to the 8-bit values, so this reproduces the 8-bit kernels.
// Set the parameters with SetBitDepth16 before filtering.

var (
	maxVal16    = 255
	shift16     = 0
	mid16       = 128
	flatThresh16 = 1
)

// SetBitDepth16 configures the highbd loop-filter parameters for bit depth bd.
func SetBitDepth16(bd int) {
	maxVal16 = (1 << uint(bd)) - 1
	shift16 = bd - 8
	mid16 = 1 << uint(bd-1)
	flatThresh16 = 1 << uint(bd-8)
}

func clampPix16(v int) uint16 {
	if v < 0 {
		return 0
	}
	if v > maxVal16 {
		return uint16(maxVal16)
	}
	return uint16(v)
}

// sclampHigh clamps v to [-128<<s, (128<<s)-1] = [-mid16, mid16-1].
func sclampHigh(v int) int {
	if v < -mid16 {
		return -mid16
	}
	if v > mid16-1 {
		return mid16 - 1
	}
	return v
}

func isFlat216(buf []uint16, q0, step int) bool {
	p0 := int(buf[q0-1*step])
	q0v := int(buf[q0])
	return abs8(int(buf[q0-5*step])-p0) <= flatThresh16 &&
		abs8(int(buf[q0-6*step])-p0) <= flatThresh16 &&
		abs8(int(buf[q0-7*step])-p0) <= flatThresh16 &&
		abs8(int(buf[q0-8*step])-p0) <= flatThresh16 &&
		abs8(int(buf[q0+4*step])-q0v) <= flatThresh16 &&
		abs8(int(buf[q0+5*step])-q0v) <= flatThresh16 &&
		abs8(int(buf[q0+6*step])-q0v) <= flatThresh16 &&
		abs8(int(buf[q0+7*step])-q0v) <= flatThresh16
}

func applyFilter4_16(buf []uint16, q0, step int, highEdge bool) {
	i1 := q0 - 2*step
	i0 := q0 - 1*step
	j0 := q0
	j1 := q0 + 1*step

	ps1 := int(buf[i1]) - mid16
	ps0 := int(buf[i0]) - mid16
	qs0 := int(buf[j0]) - mid16
	qs1 := int(buf[j1]) - mid16

	var f int
	if highEdge {
		f = sclampHigh(ps1 - qs1)
	}
	f = sclampHigh(f + 3*(qs0-ps0))

	f1 := sclampHigh(f+4) >> 3
	f2 := sclampHigh(f+3) >> 3

	buf[j0] = uint16(sclampHigh(qs0-f1) + mid16)
	buf[i0] = uint16(sclampHigh(ps0+f2) + mid16)

	if !highEdge {
		fh := sclampHigh(f1+1) >> 1
		buf[j1] = uint16(sclampHigh(qs1-fh) + mid16)
		buf[i1] = uint16(sclampHigh(ps1+fh) + mid16)
	}
}

func applyFilter8Vals16(buf []uint16, q0, step, p3, p2, p1, p0, q0v, q1, q2, q3 int) {
	buf[q0-3*step] = clampPix16(round3(3*p3 + 2*p2 + p1 + p0 + q0v))
	buf[q0-2*step] = clampPix16(round3(2*p3 + p2 + 2*p1 + p0 + q0v + q1))
	buf[q0-1*step] = clampPix16(round3(p3 + p2 + p1 + 2*p0 + q0v + q1 + q2))
	buf[q0] = clampPix16(round3(p2 + p1 + p0 + 2*q0v + q1 + q2 + q3))
	buf[q0+1*step] = clampPix16(round3(p1 + p0 + q0v + 2*q1 + q2 + q3 + q3))
	buf[q0+2*step] = clampPix16(round3(p0 + q0v + q1 + 2*q2 + q3 + q3 + q3))
}

func applyFilter16Vals16(buf []uint16, q0, step, p3, p2, p1, p0, q0v, q1, q2, q3 int) {
	p7 := int(buf[q0-8*step])
	p6 := int(buf[q0-7*step])
	p5 := int(buf[q0-6*step])
	p4 := int(buf[q0-5*step])
	q4 := int(buf[q0+4*step])
	q5 := int(buf[q0+5*step])
	q6 := int(buf[q0+6*step])
	q7 := int(buf[q0+7*step])

	buf[q0-7*step] = clampPix16(round4(p7*7 + p6*2 + p5 + p4 + p3 + p2 + p1 + p0 + q0v))
	buf[q0-6*step] = clampPix16(round4(p7*6 + p6 + p5*2 + p4 + p3 + p2 + p1 + p0 + q0v + q1))
	buf[q0-5*step] = clampPix16(round4(p7*5 + p6 + p5 + p4*2 + p3 + p2 + p1 + p0 + q0v + q1 + q2))
	buf[q0-4*step] = clampPix16(round4(p7*4 + p6 + p5 + p4 + p3*2 + p2 + p1 + p0 + q0v + q1 + q2 + q3))
	buf[q0-3*step] = clampPix16(round4(p7*3 + p6 + p5 + p4 + p3 + p2*2 + p1 + p0 + q0v + q1 + q2 + q3 + q4))
	buf[q0-2*step] = clampPix16(round4(p7*2 + p6 + p5 + p4 + p3 + p2 + p1*2 + p0 + q0v + q1 + q2 + q3 + q4 + q5))
	buf[q0-1*step] = clampPix16(round4(p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0*2 + q0v + q1 + q2 + q3 + q4 + q5 + q6))
	buf[q0] = clampPix16(round4(p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0v*2 + q1 + q2 + q3 + q4 + q5 + q6 + q7))
	buf[q0+1*step] = clampPix16(round4(p5 + p4 + p3 + p2 + p1 + p0 + q0v + q1*2 + q2 + q3 + q4 + q5 + q6 + q7*2))
	buf[q0+2*step] = clampPix16(round4(p4 + p3 + p2 + p1 + p0 + q0v + q1 + q2*2 + q3 + q4 + q5 + q6 + q7*3))
	buf[q0+3*step] = clampPix16(round4(p3 + p2 + p1 + p0 + q0v + q1 + q2 + q3*2 + q4 + q5 + q6 + q7*4))
	buf[q0+4*step] = clampPix16(round4(p2 + p1 + p0 + q0v + q1 + q2 + q3 + q4*2 + q5 + q6 + q7*5))
	buf[q0+5*step] = clampPix16(round4(p1 + p0 + q0v + q1 + q2 + q3 + q4 + q5*2 + q6 + q7*6))
	buf[q0+6*step] = clampPix16(round4(p0 + q0v + q1 + q2 + q3 + q4 + q5 + q6*2 + q7*7))
}

// FilterEdge4_16 is FilterEdge4 for the high-bit-depth uint16 path.
func FilterEdge4_16(buf []uint16, q0, step, width int, p Params) {
	p3 := int(buf[q0-4*step])
	p2 := int(buf[q0-3*step])
	p1 := int(buf[q0-2*step])
	p0 := int(buf[q0-1*step])
	q0v := int(buf[q0])
	q1 := int(buf[q0+1*step])
	q2 := int(buf[q0+2*step])
	q3 := int(buf[q0+3*step])

	limI := int(p.Limit) << uint(shift16)
	blimI := int(p.BLimit) << uint(shift16)
	if abs8(p3-p2) > limI || abs8(p2-p1) > limI || abs8(p1-p0) > limI ||
		abs8(q0v-q1) > limI || abs8(q1-q2) > limI || abs8(q2-q3) > limI ||
		abs8(p0-q0v)*2+abs8(p1-q1)/2 > blimI {
		return
	}

	t := int(p.Thresh) << uint(shift16)
	highEdge := abs8(p1-p0) > t || abs8(q0v-q1) > t

	flat := abs8(p1-p0) <= flatThresh16 && abs8(q1-q0v) <= flatThresh16 && abs8(p2-p0) <= flatThresh16 &&
		abs8(q2-q0v) <= flatThresh16 && abs8(p3-p0) <= flatThresh16 && abs8(q3-q0v) <= flatThresh16

	switch width {
	case 16:
		if flat && isFlat216(buf, q0, step) {
			applyFilter16Vals16(buf, q0, step, p3, p2, p1, p0, q0v, q1, q2, q3)
			return
		}
		fallthrough
	case 8:
		if flat {
			applyFilter8Vals16(buf, q0, step, p3, p2, p1, p0, q0v, q1, q2, q3)
			return
		}
		fallthrough
	default:
		applyFilter4_16(buf, q0, step, highEdge)
	}
}
