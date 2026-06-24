package transform

import "testing"

// ---- helpers ----------------------------------------------------------------

func makeCoeffs(bs int, dc int16) []int16 {
	c := make([]int16, bs*bs)
	c[0] = dc
	return c
}

func allEqual(t *testing.T, label string, buf []int16, want int16) {
	t.Helper()
	for i, v := range buf {
		if v != want {
			t.Fatalf("%s: buf[%d] = %d, want %d", label, i, v, want)
		}
	}
}

// ---- 1-D transforms ---------------------------------------------------------

func TestIDCT4Zero(t *testing.T) {
	io := make([]int16, 4)
	idct4(io)
	allEqual(t, "idct4 zero", io, 0)
}

func TestIDCT4DC(t *testing.T) {
	// idct4([1024,0,0,0]) → all drs(1024*11585) = 724
	io := []int16{1024, 0, 0, 0}
	idct4(io)
	allEqual(t, "idct4 dc=1024", io, 724)
}

func TestIDCT4Known(t *testing.T) {
	// Hand-computed: idct4([1,2,3,4])
	// a = b = (1+3)*11585 = 46340; c = 2*6270-4*15137 = -48008; d = 2*15137+4*6270 = 55354
	// io[0]=drs(46340+55354)=drs(101694)=6
	// io[1]=drs(46340-55354+0-48008)=drs(-71178)=-4  (b+c with b=-23170)
	// io[2]=drs(-23170+48008)=drs(24838)=2
	// io[3]=drs(46340-55354)=drs(-9014)=-1
	io := []int16{1, 2, 3, 4}
	idct4(io)
	want := [4]int16{6, -4, 2, -1}
	for i, v := range io {
		if v != want[i] {
			t.Fatalf("idct4[%d] = %d, want %d", i, v, want[i])
		}
	}
}

func TestIDCT8Zero(t *testing.T) {
	io := make([]int16, 8)
	idct8(io)
	allEqual(t, "idct8 zero", io, 0)
}

func TestIDCT8DC(t *testing.T) {
	// idct8([1024,0,...]) → all e0 = drs(1024*11585) = 724
	io := make([]int16, 8)
	io[0] = 1024
	idct8(io)
	allEqual(t, "idct8 dc=1024", io, 724)
}

func TestIDCT16Zero(t *testing.T) {
	io := make([]int16, 16)
	idct16(io)
	allEqual(t, "idct16 zero", io, 0)
}

func TestIDCT16DC(t *testing.T) {
	// idct16([1024,0,...]) → all drs(1024*cp16) = 724
	io := make([]int16, 16)
	io[0] = 1024
	idct16(io)
	allEqual(t, "idct16 dc=1024", io, 724)
}

func TestIDCT32Zero(t *testing.T) {
	io := make([]int16, 32)
	idct32(io)
	allEqual(t, "idct32 zero", io, 0)
}

func TestIDCT32DC(t *testing.T) {
	// idct32([1024,0,...]) → all drs(1024*cp16) = 724
	io := make([]int16, 32)
	io[0] = 1024
	idct32(io)
	allEqual(t, "idct32 dc=1024", io, 724)
}

// ---- 2-D transforms ---------------------------------------------------------

// For any block size, 2D DCT_DCT with DC=1024 only:
//   row pass: [1024,0,...] → uniform 724 → >>4 → 45
//   col pass: [45,0,...] → uniform 32 → >>4 → 2
func Test2D_DCT_DC_4(t *testing.T)  { test2DDC(t, 4) }
func Test2D_DCT_DC_8(t *testing.T)  { test2DDC(t, 8) }
func Test2D_DCT_DC_16(t *testing.T) { test2DDC(t, 16) }
func Test2D_DCT_DC_32(t *testing.T) { test2DDC(t, 32) }

func test2DDC(t *testing.T, bs int) {
	t.Helper()
	c := makeCoeffs(bs, 1024)
	InvTransform2D(c, bs, DCT_DCT)
	allEqual(t, "2d dct dc=1024 bs="+itoa(bs), c, 2)
}

// ---- ADST 1-D: zero and symmetry checks ------------------------------------

func TestIADST4Zero(t *testing.T) {
	io := make([]int16, 4)
	iadst4(io)
	allEqual(t, "iadst4 zero", io, 0)
}

func TestIADST8Zero(t *testing.T) {
	io := make([]int16, 8)
	iadst8(io)
	allEqual(t, "iadst8 zero", io, 0)
}

func TestIADST16Zero(t *testing.T) {
	io := make([]int16, 16)
	iadst16(io)
	allEqual(t, "iadst16 zero", io, 0)
}

// 2-D ADST paths: smoke test that output values stay in int16 range.
func TestInv2D_ADST_DCT_4(t *testing.T)  { test2DAdst(t, 4, ADST_DCT) }
func TestInv2D_DCT_ADST_4(t *testing.T)  { test2DAdst(t, 4, DCT_ADST) }
func TestInv2D_ADST_ADST_4(t *testing.T) { test2DAdst(t, 4, ADST_ADST) }
func TestInv2D_ADST_DCT_8(t *testing.T)  { test2DAdst(t, 8, ADST_DCT) }
func TestInv2D_DCT_ADST_8(t *testing.T)  { test2DAdst(t, 8, DCT_ADST) }

func test2DAdst(t *testing.T, bs int, tt TxType) {
	t.Helper()
	c := make([]int16, bs*bs)
	c[0] = 256 // small DC to avoid overflow
	InvTransform2D(c, bs, tt)
	for i, v := range c {
		if v < -2048 || v > 2047 {
			t.Fatalf("2d adst bs=%d tt=%d c[%d]=%d out of residual range", bs, tt, i, v)
		}
	}
}

// itoa for small non-negative ints without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [10]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
