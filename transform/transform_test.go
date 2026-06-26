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
	// Hand-computed against libvpx idct4_c (drs each stage-1 term separately):
	// s0=drs(4*11585)=3, s1=drs(-2*11585)=-1, s2=drs(2*6270-4*15137)=-3,
	// s3=drs(2*15137+4*6270)=3
	// out[0]=s0+s3=6, out[1]=s1+s2=-4, out[2]=s1-s2=2, out[3]=s0-s3=0
	io := []int16{1, 2, 3, 4}
	idct4(io)
	want := [4]int16{6, -4, 2, 0}
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

// For any block size, 2D DCT_DCT with DC=1024 only (libvpx vpx_idctNxN_1_add):
//   out = drs(1024·cp16) = 724;  out = drs(724·cp16) = 512;
//   result = ROUND_POWER_OF_TWO(512, FinalShift(bs)).
// FinalShift = 4/5/6/6 → result = 32/16/8/8 for bs = 4/8/16/32.
func Test2D_DCT_DC_4(t *testing.T)  { test2DDC(t, 4, 32) }
func Test2D_DCT_DC_8(t *testing.T)  { test2DDC(t, 8, 16) }
func Test2D_DCT_DC_16(t *testing.T) { test2DDC(t, 16, 8) }
func Test2D_DCT_DC_32(t *testing.T) { test2DDC(t, 32, 8) }

func test2DDC(t *testing.T, bs int, want int16) {
	t.Helper()
	c := makeCoeffs(bs, 1024)
	InvTransform2D(c, bs, DCT_DCT)
	allEqual(t, "2d dct dc=1024 bs="+itoa(bs), c, want)
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
