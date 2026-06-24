package intra

import (
	"testing"

	"github.com/mgvs/go-vp9/modeinfo"
)

// makeAbove returns a slice of n uint8 values 10, 20, 30, ...
func makeAbove(n int) []uint8 {
	s := make([]uint8, n)
	for i := range s {
		s[i] = uint8((i + 1) * 10)
	}
	return s
}

// makeLeft returns a slice of n uint8 values 100, 110, 120, ...
func makeLeft(n int) []uint8 {
	s := make([]uint8, n)
	for i := range s {
		s[i] = uint8(100 + i*10)
	}
	return s
}

// makeDst returns a zeroed dst slice of stride*bs bytes.
func makeDst(stride, bs int) []uint8 { return make([]uint8, stride*bs) }

// ---- DC ----------------------------------------------------------------

func TestDCUniform(t *testing.T) {
	// above = [100..100], left = [200..200] → avg = (800+800+4)>>3 = 150
	bs := 4
	above := make([]uint8, bs)
	left := make([]uint8, bs)
	for i := range above {
		above[i] = 100
		left[i] = 200
	}
	dst := makeDst(bs, bs)
	Predict(modeinfo.DC, dst, bs, bs, above, left, 150)
	// sum = 4*100 + 4*200 = 1200; avg = (1200+4)>>3 = 150
	for i, v := range dst {
		if v != 150 {
			t.Fatalf("DC uniform: dst[%d] = %d, want 150", i, v)
		}
	}
}

func TestDCRoundingBs4(t *testing.T) {
	// above = [1,2,3,4], left = [5,6,7,8] → sum=36, avg=(36+4)>>3=5
	bs := 4
	above := []uint8{1, 2, 3, 4}
	left := []uint8{5, 6, 7, 8}
	dst := makeDst(bs, bs)
	Predict(modeinfo.DC, dst, bs, bs, above, left, 0)
	want := uint8((36 + 4) >> 3) // 40>>3 = 5
	for i, v := range dst {
		if v != want {
			t.Fatalf("DC bs=4: dst[%d] = %d, want %d", i, v, want)
		}
	}
}

// ---- V ------------------------------------------------------------------

func TestV(t *testing.T) {
	bs := 4
	above := []uint8{10, 20, 30, 40}
	left := makeLeft(bs)
	dst := makeDst(bs, bs)
	Predict(modeinfo.V, dst, bs, bs, above, left, 0)
	for r := 0; r < bs; r++ {
		for c := 0; c < bs; c++ {
			if dst[r*bs+c] != above[c] {
				t.Fatalf("V: dst[%d][%d] = %d, want %d", r, c, dst[r*bs+c], above[c])
			}
		}
	}
}

// ---- H ------------------------------------------------------------------

func TestH(t *testing.T) {
	bs := 4
	above := makeAbove(bs)
	left := []uint8{10, 20, 30, 40}
	dst := makeDst(bs, bs)
	Predict(modeinfo.H, dst, bs, bs, above, left, 0)
	for r := 0; r < bs; r++ {
		for c := 0; c < bs; c++ {
			if dst[r*bs+c] != left[r] {
				t.Fatalf("H: dst[%d][%d] = %d, want left[%d]=%d", r, c, dst[r*bs+c], r, left[r])
			}
		}
	}
}

// ---- TM ------------------------------------------------------------------

func TestTM(t *testing.T) {
	bs := 4
	// above = [100,110,120,130], left = [200,210,220,230], topLeft = 100
	// dst[r][c] = clamp(above[c] + left[r] - topLeft)
	above := []uint8{100, 110, 120, 130}
	left := []uint8{200, 210, 220, 230}
	topLeft := uint8(100)
	dst := makeDst(bs, bs)
	Predict(modeinfo.TM, dst, bs, bs, above, left, topLeft)
	for r := 0; r < bs; r++ {
		for c := 0; c < bs; c++ {
			want := clamp8(int(above[c]) + int(left[r]) - int(topLeft))
			got := dst[r*bs+c]
			if got != want {
				t.Fatalf("TM[%d][%d] = %d, want %d", r, c, got, want)
			}
		}
	}
}

func TestTMClampHigh(t *testing.T) {
	bs := 4
	above := []uint8{200, 200, 200, 200}
	left := []uint8{200, 200, 200, 200}
	topLeft := uint8(0)
	dst := makeDst(bs, bs)
	Predict(modeinfo.TM, dst, bs, bs, above, left, topLeft)
	for _, v := range dst {
		if v != 255 {
			t.Fatalf("TM clamp high: got %d, want 255", v)
		}
	}
}

func TestTMClampLow(t *testing.T) {
	bs := 4
	above := []uint8{0, 0, 0, 0}
	left := []uint8{0, 0, 0, 0}
	topLeft := uint8(200)
	dst := makeDst(bs, bs)
	Predict(modeinfo.TM, dst, bs, bs, above, left, topLeft)
	for _, v := range dst {
		if v != 0 {
			t.Fatalf("TM clamp low: got %d, want 0", v)
		}
	}
}

// ---- D45 ----------------------------------------------------------------

func TestD45(t *testing.T) {
	bs := 4
	// above[0..7] = 10,20,30,40,50,60,70,80
	above := makeAbove(2 * bs)
	left := makeLeft(bs)
	dst := makeDst(bs, bs)
	Predict(modeinfo.D45, dst, bs, bs, above, left, 0)

	// Verify a few pixels:
	// dst[0][0] = avg3(10,20,30) = (10+40+30+2)/4 = 82/4 = 20
	if dst[0] != 20 {
		t.Fatalf("D45[0][0] = %d, want 20", dst[0])
	}
	// dst[0][1] = avg3(20,30,40) = (20+60+40+2)/4 = 122/4 = 30
	if dst[1] != 30 {
		t.Fatalf("D45[0][1] = %d, want 30", dst[1])
	}
	// dst[3][3] = above[7] = 80 (corner case: r+c+2 = 8 = 2*bs)
	if dst[3*bs+3] != 80 {
		t.Fatalf("D45[3][3] = %d, want 80", dst[3*bs+3])
	}
	// dst[1][0] same as dst[0][1] = 30
	if dst[bs] != 30 {
		t.Fatalf("D45[1][0] = %d, want 30", dst[bs])
	}
}

// ---- D135 ---------------------------------------------------------------

func TestD135Diagonal(t *testing.T) {
	// Uniform input: all above=128, left=128, topLeft=128 → all output = 128.
	bs := 4
	above := make([]uint8, bs)
	left := make([]uint8, bs)
	for i := range above {
		above[i] = 128
		left[i] = 128
	}
	dst := makeDst(bs, bs)
	Predict(modeinfo.D135, dst, bs, bs, above, left, 128)
	for i, v := range dst {
		if v != 128 {
			t.Fatalf("D135 uniform: dst[%d] = %d, want 128", i, v)
		}
	}
}

func TestD135Symmetry(t *testing.T) {
	// For D135, dst[r][c] == dst[r+1][c+1] when both are in range
	// (samples repeat along the upper-left diagonal).
	bs := 4
	above := makeAbove(bs)
	left := makeLeft(bs)
	dst := makeDst(bs, bs)
	Predict(modeinfo.D135, dst, bs, bs, above, left, 77)
	for r := 0; r+1 < bs; r++ {
		for c := 0; c+1 < bs; c++ {
			if dst[r*bs+c] != dst[(r+1)*bs+(c+1)] {
				t.Fatalf("D135 diagonal symmetry broken at [%d][%d] vs [%d][%d]: %d != %d",
					r, c, r+1, c+1, dst[r*bs+c], dst[(r+1)*bs+(c+1)])
			}
		}
	}
}

// ---- D207 ---------------------------------------------------------------

func TestD207Corner(t *testing.T) {
	// left all same value → all output = that value.
	bs := 4
	left := make([]uint8, bs)
	for i := range left {
		left[i] = 100
	}
	dst := makeDst(bs, bs)
	Predict(modeinfo.D207, dst, bs, bs, makeAbove(bs), left, 100)
	for i, v := range dst {
		if v != 100 {
			t.Fatalf("D207 uniform: dst[%d] = %d, want 100", i, v)
		}
	}
}

// ---- all modes × all block sizes: no panic, values in [0,255] -----------

func TestAllModesAllSizes(t *testing.T) {
	blockSizes := []int{4, 8, 16, 32}
	for _, bs := range blockSizes {
		above := makeAbove(2 * bs) // max needed for D45
		left := makeLeft(2 * bs)  // extra slack
		topLeft := uint8(128)
		dst := make([]uint8, bs*bs)
		for m := 0; m < modeinfo.Count; m++ {
			for i := range dst {
				dst[i] = 0
			}
			Predict(modeinfo.IntraMode(m), dst, bs, bs, above, left, topLeft)
			for i, v := range dst {
				_ = v // all values are uint8 so always [0,255]
				if dst[i] != v {
					t.Fatalf("mode=%d bs=%d: impossible", m, bs)
				}
			}
		}
	}
}

// ---- D63 ----------------------------------------------------------------

func TestD63Bs4(t *testing.T) {
	bs := 4
	// above = [10,20,30,40,50,60,70] (need bs+bs/2+2 = 8)
	above := makeAbove(2 * bs)
	left := makeLeft(bs)
	dst := makeDst(bs, bs)
	Predict(modeinfo.D63, dst, bs, bs, above, left, 0)
	// Row 0 (even): avg2(above[0],above[1])..avg2(above[3],above[4])
	// dst[0][0] = avg2(10,20) = (10+20+1)/2 = 15
	if dst[0] != 15 {
		t.Fatalf("D63[0][0] = %d, want 15", dst[0])
	}
	// Row 1 (odd): avg3(above[0],above[1],above[2])..
	// dst[1][0] = avg3(10,20,30) = (10+40+30+2)/4 = 20
	if dst[bs] != 20 {
		t.Fatalf("D63[1][0] = %d, want 20", dst[bs])
	}
}

// ---- D117 ---------------------------------------------------------------

func TestD117Row0(t *testing.T) {
	bs := 4
	above := []uint8{20, 40, 60, 80}
	left := makeLeft(bs)
	topLeft := uint8(10)
	dst := makeDst(bs, bs)
	Predict(modeinfo.D117, dst, bs, bs, above, left, topLeft)
	// Row 0: dst[0][0] = avg2(topLeft,above[0]) = avg2(10,20) = 15
	if dst[0] != 15 {
		t.Fatalf("D117[0][0] = %d, want 15", dst[0])
	}
	// dst[0][1] = avg2(above[0],above[1]) = avg2(20,40) = 30
	if dst[1] != 30 {
		t.Fatalf("D117[0][1] = %d, want 30", dst[1])
	}
}

// ---- D153 ---------------------------------------------------------------

func TestD153Col0(t *testing.T) {
	bs := 4
	above := makeAbove(bs)
	left := []uint8{20, 40, 60, 80}
	topLeft := uint8(10)
	dst := makeDst(bs, bs)
	Predict(modeinfo.D153, dst, bs, bs, above, left, topLeft)
	// dst[0][0] = avg2(topLeft,left[0]) = avg2(10,20) = 15
	if dst[0] != 15 {
		t.Fatalf("D153[0][0] = %d, want 15", dst[0])
	}
	// dst[1][0] = avg2(left[0],left[1]) = avg2(20,40) = 30
	if dst[bs] != 30 {
		t.Fatalf("D153[1][0] = %d, want 30", dst[bs])
	}
}
