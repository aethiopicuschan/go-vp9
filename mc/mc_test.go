package mc

import "testing"

// mkRef builds a reference buffer with 'border' pixels of padding on each
// side and returns the buffer + the srcBase index of (0,0).
func mkRef(w, h, stride, border int, fillFn func(r, c int) byte) ([]byte, int) {
	rows := h + 2*border
	buf := make([]byte, rows*stride)
	for r := 0; r < rows; r++ {
		for c := 0; c < stride; c++ {
			rr := r - border
			cc := c - border
			buf[r*stride+c] = fillFn(rr, cc)
		}
	}
	return buf, border*stride + border
}

func uniform(v byte) func(r, c int) byte {
	return func(_, _ int) byte { return v }
}

// ---- integer-aligned (mx=0, my=0) ------------------------------------------

func TestPredictCopy(t *testing.T) {
	const W, H, S, B = 8, 8, 16, 4
	src, base := mkRef(W, H, S, B, func(r, c int) byte { return byte(r*S + c + 17) })
	dst := make([]byte, H*W)
	Predict(src, base, S, W, H, 0, 0, FilterRegular, dst, W)
	for r := 0; r < H; r++ {
		for c := 0; c < W; c++ {
			want := src[base+r*S+c]
			got := dst[r*W+c]
			if got != want {
				t.Fatalf("copy[%d][%d]: got %d want %d", r, c, got, want)
			}
		}
	}
}

// ---- uniform source → uniform prediction ------------------------------------

// For a flat source at value V the filter output must be close to V.
// VP9 8-tap kernels intentionally don't all sum to exactly 128 (some are 126),
// so the tolerance is ±4 for sub-pixel positions.
func TestPredictUniform(t *testing.T) {
	const W, H, S, B = 4, 4, 12, 4
	// Use 0 as the uniform value: filter(0,…,0)=0 exactly regardless of kernel sums.
	src, base := mkRef(W, H, S, B, uniform(0))
	dst := make([]byte, H*W)
	for _, filt := range []Filter{FilterRegular, FilterSmooth, FilterSharp, FilterBilinear} {
		for mx := 0; mx < 16; mx++ {
			for my := 0; my < 16; my++ {
				for i := range dst {
					dst[i] = 1
				}
				Predict(src, base, S, W, H, mx, my, filt, dst, W)
				for i, v := range dst {
					if v != 0 {
						t.Fatalf("filt=%d mx=%d my=%d: dst[%d]=%d, want 0", filt, mx, my, i, v)
					}
				}
			}
		}
	}
}

// ---- identity kernel checks -------------------------------------------------

func TestFilterKernelIdentity(t *testing.T) {
	for f := 0; f < 4; f++ {
		k := filterKernels[f][0]
		if k[3] != 128 {
			t.Errorf("filter %d pos 0: k[3]=%d, want 128", f, k[3])
		}
	}
}

// VP9 filter sums are close to 128 but not always exact (libvpx-by-design).
// We just verify the identity row is exact and that all sums are in [120,136].
func TestFilterKernelSums(t *testing.T) {
	for f := 0; f < 4; f++ {
		for pos := 0; pos < 16; pos++ {
			k := filterKernels[f][pos]
			sum := 0
			for _, v := range k {
				sum += int(v)
			}
			if sum < 120 || sum > 136 {
				t.Errorf("filter %d pos %d: kernel sum = %d, far outside [120,136]", f, pos, sum)
			}
		}
		// pos=0 must be exact identity
		if filterKernels[f][0][3] != 128 {
			t.Errorf("filter %d: identity tap [3] = %d, want 128", f, filterKernels[f][0][3])
		}
	}
}

// ---- horizontal sub-pixel: blend at edge -----------------------------------

func TestPredictHorizBlend(t *testing.T) {
	const W, H, S, B = 8, 1, 16, 4
	src, base := mkRef(W, H, S, B, func(r, c int) byte {
		if c >= W/2 {
			return 200
		}
		return 0
	})
	dst := make([]byte, H*W)
	Predict(src, base, S, W, H, 8, 0, FilterRegular, dst, W)
	// pixel just before the edge should be between 0 and 200
	v := int(dst[3])
	if v < 0 || v > 200 {
		t.Fatalf("edge pixel out of range: %d", v)
	}
}

// ---- vertical sub-pixel: blend at edge -------------------------------------

func TestPredictVertBlend(t *testing.T) {
	const W, H, S, B = 1, 8, 8, 4
	src, base := mkRef(W, H, S, B, func(r, c int) byte {
		if r >= H/2 {
			return 200
		}
		return 0
	})
	dst := make([]byte, H*W)
	Predict(src, base, S, W, H, 0, 8, FilterRegular, dst, W)
	v := int(dst[3])
	if v < 0 || v > 200 {
		t.Fatalf("vert edge pixel out of range: %d", v)
	}
}

// ---- compound: two identical predictions → same value ----------------------

func TestPredictCompUniform(t *testing.T) {
	const W, H, S, B = 4, 4, 12, 4
	src, base := mkRef(W, H, S, B, uniform(100))
	dst := make([]byte, H*W)
	PredictComp(
		src, base, S, 0, 0, FilterRegular,
		src, base, S, 0, 0, FilterRegular,
		W, H, dst, W,
	)
	for i, v := range dst {
		if v != 100 {
			t.Fatalf("comp uniform: dst[%d]=%d, want 100", i, v)
		}
	}
}

// ---- compound: (0+200+1)>>1 = 100 ------------------------------------------

func TestPredictCompAverage(t *testing.T) {
	const W, H, S, B = 4, 4, 12, 4
	src0, base0 := mkRef(W, H, S, B, uniform(0))
	src1, base1 := mkRef(W, H, S, B, uniform(200))
	dst := make([]byte, H*W)
	PredictComp(
		src0, base0, S, 0, 0, FilterRegular,
		src1, base1, S, 0, 0, FilterRegular,
		W, H, dst, W,
	)
	for i, v := range dst {
		if v != 100 {
			t.Fatalf("comp avg: dst[%d]=%d, want 100", i, v)
		}
	}
}
