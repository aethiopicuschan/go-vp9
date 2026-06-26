package intra

// Directional intra predictors — direct ports of libvpx vpx_dsp/intrapred.c.
// VP9 hand-codes a distinct 4x4 variant (full above-right diagonal) and uses a
// generic algorithm for 8/16/32 (which replicates above[bs-1]/left[bs-1] past
// the diagonal). DST(x,y) in libvpx is dst[x + y*stride]; AVG2/AVG3 = avg2/avg3.
// `topLeft` corresponds to libvpx above[-1].

// ---- D45 (libvpx d45_predictor) --------------------------------------------

func d45Predict(dst []uint8, stride, bs int, above []uint8) {
	if bs == 4 {
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				if k := r + c; k < 6 {
					dst[r*stride+c] = avg3(above[k], above[k+1], above[k+2])
				} else {
					dst[r*stride+c] = above[7]
				}
			}
		}
		return
	}
	ar := above[bs-1]
	for r := 0; r < bs; r++ {
		for c := 0; c < bs; c++ {
			if r+c < bs-1 {
				dst[r*stride+c] = avg3(above[r+c], above[r+c+1], above[r+c+2])
			} else {
				dst[r*stride+c] = ar
			}
		}
	}
}

// ---- D63 (libvpx d63_predictor) --------------------------------------------

func d63Predict(dst []uint8, stride, bs int, above []uint8) {
	if bs == 4 {
		// vpx_d63_predictor_4x4_c: clamped above-right diagonal.
		for r := 0; r < 4; r++ {
			half := r >> 1
			for c := 0; c < 4; c++ {
				i0, i1, i2 := clamp(half+c, 7), clamp(half+c+1, 7), clamp(half+c+2, 7)
				if r&1 == 0 {
					dst[r*stride+c] = avg2(above[i0], above[i1])
				} else {
					dst[r*stride+c] = avg3(above[i0], above[i1], above[i2])
				}
			}
		}
		return
	}
	for c := 0; c < bs; c++ {
		dst[c] = avg2(above[c], above[c+1])
		dst[stride+c] = avg3(above[c], above[c+1], above[c+2])
	}
	ar := above[bs-1]
	size := bs - 2
	for r := 2; r < bs; r, size = r+2, size-1 {
		h := r >> 1
		for c := 0; c < bs; c++ {
			if c < size {
				dst[r*stride+c] = dst[h+c]            // row 0 offset r/2
				dst[(r+1)*stride+c] = dst[stride+h+c] // row 1 offset r/2
			} else {
				dst[r*stride+c] = ar
				dst[(r+1)*stride+c] = ar
			}
		}
	}
}

// ---- D117 (libvpx d117_predictor) ------------------------------------------

func d117Predict(dst []uint8, stride, bs int, above, left []uint8, topLeft uint8) {
	if bs == 4 {
		I, J, K := int(left[0]), int(left[1]), int(left[2])
		X := topLeft
		A, B, C, D := above[0], above[1], above[2], above[3]
		set := func(x, y int, v uint8) { dst[y*stride+x] = v }
		set(0, 0, avg2(X, A))
		set(1, 2, avg2(X, A))
		set(1, 0, avg2(A, B))
		set(2, 2, avg2(A, B))
		set(2, 0, avg2(B, C))
		set(3, 2, avg2(B, C))
		set(3, 0, avg2(C, D))
		set(0, 3, avg3u(K, J, I))
		set(0, 2, avg3u(J, I, int(X)))
		set(0, 1, avg3(left[0], X, A))
		set(1, 3, avg3(left[0], X, A))
		set(1, 1, avg3(X, A, B))
		set(2, 3, avg3(X, A, B))
		set(2, 1, avg3(A, B, C))
		set(3, 3, avg3(A, B, C))
		set(3, 1, avg3(B, C, D))
		return
	}
	// row 0: AVG2(above[c-1], above[c]); above[-1]=topLeft
	dst[0] = avg2(topLeft, above[0])
	for c := 1; c < bs; c++ {
		dst[c] = avg2(above[c-1], above[c])
	}
	// row 1
	dst[stride] = avg3(left[0], topLeft, above[0])
	dst[stride+1] = avg3(topLeft, above[0], above[1])
	for c := 2; c < bs; c++ {
		dst[stride+c] = avg3(above[c-2], above[c-1], above[c])
	}
	// rest of first column
	dst[2*stride] = avg3(topLeft, left[0], left[1])
	for r := 3; r < bs; r++ {
		dst[r*stride] = avg3(left[r-3], left[r-2], left[r-1])
	}
	// rest of the block: row r col c = row r-2 col c-1
	for r := 2; r < bs; r++ {
		for c := 1; c < bs; c++ {
			dst[r*stride+c] = dst[(r-2)*stride+c-1]
		}
	}
}

// ---- D153 (libvpx d153_predictor) ------------------------------------------

func d153Predict(dst []uint8, stride, bs int, above, left []uint8, topLeft uint8) {
	if bs == 4 {
		I, J, K, L := left[0], left[1], left[2], left[3]
		X := topLeft
		A, B, C := above[0], above[1], above[2]
		set := func(x, y int, v uint8) { dst[y*stride+x] = v }
		set(0, 0, avg2(I, X))
		set(2, 1, avg2(I, X))
		set(0, 1, avg2(J, I))
		set(2, 2, avg2(J, I))
		set(0, 2, avg2(K, J))
		set(2, 3, avg2(K, J))
		set(0, 3, avg2(L, K))
		set(3, 0, avg3(A, B, C))
		set(2, 0, avg3(X, A, B))
		set(1, 0, avg3(I, X, A))
		set(3, 1, avg3(I, X, A))
		set(1, 1, avg3(J, I, X))
		set(3, 2, avg3(J, I, X))
		set(1, 2, avg3(K, J, I))
		set(3, 3, avg3(K, J, I))
		set(1, 3, avg3(L, K, J))
		return
	}
	// column 0
	dst[0] = avg2(topLeft, left[0])
	for r := 1; r < bs; r++ {
		dst[r*stride] = avg2(left[r-1], left[r])
	}
	// column 1
	dst[1] = avg3(left[0], topLeft, above[0])
	dst[stride+1] = avg3(topLeft, left[0], left[1])
	for r := 2; r < bs; r++ {
		dst[r*stride+1] = avg3(left[r-2], left[r-1], left[r])
	}
	// column 2, row 0
	dst[2] = avg3(topLeft, above[0], above[1])
	for c := 1; c < bs-2; c++ {
		dst[2+c] = avg3(above[c-1], above[c], above[c+1])
	}
	// rest: row r col(2+c) = row r-1 col c
	for r := 1; r < bs; r++ {
		for c := 0; c < bs-2; c++ {
			dst[r*stride+2+c] = dst[(r-1)*stride+c]
		}
	}
}

// ---- D207 (libvpx d207_predictor) ------------------------------------------

func d207Predict(dst []uint8, stride, bs int, left []uint8) {
	if bs == 4 {
		I, J, K, L := left[0], left[1], left[2], left[3]
		set := func(x, y int, v uint8) { dst[y*stride+x] = v }
		set(0, 0, avg2(I, J))
		set(2, 0, avg2(J, K))
		set(0, 1, avg2(J, K))
		set(2, 1, avg2(K, L))
		set(0, 2, avg2(K, L))
		set(1, 0, avg3(I, J, K))
		set(3, 0, avg3(J, K, L))
		set(1, 1, avg3(J, K, L))
		set(3, 1, avg3(K, L, L))
		set(1, 2, avg3(K, L, L))
		set(3, 2, L)
		set(2, 2, L)
		set(0, 3, L)
		set(1, 3, L)
		set(2, 3, L)
		set(3, 3, L)
		return
	}
	// column 0
	for r := 0; r < bs-1; r++ {
		dst[r*stride] = avg2(left[r], left[r+1])
	}
	dst[(bs-1)*stride] = left[bs-1]
	// column 1
	for r := 0; r < bs-2; r++ {
		dst[r*stride+1] = avg3(left[r], left[r+1], left[r+2])
	}
	dst[(bs-2)*stride+1] = avg3(left[bs-2], left[bs-1], left[bs-1])
	dst[(bs-1)*stride+1] = left[bs-1]
	// rest of last row (cols 2..bs-1) and recurrence
	for c := 0; c < bs-2; c++ {
		dst[(bs-1)*stride+2+c] = left[bs-1]
	}
	for r := bs - 2; r >= 0; r-- {
		for c := 0; c < bs-2; c++ {
			dst[r*stride+2+c] = dst[(r+1)*stride+c]
		}
	}
}

// ---- helpers ---------------------------------------------------------------

func clamp(i, max int) int {
	if i > max {
		return max
	}
	return i
}

// avg3u is avg3 on int args (for negative/promoted intermediates).
func avg3u(a, b, c int) uint8 { return uint8((a + 2*b + c + 2) >> 2) }
