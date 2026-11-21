package wht

import "testing"

// TestIWHT4x4ZeroInput verifies that all-zero coefficients produce all-zero output.
func TestIWHT4x4ZeroInput(t *testing.T) {
	c := make([]int16, 16)
	IWHT4x4(c)
	for i, v := range c {
		if v != 0 {
			t.Fatalf("zero input: c[%d] = %d, want 0", i, v)
		}
	}
}

// TestIWHT4x4DCOnly64 verifies the DC-only case: input[0]=64 → all output = 4.
// WHT gain with UNIT_QUANT_SHIFT=2: DC value N → N/16.
func TestIWHT4x4DCOnly64(t *testing.T) {
	c := make([]int16, 16)
	c[0] = 64
	IWHT4x4(c)
	for i, v := range c {
		if v != 4 {
			t.Fatalf("DC=64: c[%d] = %d, want 4", i, v)
		}
	}
}

// TestIWHT4x4DCOnly256 verifies DC=256 → all output = 16.
func TestIWHT4x4DCOnly256(t *testing.T) {
	c := make([]int16, 16)
	c[0] = 256
	IWHT4x4(c)
	for i, v := range c {
		if v != 16 {
			t.Fatalf("DC=256: c[%d] = %d, want 16", i, v)
		}
	}
}

// TestIWHT4x4DCAndB verifies a two-nonzero case to exercise the b-coefficient
// input path (coeff[3] maps to butterfly variable b).
//
// Input: c[0]=16 (DC), c[3]=16 (b-coeff in row 0), rest zero.
//
// Hand-computed expected output:
//
//	col:   0  1  2  3
//	row 0: 2  0  2  0
//	row 1: 2  0  2  0
//	row 2: 2  0  2  0
//	row 3: 2  0  2  0
func TestIWHT4x4DCAndB(t *testing.T) {
	c := make([]int16, 16)
	c[0] = 16 // DC
	c[3] = 16 // b-coeff (row 0, index 3)
	IWHT4x4(c)

	want := [16]int16{
		2, 0, 2, 0,
		2, 0, 2, 0,
		2, 0, 2, 0,
		2, 0, 2, 0,
	}
	for i, v := range c {
		if v != want[i] {
			t.Fatalf("DC+b: c[%d] = %d, want %d", i, v, want[i])
		}
	}
}

// TestIWHT4x4Symmetry verifies that identical non-zero DC coefficients across
// all rows produce a uniform output (row-symmetry property of WHT).
//
// Input: c[0]=c[4]=c[8]=c[12]=16 (DC of each row = 16), rest zero.
// Row pass: each row [16,0,0,0] → shifts to [4,0,0,0] → WHT → [2,2,2,2]
// Column pass: each column [2,2,2,2] → WHT with no shift → [2,2,2,2]
// (verified: iwht4(2,2,2,2): a1=4,d1=0,e1=2,b=0,c=0,a=4,d=0 → (4,0,0,0)?
//
// Re-computed: let all row DCs = 16:
// Row pass per row: a=4,b=0,cc=0,d=0 → e1=2,b=2,cc=2,a=2,d=2 → [2,2,2,2]
// After row pass: all 16 values = 2.
// Column pass: each col has a=2,b=2,cc=2,d=2
//   iwht4(2,2,2,2): a1=4,d1=0,e1=2,b=0,cc=0,a=4,d=0 → (4,0,0,0)
// Final: col[row0]=4, col[row1]=0, col[row2]=0, col[row3]=0 for each col.
func TestIWHT4x4RowSymmetry(t *testing.T) {
	c := make([]int16, 16)
	c[0] = 16  // row 0 DC
	c[4] = 16  // row 1 DC
	c[8] = 16  // row 2 DC
	c[12] = 16 // row 3 DC
	IWHT4x4(c)

	// Row 0: [4,4,4,4]; rows 1-3: [0,0,0,0]
	want := [16]int16{
		4, 4, 4, 4,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	for i, v := range c {
		if v != want[i] {
			t.Fatalf("row-symmetry: c[%d] = %d, want %d", i, v, want[i])
		}
	}
}

// TestIWHT4x4ColumnOrder pins the column pass input mapping of libvpx
// vpx_iwht4x4_16_add_c (a1,c1,d1,b1 = rows 0,1,2,3). Input: only c[4]=16.
// Row pass: row 1 = [2,2,2,2]. Column pass on (0,2,0,0): a1=2, d1=0, e1=1,
// b1=1, c1=-1, a1=1, d1=-1 → rows 1, 1, -1, -1.
func TestIWHT4x4ColumnOrder(t *testing.T) {
	want := [16]int16{
		1, 1, 1, 1,
		1, 1, 1, 1,
		-1, -1, -1, -1,
		-1, -1, -1, -1,
	}
	c := make([]int16, 16)
	c[4] = 16
	IWHT4x4(c)
	for i, v := range c {
		if v != want[i] {
			t.Fatalf("IWHT4x4: c[%d] = %d, want %d", i, v, want[i])
		}
	}
	c16 := make([]int64, 16)
	c16[4] = 16
	IWHT4x4_16(c16)
	for i, v := range c16 {
		if v != int64(want[i]) {
			t.Fatalf("IWHT4x4_16: c[%d] = %d, want %d", i, v, want[i])
		}
	}
}
