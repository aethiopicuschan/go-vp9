package dequant

import "testing"

func TestDCQuantBoundary(t *testing.T) {
	if DCQuant(0) != 4 {
		t.Fatalf("DCQuant(0) = %d, want 4", DCQuant(0))
	}
	if DCQuant(255) != 1336 {
		t.Fatalf("DCQuant(255) = %d, want 1336", DCQuant(255))
	}
}

func TestACQuantBoundary(t *testing.T) {
	if ACQuant(0) != 4 {
		t.Fatalf("ACQuant(0) = %d, want 4", ACQuant(0))
	}
	if ACQuant(255) != 1880 {
		t.Fatalf("ACQuant(255) = %d, want 1880", ACQuant(255))
	}
}

func TestDequantBlock(t *testing.T) {
	coeffs := []int16{3, 2, -1, 0}
	DequantBlock(coeffs, 10, 8, 0)
	// dc: 3*10=30; ac: 2*8=16, -1*8=-8, 0*8=0
	want := []int16{30, 16, -8, 0}
	for i, v := range coeffs {
		if v != want[i] {
			t.Fatalf("DequantBlock[%d] = %d, want %d", i, v, want[i])
		}
	}
}

func TestDequantBlockClamp(t *testing.T) {
	coeffs := []int16{32767, 32767}
	DequantBlock(coeffs, 2, 2, 0)
	// 32767 * 2 = 65534 > 32767 → clamp to 32767
	if coeffs[0] != 32767 {
		t.Fatalf("clamp dc: got %d, want 32767", coeffs[0])
	}
	if coeffs[1] != 32767 {
		t.Fatalf("clamp ac: got %d, want 32767", coeffs[1])
	}
}

func TestQIndexClamp(t *testing.T) {
	if DCQuant(-1) != DCQuant(0) {
		t.Fatal("negative qIndex not clamped to 0")
	}
	if DCQuant(256) != DCQuant(255) {
		t.Fatal("qIndex > 255 not clamped to 255")
	}
}
