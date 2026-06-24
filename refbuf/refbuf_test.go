package refbuf

import "testing"

// ---- NewFrame / layout tests ------------------------------------------------

func TestNewFrameSize(t *testing.T) {
	f := NewFrame(64, 48)
	if f.Width != 64 || f.Height != 48 {
		t.Fatalf("dims: got %dx%d, want 64x48", f.Width, f.Height)
	}
	expectedYStride := 64 + 2*BORDER
	if f.YStride != expectedYStride {
		t.Fatalf("YStride: got %d, want %d", f.YStride, expectedYStride)
	}
	expectedYLen := (48 + 2*BORDER) * expectedYStride
	if len(f.Y) != expectedYLen {
		t.Fatalf("Y len: got %d, want %d", len(f.Y), expectedYLen)
	}
}

func TestNewFrameOddDims(t *testing.T) {
	f := NewFrame(7, 5)
	// Should be rounded up to 8×6.
	if f.Width != 8 || f.Height != 6 {
		t.Fatalf("odd dims not rounded: got %dx%d", f.Width, f.Height)
	}
}

func TestYBaseInBounds(t *testing.T) {
	f := NewFrame(16, 16)
	base := f.YBase()
	if base < 0 || base >= len(f.Y) {
		t.Fatalf("YBase %d out of range [0,%d)", base, len(f.Y))
	}
	// Verify there are BORDER rows above and BORDER pixels to the left.
	if base-BORDER < 0 {
		t.Fatal("not enough top border")
	}
	if base-BORDER*f.YStride < 0 {
		t.Fatal("not enough left border")
	}
}

// ---- ExtendBorders ----------------------------------------------------------

func TestExtendBordersLeft(t *testing.T) {
	f := NewFrame(4, 4)
	base := f.YBase()
	// Fill visible area with recognizable values.
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			f.Y[base+r*f.YStride+c] = byte(r*10 + c + 1)
		}
	}
	ExtendBorders(f)
	// Left border of row 0 must equal the first visible pixel (1).
	for i := 1; i <= BORDER; i++ {
		got := f.Y[base-i]
		if got != 1 {
			t.Fatalf("left border pixel -%d: got %d, want 1", i, got)
		}
	}
}

func TestExtendBordersRight(t *testing.T) {
	f := NewFrame(4, 4)
	base := f.YBase()
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			f.Y[base+r*f.YStride+c] = byte(r*10 + c + 1)
		}
	}
	ExtendBorders(f)
	// Right border of row 0: last visible pixel is at col=3 → value = 4.
	for i := 1; i <= BORDER; i++ {
		got := f.Y[base+4-1+i]
		if got != 4 {
			t.Fatalf("right border pixel +%d: got %d, want 4", i, got)
		}
	}
}

func TestExtendBordersTop(t *testing.T) {
	f := NewFrame(4, 4)
	base := f.YBase()
	for r := 0; r < 4; r++ {
		for c := 0; c < 4; c++ {
			f.Y[base+r*f.YStride+c] = byte(r*10 + c + 1)
		}
	}
	ExtendBorders(f)
	// Top border rows must equal row 0.
	for br := 1; br <= BORDER; br++ {
		for c := 0; c < 4; c++ {
			got := f.Y[base-br*f.YStride+c]
			want := f.Y[base+c]
			if got != want {
				t.Fatalf("top border row -%d col %d: got %d, want %d", br, c, got, want)
			}
		}
	}
}

// ---- Buffer.Update / slot management ----------------------------------------

func TestBufferUpdateMask(t *testing.T) {
	var buf Buffer
	f := NewFrame(8, 8)
	buf.Update(f, 0b00000011) // slots 0 and 1
	if buf.Get(0) != f {
		t.Fatal("slot 0 not updated")
	}
	if buf.Get(1) != f {
		t.Fatal("slot 1 not updated")
	}
	if buf.Get(2) != nil {
		t.Fatal("slot 2 should be nil")
	}
}

func TestBufferUpdateNoOverwrite(t *testing.T) {
	var buf Buffer
	f1 := NewFrame(8, 8)
	f2 := NewFrame(8, 8)
	buf.Update(f1, 0b00000001) // slot 0 = f1
	buf.Update(f2, 0b00000010) // slot 1 = f2 (slot 0 unchanged)
	if buf.Get(0) != f1 {
		t.Fatal("slot 0 should still be f1")
	}
	if buf.Get(1) != f2 {
		t.Fatal("slot 1 should be f2")
	}
}

func TestBufferRefAccessors(t *testing.T) {
	var buf Buffer
	f := NewFrame(8, 8)
	buf.Update(f, 0b00000001) // slot 0
	buf.SetRef(0, 0)          // LAST = slot 0
	// All Refs default to 0, so Golden() and AltRef() also point to slot 0.
	if buf.Last() != f {
		t.Fatal("Last() should return f")
	}
	if buf.Golden() != f {
		t.Fatal("Golden() should also be f (Refs[1] defaults to slot 0)")
	}
	if buf.AltRef() != f {
		t.Fatal("AltRef() should also be f (Refs[2] defaults to slot 0)")
	}
}

func TestBufferSetRef(t *testing.T) {
	var buf Buffer
	fLast := NewFrame(8, 8)
	fGolden := NewFrame(16, 16)
	buf.Update(fLast, 0b00000001)   // slot 0
	buf.Update(fGolden, 0b00000010) // slot 1
	buf.SetRef(0, 0)                // LAST = slot 0
	buf.SetRef(1, 1)                // GOLDEN = slot 1
	if buf.Last() != fLast {
		t.Fatal("Last() mismatch")
	}
	if buf.Golden() != fGolden {
		t.Fatal("Golden() mismatch")
	}
}
