package ctx

import "testing"

// ---- Row / Col construction --------------------------------------------------

func TestNewRowN4(t *testing.T) {
	r := NewRow(64)
	if r.N4 != 16 {
		t.Fatalf("N4: got %d, want 16", r.N4)
	}
	if len(r.TxSize) != 16 {
		t.Fatalf("TxSize len: got %d, want 16", len(r.TxSize))
	}
	if len(r.Partition) != 8 {
		t.Fatalf("Partition len: got %d, want 8", len(r.Partition))
	}
	for p := 0; p < 3; p++ {
		if len(r.NZC[p]) != 16 {
			t.Fatalf("NZC[%d] len: got %d, want 16", p, len(r.NZC[p]))
		}
	}
}

func TestNewRowOddWidth(t *testing.T) {
	r := NewRow(65)
	if r.N4 != 17 {
		t.Fatalf("N4 for 65px: got %d, want 17", r.N4)
	}
}

func TestNewColN4(t *testing.T) {
	c := NewCol()
	if c.N4 != 16 {
		t.Fatalf("Col N4: got %d, want 16", c.N4)
	}
	if len(c.Partition) != 8 {
		t.Fatalf("Col Partition len: got %d, want 8", len(c.Partition))
	}
}

// ---- Reset defaults ----------------------------------------------------------

func TestRowResetDefaults(t *testing.T) {
	r := NewRow(32) // N4=8
	// After Reset, TxSize should be TX32x32, Ref should be [RefIntra, RefIntra].
	for i, tx := range r.TxSize {
		if tx != TX32x32 {
			t.Fatalf("TxSize[%d] after Reset: got %d, want TX32x32=%d", i, tx, TX32x32)
		}
	}
	for i, ref := range r.Ref {
		if ref != [2]int{RefIntra, RefIntra} {
			t.Fatalf("Ref[%d] after Reset: got %v, want intra", i, ref)
		}
	}
}

func TestRowResetClearsNZC(t *testing.T) {
	r := NewRow(16) // N4=4
	r.NZC[0][0] = true
	r.Reset()
	if r.NZC[0][0] {
		t.Fatal("NZC[0][0] should be false after Reset")
	}
}

// ---- Partition context -------------------------------------------------------

func TestPartitionAboveFalseDefault(t *testing.T) {
	r := NewRow(64)
	// After Reset, all partition masks are 0 → no bit set → false for any bsl.
	for bsl := 1; bsl <= 4; bsl++ {
		if PartitionAbove(r, 0, bsl) {
			t.Fatalf("PartitionAbove(col4=0, bsl=%d): want false after Reset", bsl)
		}
	}
}

func TestSetPartitionAndRead(t *testing.T) {
	r := NewRow(64)
	// Write mask=15 (0b00001111) to 8×8 position 0 (4 8×8 units).
	r.SetPartition(0, 1, 15)
	// bsl=1 → bit 0 → (15>>0)&1 = 1 → true
	if !PartitionAbove(r, 0, 1) {
		t.Fatal("bsl=1: expected true")
	}
	// bsl=4 → bit 3 → (15>>3)&1 = 1 → true
	if !PartitionAbove(r, 0, 4) {
		t.Fatal("bsl=4: expected true (15>>3=1)")
	}

	// Write mask=0 (64x64 NONE partition): all bits clear.
	r.SetPartition(0, 1, 0)
	for bsl := 1; bsl <= 4; bsl++ {
		if PartitionAbove(r, 0, bsl) {
			t.Fatalf("bsl=%d: expected false after mask=0", bsl)
		}
	}
}

func TestSetPartitionAboveMask8x8(t *testing.T) {
	// PartitionAboveMasks[1]=15 for 8×8 NONE partition.
	if PartitionAboveMasks[1] != 15 {
		t.Fatalf("PartitionAboveMasks[bsl=1]: got %d, want 15", PartitionAboveMasks[1])
	}
}

func TestSetPartitionAboveMask64x64(t *testing.T) {
	if PartitionAboveMasks[4] != 0 {
		t.Fatalf("PartitionAboveMasks[bsl=4]: got %d, want 0", PartitionAboveMasks[4])
	}
}

// ---- SetBlock ---------------------------------------------------------------

func TestRowSetBlock(t *testing.T) {
	r := NewRow(64) // N4=16
	r.SetBlock(4, 4, TX8x8, true, [2]int{RefLast, RefIntra}, true, false, 1, InterpSwitchable)
	for i := 4; i < 8; i++ {
		if r.TxSize[i] != TX8x8 {
			t.Fatalf("TxSize[%d]: got %d, want TX8x8", i, r.TxSize[i])
		}
		if !r.Skip[i] {
			t.Fatalf("Skip[%d]: want true", i)
		}
		if r.Ref[i] != [2]int{RefLast, RefIntra} {
			t.Fatalf("Ref[%d]: got %v", i, r.Ref[i])
		}
	}
	// Outside range unchanged.
	if r.TxSize[3] != TX32x32 {
		t.Fatal("TxSize[3] should be unchanged (TX32x32)")
	}
}

// ---- SkipCtx ----------------------------------------------------------------

func TestSkipCtxBoth(t *testing.T) {
	if got := SkipCtx(true, true); got != 2 {
		t.Fatalf("both skip: got %d, want 2", got)
	}
}

func TestSkipCtxOne(t *testing.T) {
	if got := SkipCtx(true, false); got != 1 {
		t.Fatalf("one skip: got %d, want 1", got)
	}
}

func TestSkipCtxNone(t *testing.T) {
	if got := SkipCtx(false, false); got != 0 {
		t.Fatalf("none skip: got %d, want 0", got)
	}
}

// ---- TxSizeCtx --------------------------------------------------------------

func TestTxSizeCtxBothLarge(t *testing.T) {
	// Both >= maxTx: would be 2 but clamped to 1.
	if got := TxSizeCtx(true, true, TX32x32, TX32x32, TX32x32); got != 1 {
		t.Fatalf("both large: got %d, want 1", got)
	}
}

func TestTxSizeCtxOneLarge(t *testing.T) {
	if got := TxSizeCtx(true, true, TX32x32, TX8x8, TX32x32); got != 1 {
		t.Fatalf("one large: got %d, want 1", got)
	}
}

func TestTxSizeCtxNoneAvail(t *testing.T) {
	if got := TxSizeCtx(false, false, TX32x32, TX32x32, TX32x32); got != 0 {
		t.Fatalf("none avail: got %d, want 0", got)
	}
}

// ---- IsInterCtx -------------------------------------------------------------

func TestIsInterCtxBothIntra(t *testing.T) {
	if got := IsInterCtx(true, true, false, false); got != 3 {
		t.Fatalf("both intra: got %d, want 3", got)
	}
}

func TestIsInterCtxBothInter(t *testing.T) {
	if got := IsInterCtx(true, true, true, true); got != 0 {
		t.Fatalf("both inter: got %d, want 0", got)
	}
}

func TestIsInterCtxMixed(t *testing.T) {
	got := IsInterCtx(true, true, true, false)
	if got != 1 {
		t.Fatalf("mixed: got %d, want 1", got)
	}
}

func TestIsInterCtxOneIntra(t *testing.T) {
	// Only above available, and it's intra.
	if got := IsInterCtx(true, false, false, false); got != 2 {
		t.Fatalf("one-intra: got %d, want 2", got)
	}
}

func TestIsInterCtxOneInter(t *testing.T) {
	// Only left available, and it's inter.
	if got := IsInterCtx(false, true, false, true); got != 0 {
		t.Fatalf("one-inter: got %d, want 0", got)
	}
}

func TestIsInterCtxNone(t *testing.T) {
	if got := IsInterCtx(false, false, false, false); got != 0 {
		t.Fatalf("none: got %d, want 0", got)
	}
}

// ---- NZCInitCtx -------------------------------------------------------------

func TestNZCInitCtxBoth(t *testing.T) {
	if got := NZCInitCtx(true, true); got != 2 {
		t.Fatalf("both NZC: got %d, want 2", got)
	}
}

func TestNZCInitCtxOne(t *testing.T) {
	if got := NZCInitCtx(true, false); got != 1 {
		t.Fatalf("one NZC: got %d, want 1", got)
	}
}

func TestNZCInitCtxNone(t *testing.T) {
	if got := NZCInitCtx(false, false); got != 0 {
		t.Fatalf("none NZC: got %d, want 0", got)
	}
}

// ---- InterModeCtx -----------------------------------------------------------

func TestInterModeCtxBothIntra(t *testing.T) {
	// Both neighbours are intra → contribute 0 each.
	got := InterModeCtx(true, true,
		[2]int{RefIntra, RefIntra}, [2]int{RefIntra, RefIntra},
		false, false, false, false, RefLast)
	if got != 0 {
		t.Fatalf("both intra: got %d, want 0", got)
	}
}

func TestInterModeCtxDiffRef(t *testing.T) {
	// Both inter but different reference → +4 each = 8, clamped to 6.
	got := InterModeCtx(true, true,
		[2]int{RefGolden, RefIntra}, [2]int{RefAltRef, RefIntra},
		true, true, false, false, RefLast)
	if got != 6 {
		t.Fatalf("diff ref: got %d, want 6", got)
	}
}

func TestInterModeCtxSameRefZeroMV(t *testing.T) {
	// Both inter, same ref, zero MV → +1 each = 2.
	got := InterModeCtx(true, true,
		[2]int{RefLast, RefIntra}, [2]int{RefLast, RefIntra},
		true, true, true, true, RefLast)
	if got != 2 {
		t.Fatalf("zero MV: got %d, want 2", got)
	}
}

func TestInterModeCtxSameRefNonZeroMV(t *testing.T) {
	// Both inter, same ref, non-zero MV → +3 each = 6.
	got := InterModeCtx(true, true,
		[2]int{RefLast, RefIntra}, [2]int{RefLast, RefIntra},
		true, true, false, false, RefLast)
	if got != 6 {
		t.Fatalf("non-zero MV: got %d, want 6", got)
	}
}
