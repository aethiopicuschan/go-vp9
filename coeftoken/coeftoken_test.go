package coeftoken

import (
	"testing"

	"github.com/aethiopicuschan/go-vp9/boolcoder"
	"github.com/aethiopicuschan/go-vp9/compheader"
)

// trivialScan returns identity scan (raster order) for n positions.
func trivialScan(n int) []int16 {
	s := make([]int16, n)
	for i := range s {
		s[i] = int16(i)
	}
	return s
}

// trivialNb returns an all-zero neighbor table (every position's context
// neighbors point at the DC position) of the size DecodeBlock expects.
func trivialNb(n int) []int16 { return make([]int16, 2*(n+1)) }

// encodeToken writes a minimal token bitstream via boolcoder.Writer.
// Uses neutral probabilities (128) for writing so ReadBool(128) is unbiased.
// Pattern: write one non-zero coefficient then EOB.
func writeToken(val int) []byte {
	w := boolcoder.NewWriter()
	// Node 0 = 1 (not EOB)
	w.WriteBool(128, 1)
	// Node 1 = 1 (not zero)
	w.WriteBool(128, 1)
	// Node 2 = 0 → value is 1 (ONE_TOKEN)
	if val == 1 || val == -1 {
		w.WriteBool(128, 0)
	} else {
		w.WriteBool(128, 1)
		// category selection: val=2 → cat0 in readMoreThanOne
		// For cat0: ReadBool(128)==0 then ReadBool(128)==0 → cat=0 (CAT1, base=5)
		// Then 1 extra bit: val = 5 | bit
		// For val=6: bit=1; for val=5: bit=0
		// We'll just write val=5 for simplicity
		// cat0: w(128,0), w(128,0) → cat=0; extra: w(159, bit)
		w.WriteBool(128, 0) // not high half
		w.WriteBool(128, 0) // cat=0
		// cat0 probs = {159}, base=5; write bit=(val-5)&1
		// use val=5 → bit=0
		w.WriteBool(159, 0)
	}
	// Sign: 0=positive
	sign := 0
	if val < 0 {
		sign = 1
	}
	w.WriteBit(sign)
	// EOB for position 1
	w.WriteBool(128, 0)
	return w.Bytes()
}

func TestDecodeBlockAllZero(t *testing.T) {
	// Write immediate EOB at position 0.
	w := boolcoder.NewWriter()
	w.WriteBool(128, 0) // Node 0 = 0 → EOB
	br := boolcoder.NewReader(w.Bytes())

	// Use neutral (all-128) probs so they match what was written above.
	probs := (*CoefProbs)(&compheader.NewProbs().CoefProbs)
	out := DecodeBlock(br, TX4x4, 0, 0, trivialScan(16), trivialNb(16), 0, probs, nil)
	for i, v := range out {
		if v != 0 {
			t.Fatalf("all-zero block: out[%d] = %d, want 0", i, v)
		}
	}
}

func TestDecodeBlockOne(t *testing.T) {
	// Encode ONE token at position 0, then EOB.
	bs := writeToken(1)
	br := boolcoder.NewReader(bs)
	// Use neutral (all-128) probs to match what writeToken wrote with.
	probs := (*CoefProbs)(&compheader.NewProbs().CoefProbs)
	out := DecodeBlock(br, TX4x4, 0, 0, trivialScan(16), trivialNb(16), 1, probs, nil)
	if out[0] != 1 {
		t.Fatalf("one token: out[0] = %d, want 1", out[0])
	}
	for i := 1; i < 16; i++ {
		if out[i] != 0 {
			t.Fatalf("one token: out[%d] = %d, want 0", i, out[i])
		}
	}
}

func TestNCoefs(t *testing.T) {
	cases := [][2]int{{TX4x4, 16}, {TX8x8, 64}, {TX16x16, 256}, {TX32x32, 1024}}
	for _, tc := range cases {
		if got := nCoefs(tc[0]); got != tc[1] {
			t.Fatalf("nCoefs(%d) = %d, want %d", tc[0], got, tc[1])
		}
	}
}

func TestBand4x4(t *testing.T) {
	if band(TX4x4, 0) != 0 {
		t.Fatalf("band(4x4, 0) = %d, want 0", band(TX4x4, 0))
	}
	if band(TX4x4, 15) != 5 {
		t.Fatalf("band(4x4, 15) = %d, want 5", band(TX4x4, 15))
	}
}
