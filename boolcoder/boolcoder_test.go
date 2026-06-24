package boolcoder

import (
	"math/rand"
	"testing"
)

// TestRoundtripBits encodes a long pseudo-random sequence of (prob,bit) pairs and
// checks the decoder reproduces every bit — this validates the arithmetic exactly.
func TestRoundtripBits(t *testing.T) {
	for seed := int64(0); seed < 50; seed++ {
		rng := rand.New(rand.NewSource(seed))
		n := 1 + rng.Intn(4000)
		probs := make([]uint8, n)
		bits := make([]int, n)
		w := NewWriter()
		for i := range bits {
			p := uint8(1 + rng.Intn(255)) // 1..255
			b := rng.Intn(2)
			probs[i], bits[i] = p, b
			w.WriteBool(p, b)
		}
		r := NewReader(w.Bytes())
		for i := range bits {
			if got := r.ReadBool(probs[i]); got != bits[i] {
				t.Fatalf("seed %d: bit %d/%d: got %d want %d (prob %d)", seed, i, n, got, bits[i], probs[i])
			}
		}
	}
}

// TestRoundtripLiteral checks multi-bit literals survive a roundtrip.
func TestRoundtripLiteral(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	const n = 2000
	vals := make([]uint32, n)
	nbits := make([]int, n)
	w := NewWriter()
	for i := range vals {
		nb := 1 + rng.Intn(24)
		v := rng.Uint32() & ((1 << nb) - 1)
		vals[i], nbits[i] = v, nb
		w.WriteLiteral(v, nb)
	}
	r := NewReader(w.Bytes())
	for i := range vals {
		if got := r.ReadLiteral(nbits[i]); got != vals[i] {
			t.Fatalf("literal %d: got %d want %d (%d bits)", i, got, vals[i], nbits[i])
		}
	}
}

// TestExtremeProbs exercises the smallest/largest probabilities (tightest splits).
func TestExtremeProbs(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const n = 5000
	bits := make([]int, n)
	probs := make([]uint8, n)
	w := NewWriter()
	for i := range bits {
		p := uint8(1)
		switch rng.Intn(3) {
		case 1:
			p = 255
		case 2:
			p = 128
		}
		b := rng.Intn(2)
		probs[i], bits[i] = p, b
		w.WriteBool(p, b)
	}
	r := NewReader(w.Bytes())
	for i := range bits {
		if got := r.ReadBool(probs[i]); got != bits[i] {
			t.Fatalf("bit %d: got %d want %d (prob %d)", i, got, bits[i], probs[i])
		}
	}
}
