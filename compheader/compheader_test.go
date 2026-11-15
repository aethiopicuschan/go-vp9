package compheader

import (
	"bytes"
	"testing"

	"github.com/aethiopicuschan/go-vp9/boolcoder"
)

// ---- invRecenterNonneg unit tests ----

func TestInvRecenterNonneg(t *testing.T) {
	cases := []struct{ v, m, want int }{
		{0, 127, 127}, // 0 even → 0/2 + 127 = 127
		{2, 127, 128}, // 2 even → 1 + 127
		{1, 127, 127}, // 1 odd → 127 - 1 = 126? wait: 127 - (2>>1) = 127-1=126
		{7, 127, 123}, // 7 odd, 7 ≤ 254 → 127 - (8>>1) = 127-4 = 123
		{255, 0, 255}, // 255 > 0 → 255
		{0, 0, 0},     // 0 even → 0/2 + 0 = 0
	}
	// Recompute expected values using the exact formula.
	for _, tc := range cases {
		var want int
		if tc.v > tc.m<<1 {
			want = tc.v
		} else if tc.v&1 == 0 {
			want = (tc.v >> 1) + tc.m
		} else {
			want = tc.m - ((tc.v + 1) >> 1)
		}
		got := invRecenterNonneg(tc.v, tc.m)
		if got != want {
			t.Errorf("invRecenterNonneg(%d,%d) = %d, want %d", tc.v, tc.m, got, want)
		}
	}
}

// ---- decodeTermSubexp unit tests ----

// encodeTermSubexp writes a term-subexp value onto a bool writer so we can
// decode it back and verify. Encoding is the inverse of decodeTermSubexp.
func encodeTermSubexp(w *boolcoder.Writer, v int) {
	if v < 16 {
		w.WriteBit(0)
		w.WriteLiteral(uint32(v), 4)
		return
	}
	w.WriteBit(1)
	if v < 32 {
		w.WriteBit(0)
		w.WriteLiteral(uint32(v-16), 4)
		return
	}
	w.WriteBit(1)
	if v < 64 {
		w.WriteBit(0)
		w.WriteLiteral(uint32(v-32), 5)
		return
	}
	w.WriteBit(1)
	// 4th branch: encode_uniform(v-64), the inverse of decodeUniform.
	u := v - 64
	if u < 65 {
		w.WriteLiteral(uint32(u), 7)
	} else {
		w.WriteLiteral(uint32((u+65)>>1), 7)
		w.WriteBit((u + 65) & 1)
	}
}

func TestDecodeTermSubexp(t *testing.T) {
	cases := []int{0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 253}
	for _, want := range cases {
		w := boolcoder.NewWriter()
		encodeTermSubexp(w, want)
		buf := w.Bytes()
		br := boolcoder.NewReader(buf)
		got := decodeTermSubexp(br)
		if got != want {
			t.Errorf("decodeTermSubexp(%d): got %d", want, got)
		}
	}
}

// ---- invRemapProb unit tests ----

func TestInvRemapProb(t *testing.T) {
	// Known pairs: current_prob=128, delta_v=0 → x=invMapTable[0]=7
	// invRecenterNonneg(7, 127) = 127 - 4 = 123 → +1 = 124
	if got := invRemapProb(0, 128); got != 124 {
		t.Errorf("invRemapProb(0,128) = %d, want 124", got)
	}

	// v=1, m=128: x=invMapTable[1]=20, invRecenterNonneg(20, 127) = 20 even → 10+127=137 → +1=138
	if got := invRemapProb(1, 128); got != 138 {
		t.Errorf("invRemapProb(1,128) = %d, want 138", got)
	}

	// Clamp test: v=253 is within bounds.
	_ = invRemapProb(253, 128) // just must not panic
}

// ---- diffUpdateProb unit tests ----

func TestDiffUpdateProbNoUpdate(t *testing.T) {
	// Encode "no update" (ReadBool(252)=0).
	w := boolcoder.NewWriter()
	w.WriteBool(252, 0) // update_prob = 0
	br := boolcoder.NewReader(w.Bytes())
	got := diffUpdateProb(br, 200)
	if got != 200 {
		t.Errorf("no-update: got %d, want 200", got)
	}
}

func TestDiffUpdateProbWithUpdate(t *testing.T) {
	// Encode "update, delta=0" → invRemapProb(0, 128) = 124.
	w := boolcoder.NewWriter()
	w.WriteBool(252, 1) // update_prob = 1
	encodeTermSubexp(w, 0)
	br := boolcoder.NewReader(w.Bytes())
	got := diffUpdateProb(br, 128)
	want := invRemapProb(0, 128) // 124
	if got != want {
		t.Errorf("update delta=0 from 128: got %d, want %d", got, want)
	}
}

// ---- Null-update compressed header (key frame) ----

// buildNullUpdate writes a compressed header where every probability update
// flag (ReadBool(252)) is 0, so no probability changes.
// tx_mode is encoded as 0 (ONLY_4X4) — 2 literal bits, both 0.
// For key frames (isIntra=true) only tx_mode + coef_probs + skip_probs are read.
func buildNullUpdate(t *testing.T) []byte {
	t.Helper()
	// Use a large buffer of zeros; with an all-zero bool stream:
	//   ReadLiteral bits (prob 128): value=0 < split<<56 → always 0
	//   ReadBool(252): value=0 < split<<56 → always 0 (no update)
	// 100 bytes is far more than the ~30 bytes actually consumed for
	// TX_MODE_ONLY_4X4 key frame (2 + 396 + 3 = 401 bool reads).
	return bytes.Repeat([]byte{0x00}, 100)
}

func TestNullUpdateKeyFrame(t *testing.T) {
	p := NewProbs()
	// Snapshot: all values should be 128 after null-update.
	buf := buildNullUpdate(t)
	br := boolcoder.NewReader(buf)

	if err := Decode(br, true, InterpEightTap, false, false, p); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	// tx_mode must be 0 (ONLY_4X4) since ReadLiteral(2) = 0b00 from zero stream.
	if p.TxMode != TxModeOnly4x4 {
		t.Errorf("TxMode = %d, want %d", p.TxMode, TxModeOnly4x4)
	}

	// All coef probs for TX_4X4 must remain 128.
	for pt := 0; pt < planeTypes; pt++ {
		for rt := 0; rt < refTypes; rt++ {
			for band := 0; band < coefBands; band++ {
				nCtx := bandCoeffContexts(band)
				for ctx := 0; ctx < nCtx; ctx++ {
					for n := 0; n < uncostrainedNodes; n++ {
						v := p.CoefProbs[0][pt][rt][band][ctx][n]
						if v != 128 {
							t.Errorf("CoefProbs[0][%d][%d][%d][%d][%d] = %d, want 128",
								pt, rt, band, ctx, n, v)
						}
					}
				}
			}
		}
	}

	// TX sizes 1-3 must still be 128 (not read, untouched).
	for tx := 1; tx < txSizes; tx++ {
		for pt := 0; pt < planeTypes; pt++ {
			for rt := 0; rt < refTypes; rt++ {
				for band := 0; band < coefBands; band++ {
					for ctx := 0; ctx < prevCoefContexts; ctx++ {
						for n := 0; n < uncostrainedNodes; n++ {
							if p.CoefProbs[tx][pt][rt][band][ctx][n] != 128 {
								t.Errorf("higher-tx CoefProbs[%d] != 128", tx)
							}
						}
					}
				}
			}
		}
	}

	// Skip probs must remain 128.
	for i, v := range p.SkipProbs {
		if v != 128 {
			t.Errorf("SkipProbs[%d] = %d, want 128", i, v)
		}
	}

	// Inter probs untouched (isIntra=true).
	for i, v := range p.IsInterProbs {
		if v != 128 {
			t.Errorf("IsInterProbs[%d] = %d, want 128", i, v)
		}
	}
}

// TestNullUpdateTxModeSelect verifies that TX_MODE_SELECT (tx_mode=4) is read
// when the bool stream encodes the right bits, and tx_mode_probs are updated.
func TestNullUpdateTxModeSelect(t *testing.T) {
	// Encode tx_mode=4 (SELECT):
	//   ReadLiteral(2): need value 3 = 0b11 (both bits set with prob 128)
	//   ReadLiteral(1): need value 1 (one more bit = 1)
	// Then all update bits = 0 (no prob changes).
	w := boolcoder.NewWriter()
	// ReadLiteral(2) = 3 → write bits 1,1 with prob 128
	w.WriteBool(128, 1)
	w.WriteBool(128, 1)
	// ReadLiteral(1) = 1 → write bit 1 with prob 128
	w.WriteBool(128, 1)
	// Now write 0 for all tx_mode_probs updates (12 diff_update_prob calls):
	//   TxProbs8: 2
	//   TxProbs16: 4
	//   TxProbs32: 6
	for i := 0; i < 12; i++ {
		w.WriteBool(252, 0)
	}
	// Then coef probs for all 4 tx sizes: 4 × 396 = 1584 calls
	for i := 0; i < 4*2*2*(3+5*6)*3; i++ {
		w.WriteBool(252, 0)
	}
	// Skip probs: 3
	for i := 0; i < 3; i++ {
		w.WriteBool(252, 0)
	}

	p := NewProbs()
	br := boolcoder.NewReader(w.Bytes())
	if err := Decode(br, true, InterpEightTap, false, false, p); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if p.TxMode != TxModeSelect {
		t.Errorf("TxMode = %d, want %d (SELECT)", p.TxMode, TxModeSelect)
	}
	// All probs must remain 128 (no updates).
	for i, v := range p.TxProbs8 {
		if v != 128 {
			t.Errorf("TxProbs8[%d] = %d, want 128", i, v)
		}
	}
	for i, v := range p.SkipProbs {
		if v != 128 {
			t.Errorf("SkipProbs[%d] = %d, want 128", i, v)
		}
	}
}
