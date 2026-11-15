package segment

import (
	"testing"

	"github.com/aethiopicuschan/go-vp9/boolcoder"
)

// encodeSegID encodes a segment ID into a byte slice using prob=128 for all nodes.
func encodeSegID(id int) []byte {
	w := boolcoder.NewWriter()
	// Balanced tree: 3 bits determine segment 0..7.
	// Bit 0: left half (0..3) = 0, right half (4..7) = 1
	w.WriteBool(128, (id>>2)&1)
	// Bit 1: upper or lower pair within the half
	w.WriteBool(128, (id>>1)&1)
	// Bit 2: which of the two in the pair
	w.WriteBool(128, id&1)
	return w.Bytes()
}

func TestDecodeSegIDAllValues(t *testing.T) {
	probs := [7]uint8{128, 128, 128, 128, 128, 128, 128}
	for want := 0; want < 8; want++ {
		data := encodeSegID(want)
		br := boolcoder.NewReader(data)
		got := DecodeSegID(br, probs)
		if got != want {
			t.Errorf("segment %d: encoded %v, decoded %d", want, data, got)
		}
	}
}

func TestDecodeSegIDZero(t *testing.T) {
	probs := [7]uint8{128, 128, 128, 128, 128, 128, 128}
	// Segment 0: first bit = 0 → left; second bit = 0 → lower pair; third = 0 → seg 0
	data := encodeSegID(0)
	br := boolcoder.NewReader(data)
	got := DecodeSegID(br, probs)
	if got != 0 {
		t.Fatalf("want 0, got %d", got)
	}
}

func TestDecodeSegIDSeven(t *testing.T) {
	probs := [7]uint8{128, 128, 128, 128, 128, 128, 128}
	data := encodeSegID(7)
	br := boolcoder.NewReader(data)
	got := DecodeSegID(br, probs)
	if got != 7 {
		t.Fatalf("want 7, got %d", got)
	}
}

// ---- ParamsFor --------------------------------------------------------------

func TestParamsForNoFeatures(t *testing.T) {
	var enabled [8][4]bool
	var data [8][4]int16
	p := ParamsFor(enabled, data, false, 3)
	if p.QIndexDelta != 0 || p.LFDelta != 0 || p.RefFrame != -1 || p.Skip {
		t.Fatalf("all-disabled: got %+v, want zeros (RefFrame=-1)", p)
	}
}

func TestParamsForAltQ(t *testing.T) {
	var enabled [8][4]bool
	var data [8][4]int16
	enabled[2][FeatAltQ] = true
	data[2][FeatAltQ] = 10
	p := ParamsFor(enabled, data, false, 2)
	if p.QIndexDelta != 10 {
		t.Fatalf("QIndexDelta: got %d, want 10", p.QIndexDelta)
	}
}

func TestParamsForAltQNegative(t *testing.T) {
	var enabled [8][4]bool
	var data [8][4]int16
	enabled[0][FeatAltQ] = true
	data[0][FeatAltQ] = -8
	p := ParamsFor(enabled, data, false, 0)
	if p.QIndexDelta != -8 {
		t.Fatalf("QIndexDelta: got %d, want -8", p.QIndexDelta)
	}
}

func TestParamsForSkip(t *testing.T) {
	var enabled [8][4]bool
	var data [8][4]int16
	enabled[5][FeatSkip] = true
	p := ParamsFor(enabled, data, false, 5)
	if !p.Skip {
		t.Fatal("Skip should be true")
	}
}

func TestParamsForRefFrame(t *testing.T) {
	var enabled [8][4]bool
	var data [8][4]int16
	enabled[1][FeatRefFrame] = true
	data[1][FeatRefFrame] = 2 // GOLDEN
	p := ParamsFor(enabled, data, false, 1)
	if p.RefFrame != 2 {
		t.Fatalf("RefFrame: got %d, want 2", p.RefFrame)
	}
}

func TestParamsForOutOfRange(t *testing.T) {
	var enabled [8][4]bool
	var data [8][4]int16
	p := ParamsFor(enabled, data, false, 8) // out of range
	if p.RefFrame != -1 {
		t.Fatalf("out-of-range id: RefFrame should be -1, got %d", p.RefFrame)
	}
}

// ---- ClampQIndex ------------------------------------------------------------

func TestClampQIndexDelta(t *testing.T) {
	p := Params{QIndexDelta: 5, AbsOrDelta: false}
	if got := ClampQIndex(100, p); got != 105 {
		t.Fatalf("delta: got %d, want 105", got)
	}
}

func TestClampQIndexAbsolute(t *testing.T) {
	p := Params{QIndexDelta: 30, AbsOrDelta: true}
	if got := ClampQIndex(100, p); got != 30 {
		t.Fatalf("absolute: got %d, want 30", got)
	}
}

func TestClampQIndexUnderflow(t *testing.T) {
	p := Params{QIndexDelta: -200, AbsOrDelta: false}
	if got := ClampQIndex(50, p); got != 0 {
		t.Fatalf("underflow: got %d, want 0", got)
	}
}

func TestClampQIndexOverflow(t *testing.T) {
	p := Params{QIndexDelta: 200, AbsOrDelta: false}
	if got := ClampQIndex(100, p); got != 255 {
		t.Fatalf("overflow: got %d, want 255", got)
	}
}
