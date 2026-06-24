package mv

import (
	"testing"

	"github.com/mgvs/go-vp9/boolcoder"
	"github.com/mgvs/go-vp9/compheader"
)

// newProbs128 returns a Probs with all values set to 128 (prob=0.5).
func newProbs128() *compheader.Probs {
	return compheader.NewProbs()
}

// bitsToBytes encodes a sequence of bool bits into a byte slice
// suitable for feeding to a boolcoder.Reader (big-endian, MSB first,
// using prob=128 so each 0-bit = false and 1-bit = true in the bool stream).
//
// For prob=128 (0x80) the boolcoder outputs the raw bit: a 1-bit makes
// the reader return 1, a 0-bit makes it return 0.
func boolEncode(bits []int) []byte {
	enc := boolcoder.NewWriter()
	for _, b := range bits {
		enc.WriteBool(128, b)
	}
	return enc.Bytes()
}

// ---- joint type tests -------------------------------------------------------

// Encode ZERO joint (first bit = 0) and verify MV = {0,0}.
func TestReadMVJointZero(t *testing.T) {
	buf := boolEncode([]int{0}) // first bit 0 → ZERO
	br := boolcoder.NewReader(buf)
	probs := newProbs128()
	mv := ReadMV(br, probs, nil, true)
	if mv.Row != 0 || mv.Col != 0 {
		t.Fatalf("ZERO joint: got {%d,%d}, want {0,0}", mv.Row, mv.Col)
	}
}

// Encode HNZVZ (bits 1,0): row non-zero, col zero.
// Then encode the row component (sign=0, CLASS_0 d=0, fr=0, hp=0 when prob=128).
func TestReadMVJointHNZVZ(t *testing.T) {
	// Joint: 1,0 → HNZVZ
	// Row component: sign=0, class_tree[0]=0 (CLASS_0), class0_bit=0, fp[0]=0 (→ReadBool for fp[1])=0 → fr=0, hp=0 (prob=128, read 0)
	// Note: allowHP=false → hp=1 (forced)
	buf := boolEncode([]int{1, 0, // joint=HNZVZ
		0,    // sign = 0 (positive)
		0,    // class_tree[0]=0 → CLASS_0
		0,    // class0_bit = 0 (d=0)
		0, 0, // fp: probs[0]=0 → read probs[1]=0 → fr=0
	})
	br := boolcoder.NewReader(buf)
	probs := newProbs128()
	mv := ReadMV(br, probs, nil, false) // allowHP=false → hp=1
	if mv.Col != 0 {
		t.Fatalf("HNZVZ: col should be 0, got %d", mv.Col)
	}
	// mag = base(8) + (d<<3|fr<<1|hp) = 8 + (0|0|1) = 9
	if mv.Row != 9 {
		t.Fatalf("HNZVZ: row = %d, want 9", mv.Row)
	}
}

// ---- sign test --------------------------------------------------------------

func TestReadMVNegative(t *testing.T) {
	// Joint: HNVNZ (1,1,1)
	// Row: sign=1 (negative), CLASS_0, d=0, fr=0, allowHP=false (hp=1) → mag=9 → -9
	// Col: sign=0, CLASS_0, d=0, fr=0, allowHP=false (hp=1) → mag=9 → +9
	buf := boolEncode([]int{
		1, 1, 1, // joint=HNVNZ
		1,    // row sign = 1
		0,    // row CLASS_0
		0,    // row d=0
		0, 0, // row fr=0
		0,    // col sign = 0
		0,    // col CLASS_0
		0,    // col d=0
		0, 0, // col fr=0
	})
	br := boolcoder.NewReader(buf)
	probs := newProbs128()
	mv := ReadMV(br, probs, nil, false)
	if mv.Row != -9 {
		t.Fatalf("negative row: got %d, want -9", mv.Row)
	}
	if mv.Col != 9 {
		t.Fatalf("positive col: got %d, want 9", mv.Col)
	}
}

// ---- CLASS_0 d=1 test -------------------------------------------------------

func TestReadMVClass0D1(t *testing.T) {
	// HZVNZ joint (1,1,0) → col only
	// Col: sign=0, CLASS_0, d=1, fr=0, allowHP=false → hp=1
	// mag = 8 + (1<<3|0|1) = 8 + 9 = 17
	buf := boolEncode([]int{
		1, 1, 0, // joint=HZVNZ
		0,    // col sign=0
		0,    // CLASS_0
		1,    // d=1
		0, 0, // fr=0
	})
	br := boolcoder.NewReader(buf)
	probs := newProbs128()
	mv := ReadMV(br, probs, nil, false)
	if mv.Row != 0 {
		t.Fatalf("HZVNZ: row should be 0, got %d", mv.Row)
	}
	if mv.Col != 17 {
		t.Fatalf("d=1 col: got %d, want 17", mv.Col)
	}
}

// ---- CLASS_1 test -----------------------------------------------------------

func TestReadMVClass1(t *testing.T) {
	// HNZVZ joint → row only
	// Row: sign=0, CLASS_1 (class_tree: 1,0), d[0]=0 → d=0
	// fr=0, allowHP=false → hp=1
	// base = 2<<3 = 16; mag = 16 + (0<<3|0|1) = 17
	buf := boolEncode([]int{
		1, 0, // joint=HNZVZ
		0,    // sign=0
		1, 0, // class_tree: 1→not CLASS_0, 0→CLASS_1
		0,    // bits[0]=0 → d=0
		0, 0, // fr=0
	})
	br := boolcoder.NewReader(buf)
	probs := newProbs128()
	mv := ReadMV(br, probs, nil, false)
	if mv.Row != 17 {
		t.Fatalf("CLASS_1: row=%d, want 17", mv.Row)
	}
}

// ---- allowHP test -----------------------------------------------------------

func TestReadMVAllowHP(t *testing.T) {
	// With allowHP=true the hp bit IS read.
	// HNZVZ → row only; sign=0, CLASS_0, d=0, fr=0, hp=0
	// mag = 8 + (0|0|0) = 8
	buf := boolEncode([]int{
		1, 0, // joint=HNZVZ
		0,       // sign=0
		0,       // CLASS_0
		0,       // d=0
		0, 0,    // fr=0
		0,       // hp=0
	})
	br := boolcoder.NewReader(buf)
	probs := newProbs128()
	mv := ReadMV(br, probs, nil, true)
	if mv.Row != 8 {
		t.Fatalf("allowHP hp=0: row=%d, want 8", mv.Row)
	}
}
