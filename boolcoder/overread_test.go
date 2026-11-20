package boolcoder

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/hex"
	"math/rand"
	"strings"
	"testing"
)

// zeroReader returns a NewReader over n zero bytes together with a function
// that consumes exactly one bit. Over an all-zero buffer every ReadBool(128)
// decodes 0; the first one only narrows the range 255->128 (no shift) and each
// later one shifts the register by exactly one bit, so the stream position can
// be stepped across the limit one bit at a time.
func zeroReader(t *testing.T, n int) (*Reader, func()) {
	t.Helper()
	r := NewReader(make([]byte, n))
	if r.ReadBool(128) != 0 || r.BitPos() != 0 {
		t.Fatalf("priming read: BitPos=%d, want 0", r.BitPos())
	}
	return r, func() {
		t.Helper()
		before := r.BitPos()
		if r.ReadBool(128) != 0 || r.BitPos() != before+1 {
			t.Fatalf("step did not consume exactly one bit (BitPos %d -> %d)", before, r.BitPos())
		}
	}
}

// TestOverreadBoundary pins libvpx's criterion: with N = 8*len(buf) and S =
// bits consumed, S <= N-8 is fine and S == N-7 (one bit more) is an error. The
// last 8 bits of a partition may be looked at but never consumed.
func TestOverreadBoundary(t *testing.T) {
	// Sizes on both sides of the 8-byte register, and well past it.
	for _, n := range []int{2, 3, 7, 8, 9, 15, 16, 17, 64, 1000} {
		r, step := zeroReader(t, n)
		limit := n*8 - 8
		for r.BitPos() < limit {
			if r.Overread() {
				t.Fatalf("len %d: Overread at BitPos %d, before the limit %d", n, r.BitPos(), limit)
			}
			step()
		}
		if r.Overread() {
			t.Fatalf("len %d: Overread exactly at the limit (BitPos %d == 8*len-8)", n, r.BitPos())
		}
		step()
		if !r.Overread() {
			t.Fatalf("len %d: no Overread one bit past the limit (BitPos %d)", n, r.BitPos())
		}
		// Sticky from here on, through the end of the buffer and beyond.
		for i := 0; i < 100; i++ {
			step()
			if !r.Overread() {
				t.Fatalf("len %d: Overread cleared at BitPos %d", n, r.BitPos())
			}
		}
	}
}

// TestOverreadTinyPartitions: libvpx flags an empty partition immediately
// (0 > -8; the VP9 decoder also rejects zero-sized tiles up front), while a
// one-byte partition survives its marker bit (a 0 marker only narrows the
// range, it shifts nothing) and fails on the first bit actually consumed.
func TestOverreadTinyPartitions(t *testing.T) {
	if r := NewReaderVP9(nil); !r.Overread() {
		t.Error("empty partition: want Overread")
	}
	r := NewReaderVP9([]byte{0})
	if r.Overread() {
		t.Error("1-byte partition: Overread right after the marker bit")
	}
	r.ReadBool(128) // 128 -> 64: shifts one bit
	if !r.Overread() {
		t.Errorf("1-byte partition: no Overread after consuming a bit (BitPos %d)", r.BitPos())
	}
	if r := NewReaderVP9([]byte{0, 0}); r.Overread() {
		t.Error("2-byte partition: Overread right after the marker bit")
	}
}

func TestMarkerError(t *testing.T) {
	if NewReaderVP9([]byte{0x00, 0, 0, 0}).MarkerError() {
		t.Error("marker bit 0 reported as error")
	}
	if !NewReaderVP9([]byte{0x80, 0, 0, 0}).MarkerError() {
		t.Error("marker bit 1 not reported")
	}
	if NewReader([]byte{0x80, 0, 0, 0}).MarkerError() {
		t.Error("NewReader (no marker) reported a marker error")
	}
}

// vpxReader is a literal port of libvpx's vpx_reader (vpx_dsp/bitreader.[ch],
// 64-bit BD_VALUE, no decryption), kept deliberately independent of Reader's
// bookkeeping: it has libvpx's lazy fill, its `count` (= real bits in value
// minus 8) and the LOTS_OF_BITS end-of-buffer sentinel.
type vpxReader struct {
	value uint64
	rng   uint32
	count int
	buf   []byte
	pos   int
}

const (
	vpxBDValueSize = 64
	vpxLotsOfBits  = 0x40000000
)

func newVpxReader(buf []byte) (*vpxReader, int) {
	r := &vpxReader{buf: buf, count: -8, rng: 255}
	r.fill()
	return r, r.read(128)
}

func (r *vpxReader) fill() {
	bitsLeft := (len(r.buf) - r.pos) * 8
	shift := vpxBDValueSize - 8 - (r.count + 8)
	if bitsLeft > vpxBDValueSize {
		bits := (shift &^ 7) + 8
		var be uint64
		for i := 0; i < 8; i++ {
			be = be<<8 | uint64(r.buf[r.pos+i])
		}
		nv := be >> uint(vpxBDValueSize-bits)
		r.count += bits
		r.pos += bits >> 3
		r.value |= nv << uint(shift&7)
		return
	}
	bitsOver := shift + 8 - bitsLeft
	loopEnd := 0
	if bitsOver >= 0 {
		r.count += vpxLotsOfBits
		loopEnd = bitsOver
	}
	if bitsOver < 0 || bitsLeft != 0 {
		for shift >= loopEnd {
			r.count += 8
			r.value |= uint64(r.buf[r.pos]) << uint(shift)
			r.pos++
			shift -= 8
		}
	}
}

func (r *vpxReader) read(prob int) int {
	split := (r.rng*uint32(prob) + uint32(256-prob)) >> 8
	if r.count < 0 {
		r.fill()
	}
	bigsplit := uint64(split) << (vpxBDValueSize - 8)
	bit := 0
	rng := split
	if r.value >= bigsplit {
		rng = r.rng - split
		r.value -= bigsplit
		bit = 1
	}
	shift := norm8[rng]
	r.rng = rng << shift
	r.value <<= shift
	r.count -= int(shift)
	return bit
}

func (r *vpxReader) hasError() bool {
	return r.count > vpxBDValueSize && r.count < vpxLotsOfBits
}

//go:embed testdata/vpx_reader_golden.txt
var vpxReaderGolden []byte

// TestOverreadMatchesLibvpxGolden replays vectors recorded from the real
// libvpx vpx_reader: every decoded bit, the init marker status and
// vpx_reader_has_error after init and after every read must match both Reader
// and the in-test port (which the randomized test below then relies on).
func TestOverreadMatchesLibvpxGolden(t *testing.T) {
	sc := bufio.NewScanner(bytes.NewReader(vpxReaderGolden))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	cases, flips := 0, 0
	for line := 1; sc.Scan(); line++ {
		if strings.HasPrefix(sc.Text(), "#") || sc.Text() == "" {
			continue
		}
		f := strings.Fields(sc.Text())
		if len(f) != 5 {
			t.Fatalf("line %d: %d fields", line, len(f))
		}
		var buf []byte
		if f[0] != "-" {
			var err error
			if buf, err = hex.DecodeString(f[0]); err != nil {
				t.Fatalf("line %d: %v", line, err)
			}
		}
		probs, err := hex.DecodeString(f[2])
		if err != nil || len(f[3]) != len(probs) || len(f[4]) != len(probs)+1 {
			t.Fatalf("line %d: malformed vector", line)
		}
		wantMarker := f[1] == "1"
		r := NewReaderVP9(buf)
		m, mMarker := newVpxReader(buf)
		if r.MarkerError() != wantMarker || (mMarker != 0) != wantMarker {
			t.Fatalf("line %d: marker: Reader=%v port=%d libvpx=%v", line, r.MarkerError(), mMarker, wantMarker)
		}
		check := func(i int) {
			want := f[4][i] == '1'
			if r.Overread() != want || m.hasError() != want {
				t.Fatalf("line %d (len %d): after %d reads: Overread=%v port=%v libvpx has_error=%v (BitPos %d)",
					line, len(buf), i, r.Overread(), m.hasError(), want, r.BitPos())
			}
		}
		check(0)
		for i, p := range probs {
			want := int(f[3][i] - '0')
			if got, gotM := r.ReadBool(p), m.read(int(p)); got != want || gotM != want {
				t.Fatalf("line %d: read %d: Reader=%d port=%d libvpx=%d", line, i, got, gotM, want)
			}
			check(i + 1)
		}
		if f[4][0] == '0' && strings.Contains(f[4], "1") {
			flips++
		}
		cases++
	}
	if cases < 100 || flips < 50 {
		t.Fatalf("golden file too small: %d cases, %d crossing the limit", cases, flips)
	}
}

// TestOverreadMatchesLibvpxPort drives Reader and the libvpx port with the
// same random buffers/probabilities far past the end of the data and requires
// identical bits and an identical error flag after every single read.
func TestOverreadMatchesLibvpxPort(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for c := 0; c < 3000; c++ {
		n := rng.Intn(40)
		if c%10 == 0 {
			n = 200 + rng.Intn(2000)
		}
		buf := make([]byte, n)
		switch c % 3 {
		case 1:
			rng.Read(buf)
		case 2: // marker bit clear
			rng.Read(buf)
			if n > 0 {
				buf[0] &= 0x7f
			}
		}
		r := NewReaderVP9(buf)
		m, marker := newVpxReader(buf)
		if r.MarkerError() != (marker != 0) || r.Overread() != m.hasError() {
			t.Fatalf("case %d (len %d): init mismatch", c, n)
		}
		for i := 0; i < 8*n+80; i++ {
			p := 1 + rng.Intn(255)
			switch rng.Intn(6) {
			case 0:
				p = 1
			case 1:
				p = 255
			}
			if a, b := r.ReadBool(uint8(p)), m.read(p); a != b {
				t.Fatalf("case %d (len %d): read %d: Reader=%d port=%d", c, n, i, a, b)
			}
			if r.Overread() != m.hasError() {
				t.Fatalf("case %d (len %d): read %d: Overread=%v, libvpx has_error=%v (BitPos %d of %d bits)",
					c, n, i, r.Overread(), m.hasError(), r.BitPos(), 8*n)
			}
		}
	}
}
