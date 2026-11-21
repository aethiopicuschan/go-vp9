package boolcoder

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/hex"
	"strconv"
	"strings"
	"testing"
)

//go:embed testdata/vpx_find_end_golden.txt
var vpxFindEndGolden []byte

// TestFindEndMatchesLibvpxGolden replays vectors recorded from the real libvpx
// vpx_reader: FindEnd must equal vpx_reader_find_end after init and after
// every single read, including the reads that run past the end of the buffer.
func TestFindEndMatchesLibvpxGolden(t *testing.T) {
	sc := bufio.NewScanner(bytes.NewReader(vpxFindEndGolden))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	cases, early := 0, 0
	for line := 1; sc.Scan(); line++ {
		if strings.HasPrefix(sc.Text(), "#") || sc.Text() == "" {
			continue
		}
		f := strings.Fields(sc.Text())
		if len(f) != 3 {
			t.Fatalf("line %d: %d fields", line, len(f))
		}
		var buf []byte
		if f[0] != "-" {
			var err error
			if buf, err = hex.DecodeString(f[0]); err != nil {
				t.Fatalf("line %d: %v", line, err)
			}
		}
		probs, err := hex.DecodeString(f[1])
		ends := strings.Split(f[2], ",")
		if err != nil || len(ends) != len(probs)+1 {
			t.Fatalf("line %d: malformed vector", line)
		}
		r := NewReaderVP9(buf)
		check := func(i int) {
			want, err := strconv.Atoi(ends[i])
			if err != nil {
				t.Fatalf("line %d: %v", line, err)
			}
			if got := r.FindEnd(); got != want {
				t.Fatalf("line %d (len %d): after %d reads: FindEnd=%d, libvpx vpx_reader_find_end=%d (BitPos %d)",
					line, len(buf), i, got, want, r.BitPos())
			}
			if want < len(buf) {
				early++
			}
		}
		check(0)
		for i, p := range probs {
			r.ReadBool(p)
			check(i + 1)
		}
		cases++
	}
	if cases < 100 || early < 500 {
		t.Fatalf("golden file too small: %d cases, %d positions before the end of the buffer", cases, early)
	}
}
