package transform

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"testing"
)

// TestInvTransformVsLibvpx compares go-vp9's InvTransform2D against libvpx
// reference inverse transforms (vp9_iht* / vpx_idct32x32). The oracle file is
// produced by scratchpad/txcmp/harness.c: each line is
//
//	<bs> <txtype> I <bs² input coeffs> O <bs² clamped-uint8 outputs>
//
// where the libvpx output is the transform added to a dest pre-filled with 128.
// Set TX_ORACLE to the oracle file path.
func TestInvTransformVsLibvpx(t *testing.T) {
	path := os.Getenv("TX_ORACLE")
	if path == "" {
		t.Skip("set TX_ORACLE to the harness output file")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	clip := func(v int) int {
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return v
	}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	line := 0
	mismatch := 0
	byKind := map[string]int{}
	for sc.Scan() {
		line++
		fields := strings.Fields(sc.Text())
		if len(fields) < 4 {
			continue
		}
		bs, _ := strconv.Atoi(fields[0])
		tt, _ := strconv.Atoi(fields[1])
		n := bs * bs
		// Parse I ... O ...
		iIdx, oIdx := -1, -1
		for i, f := range fields {
			if f == "I" {
				iIdx = i
			} else if f == "O" {
				oIdx = i
			}
		}
		if iIdx < 0 || oIdx < 0 || oIdx-iIdx-1 != n || len(fields)-oIdx-1 != n {
			t.Fatalf("line %d: malformed (bs=%d n=%d)", line, bs, n)
		}
		in := make([]int16, n)
		for i := 0; i < n; i++ {
			v, _ := strconv.Atoi(fields[iIdx+1+i])
			in[i] = int16(v)
		}
		want := make([]int, n)
		for i := 0; i < n; i++ {
			want[i], _ = strconv.Atoi(fields[oIdx+1+i])
		}

		// Run go-vp9 transform (in-place residual), then clamp(128+residual).
		InvTransform2D(in, bs, TxType(tt))

		diffs := 0
		for i := 0; i < n; i++ {
			got := clip(128 + int(in[i]))
			if got != want[i] {
				diffs++
			}
		}
		if diffs > 0 {
			mismatch++
			kind := strconv.Itoa(bs) + "x" + strconv.Itoa(bs) + " tt=" + strconv.Itoa(tt)
			byKind[kind]++
			if byKind[kind] <= 1 && n <= 64 {
				got := make([]int, n)
				for i := 0; i < n; i++ {
					got[i] = clip(128 + int(in[i]))
				}
				t.Errorf("line %d %s: %d/%d differ\n  want=%v\n   got=%v", line, kind, diffs, n, want, got)
			}
		}
	}
	if mismatch > 0 {
		t.Logf("mismatching tests by kind: %v", byKind)
		t.Errorf("%d/%d tests mismatch", mismatch, line)
	} else {
		t.Logf("all %d transform tests match libvpx", line)
	}
}
