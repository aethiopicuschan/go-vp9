package intra

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/aethiopicuschan/go-vp9/modeinfo"
)

// TestIntraPredictVsLibvpx compares intra.Predict against libvpx reference DSP
// predictors (vpx_<mode>_predictor_NxN_c). Oracle produced by
// scratchpad/intracmp/harness.c; each line is
//
//	<mode> <bs> C <corner> A <2bs above> L <bs left> O <bs*bs output>
//
// Set INTRA_ORACLE to the oracle file path.
func TestIntraPredictVsLibvpx(t *testing.T) {
	path := os.Getenv("INTRA_ORACLE")
	if path == "" {
		t.Skip("set INTRA_ORACLE to the harness output file")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	modeName := []string{"DC", "V", "H", "D45", "D135", "D117", "D153", "D207", "D63", "TM"}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	line := 0
	byKind := map[string]int{}
	total := 0
	for sc.Scan() {
		line++
		fields := strings.Fields(sc.Text())
		if len(fields) < 4 {
			continue
		}
		mode, _ := strconv.Atoi(fields[0])
		bs, _ := strconv.Atoi(fields[1])
		idx := map[string]int{}
		for i, f := range fields {
			switch f {
			case "C", "A", "L", "O":
				idx[f] = i
			}
		}
		corner := byteAt(fields, idx["C"]+1)
		above := make([]uint8, 2*bs)
		for i := 0; i < 2*bs; i++ {
			above[i] = byteAt(fields, idx["A"]+1+i)
		}
		left := make([]uint8, bs)
		for i := 0; i < bs; i++ {
			left[i] = byteAt(fields, idx["L"]+1+i)
		}
		want := make([]uint8, bs*bs)
		for i := 0; i < bs*bs; i++ {
			want[i] = byteAt(fields, idx["O"]+1+i)
		}

		dst := make([]uint8, bs*bs)
		Predict(modeinfo.IntraMode(mode), dst, bs, bs, above, left, corner)

		diffs := 0
		for i := range dst {
			if dst[i] != want[i] {
				diffs++
			}
		}
		total++
		if diffs > 0 {
			kind := modeName[mode] + " " + strconv.Itoa(bs)
			byKind[kind]++
			if byKind[kind] <= 1 {
				t.Errorf("line %d %s: %d/%d differ\n want=%v\n  got=%v", line, kind, diffs, bs*bs, want, dst)
			}
		}
	}
	if len(byKind) > 0 {
		t.Errorf("mismatching kinds: %v (of %d tests)", byKind, total)
	} else {
		t.Logf("all %d intra-predict tests match libvpx", total)
	}
}

func byteAt(fields []string, i int) uint8 {
	v, _ := strconv.Atoi(fields[i])
	return uint8(v)
}
