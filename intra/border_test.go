package intra

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/aethiopicuschan/go-vp9/modeinfo"
)

// TestPredictBlockVsLibvpx compares PredictBlock against libvpx
// build_intra_predictors (faster path) + DSP predictor. Oracle produced by
// scratchpad/borderpred/harness.c. Line:
//
//	<mode> <bs> <ht> <hl> <hr> C <corner> A <2bs above> L <bs left> O <bs*bs out>
//
// Set BORDER_ORACLE to the oracle file path.
func TestPredictBlockVsLibvpx(t *testing.T) {
	path := os.Getenv("BORDER_ORACLE")
	if path == "" {
		t.Skip("set BORDER_ORACLE to the harness output file")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	modeName := []string{"DC", "V", "H", "D45", "D135", "D117", "D153", "D207", "D63", "TM"}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	line, total := 0, 0
	byKind := map[string]int{}
	for sc.Scan() {
		line++
		fl := strings.Fields(sc.Text())
		if len(fl) < 6 {
			continue
		}
		mode, _ := strconv.Atoi(fl[0])
		bs, _ := strconv.Atoi(fl[1])
		ht := fl[2] == "1"
		hl := fl[3] == "1"
		hr := fl[4] == "1"
		idx := map[string]int{}
		for i, s := range fl {
			switch s {
			case "C", "A", "L", "O":
				idx[s] = i
			}
		}
		corner := bval(fl, idx["C"]+1)
		above := make([]uint8, 2*bs)
		for i := 0; i < 2*bs; i++ {
			above[i] = bval(fl, idx["A"]+1+i)
		}
		left := make([]uint8, bs)
		for i := 0; i < bs; i++ {
			left[i] = bval(fl, idx["L"]+1+i)
		}
		want := make([]uint8, bs*bs)
		for i := 0; i < bs*bs; i++ {
			want[i] = bval(fl, idx["O"]+1+i)
		}

		dst := make([]uint8, bs*bs)
		PredictBlock(modeinfo.IntraMode(mode), dst, bs, bs, above, left, corner, ht, hl, hr)

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
				t.Errorf("line %d %s ht=%v hl=%v hr=%v: %d/%d differ\n want=%v\n  got=%v",
					line, kind, ht, hl, hr, diffs, bs*bs, want, dst)
			}
		}
	}
	if len(byKind) > 0 {
		t.Errorf("mismatching kinds: %v (of %d)", byKind, total)
	} else {
		t.Logf("all %d border-predict tests match libvpx", total)
	}
}

func bval(fl []string, i int) uint8 {
	v, _ := strconv.Atoi(fl[i])
	return uint8(v)
}
