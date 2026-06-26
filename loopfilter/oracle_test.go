package loopfilter

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"testing"
)

// TestFilterEdgeVsLibvpx compares FilterEdge4 against libvpx vpx_lpf_vertical_4/
// 8/16_c. Oracle from scratchpad/lfcmp/harness.c: each line is
//
//	<width> <blimit> <limit> <thresh> I <8*40 input> O <8*40 output>
//
// (8 rows of a 40-wide buffer, edge at col 20). Set LF_ORACLE.
func TestFilterEdgeVsLibvpx(t *testing.T) {
	path := os.Getenv("LF_ORACLE")
	if path == "" {
		t.Skip("set LF_ORACLE to the harness output file")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	const W, X, rows = 40, 20, 8
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	line, mism := 0, 0
	byW := map[int]int{}
	for sc.Scan() {
		line++
		fl := strings.Fields(sc.Text())
		if len(fl) < 4 {
			continue
		}
		w, _ := strconv.Atoi(fl[0])
		bl, _ := strconv.Atoi(fl[1])
		li, _ := strconv.Atoi(fl[2])
		th, _ := strconv.Atoi(fl[3])
		ii, oi := -1, -1
		for i, s := range fl {
			if s == "I" {
				ii = i
			} else if s == "O" {
				oi = i
			}
		}
		n := rows * W
		buf := make([]byte, n)
		for i := 0; i < n; i++ {
			v, _ := strconv.Atoi(fl[ii+1+i])
			buf[i] = byte(v)
		}
		want := make([]byte, n)
		for i := 0; i < n; i++ {
			v, _ := strconv.Atoi(fl[oi+1+i])
			want[i] = byte(v)
		}
		p := Params{Limit: uint8(li), BLimit: uint8(bl), Thresh: uint8(th)}
		for r := 0; r < rows; r++ {
			FilterEdge4(buf, r*W+X, 1, w, p)
		}
		diffs := 0
		for i := 0; i < n; i++ {
			if buf[i] != want[i] {
				diffs++
			}
		}
		if diffs > 0 {
			mism++
			byW[w]++
			if byW[w] <= 1 {
				// show the edge neighbourhood of the first differing row
				for r := 0; r < rows; r++ {
					bad := false
					for c := X - 8; c <= X+7; c++ {
						if buf[r*W+c] != want[r*W+c] {
							bad = true
						}
					}
					if bad {
						t.Errorf("line %d w=%d row=%d bl=%d li=%d th=%d", line, w, r, bl, li, th)
						gs, ws := "", ""
						for c := X - 8; c <= X+7; c++ {
							gs += strconv.Itoa(int(buf[r*W+c])) + " "
							ws += strconv.Itoa(int(want[r*W+c])) + " "
						}
						t.Errorf("  got=%s\n  want=%s", gs, ws)
						break
					}
				}
			}
		}
	}
	if mism > 0 {
		t.Errorf("%d/%d lines mismatch, by width: %v", mism, line, byW)
	} else {
		t.Logf("all %d loop-filter kernel tests match libvpx", line)
	}
}
