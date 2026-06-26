package decoder

import (
	"encoding/binary"
	"os"
	"strconv"
	"testing"
)

// TestInterStream decodes a multi-frame VP9 IVF (env VP9_IVF) and compares every
// frame against a concatenated I420 reference (env VP9_REF, from vpxdec --i420).
// Prints per-frame exact-pixel counts to localise the first inter divergence.
func TestInterStream(t *testing.T) {
	ivfPath := os.Getenv("VP9_IVF")
	refPath := os.Getenv("VP9_REF")
	if ivfPath == "" || refPath == "" {
		t.Skip("set VP9_IVF and VP9_REF")
	}
	ivf, err := os.ReadFile(ivfPath)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(ivf) < 32 || string(ivf[:4]) != "DKIF" {
		t.Fatal("not an IVF file")
	}
	w := int(binary.LittleEndian.Uint16(ivf[12:]))
	h := int(binary.LittleEndian.Uint16(ivf[14:]))
	fsz := w * h * 3 / 2
	cw, ch := w/2, h/2

	dec := New()
	off := 32
	frame := 0
	for off+12 <= len(ivf) {
		sz := int(binary.LittleEndian.Uint32(ivf[off:]))
		off += 12
		if off+sz > len(ivf) {
			break
		}
		payload := ivf[off : off+sz]
		off += sz

		refOff := frame * fsz
		if refOff+fsz > len(ref) {
			break // no more reference frames
		}

		f, derr := dec.DecodeFrame(payload)
		if derr != nil || f == nil {
			t.Fatalf("frame %d: decode error: %v", frame, derr)
		}

		yb, ub, vb := f.YBase(), f.UBase(), f.VBase()
		nY, nU, nV := 0, 0, 0
		var firstY [3]int
		var atEdge, interior, big int
		var firstBig [3]int
		firstBig[0] = -1
		for r := 0; r < h; r++ {
			for c := 0; c < w; c++ {
				d := int(f.Y[yb+r*f.YStride+c]) - int(ref[refOff+r*w+c])
				if d != 0 {
					if nY == 0 {
						firstY = [3]int{r, c, d}
					}
					nY++
					if d > 1 || d < -1 {
						big++
						if firstBig[0] < 0 {
							firstBig = [3]int{r, c, d}
						}
					}
					if c%4 == 0 || r%4 == 0 || (c+1)%4 == 0 || (r+1)%4 == 0 {
						atEdge++
					} else {
						interior++
					}
				}
			}
		}
		if frame == 1 && os.Getenv("VP9_DIFFMAP") != "" {
			t.Logf("frame1 diff map: total=%d |d|>1=%d firstBig=(%d,%d)d=%d at-edge=%d interior=%d", nY, big, firstBig[0], firstBig[1], firstBig[2], atEdge, interior)
		}
		uo := refOff + w*h
		vo := uo + cw*ch
		firstU := [3]int{-1, 0, 0}
		var uBig, uEdge, uInterior int
		for r := 0; r < ch; r++ {
			for c := 0; c < cw; c++ {
				if du := int(f.U[ub+r*f.UVStride+c]) - int(ref[uo+r*cw+c]); du != 0 {
					nU++
					if firstU[0] < 0 {
						firstU = [3]int{r, c, du}
					}
					if du > 1 || du < -1 {
						uBig++
					}
					if c%4 == 0 || r%4 == 0 || (c+1)%4 == 0 || (r+1)%4 == 0 {
						uEdge++
					} else {
						uInterior++
					}
				}
				if int(f.V[vb+r*f.UVStride+c]) != int(ref[vo+r*cw+c]) {
					nV++
				}
			}
		}
		if os.Getenv("VP9_DIFFMAP") != "" {
			t.Logf("  frame %d firstU=(%d,%d)d=%d  U|d|>1=%d edge=%d interior=%d", frame, firstU[0], firstU[1], firstU[2], uBig, uEdge, uInterior)
			if dbgF, _ := strconv.Atoi(os.Getenv("VP9_DIFFMAP")); dbgF == frame {
				// 4×4-region histogram of U diffs (which 32×32 chroma cells differ).
				var grid [8][16]int
				for r := 0; r < ch; r++ {
					for c := 0; c < cw; c++ {
						if int(f.U[ub+r*f.UVStride+c]) != int(ref[uo+r*cw+c]) {
							grid[r/16][c/16]++
						}
					}
				}
				for r := 0; r < ch/16+1 && r < 8; r++ {
					t.Logf("    Urow%3d: %v", r*16, grid[r])
				}
			}
		}
		kind := "inter"
		if frame == 0 {
			kind = "KEY"
		}
		t.Logf("frame %d (%s): Y diff=%d U=%d V=%d  (Y=%d/%d exact)  firstY=(%d,%d)d=%d",
			frame, kind, nY, nU, nV, w*h-nY, w*h, firstY[0], firstY[1], firstY[2])
		frame++
	}
}
