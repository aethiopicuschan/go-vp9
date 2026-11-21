package decoder

import (
	"crypto/md5"
	"embed"
	"encoding/binary"
	"fmt"
	"strings"
	"testing"
)

// Non-4:2:0 chroma formats (profile 1: 8-bit, profile 3: 10/12-bit), pinned to
// libvpx's output.
//
// Each chroma_NAME.ivf is a small two-pass libvpx-vp9 encode (ffmpeg, lavfi
// rotating testsrc2, 104x72 = 13x9 mode-info units so the right and bottom
// superblocks, and for the subsampled axes the last chroma 8x8 units, are
// partial; 24 shown frames with hidden alt-refs in superframes).
// chroma_NAME.md5 holds one md5 per shown frame of the reference decode
// `ffmpeg -c:v libvpx-vp9 -i chroma_NAME.ivf -f framemd5` in the stream's
// native pixel format (planar, 16-bit little-endian samples for profile 3).
// Every stream contains intra blocks, single and compound inter prediction,
// sub-8x8 inter blocks (whose chroma MVs depend on the subsampling) and loop
// filtering.
//
//   - chroma_444 / chroma_444_p3 / chroma_444_p3_12: 4:4:4 at 8 / 10 / 12 bit.
//   - chroma_422 / chroma_422_p3: 4:2:2 at 8 / 10 bit.
//   - chroma_440 / chroma_440_p3: 4:4:0 at 8 / 10 bit.
//   - chroma_srgb: 8-bit GBR (color_space 7, which implies 4:4:4 and has no
//     subsampling bits in the header); planes are G, B, R.
//
// Before non-4:2:0 support every one of these decoded the key frame to wrong
// pixels (as 4:2:0) and hit an entropy desync on the first or second inter
// packet. Exact generation commands are in testdata/chroma_streams.txt.
//
//go:embed testdata/chroma_*.ivf testdata/chroma_*.md5
var chromaStreams embed.FS

func TestChromaFormatStreams(t *testing.T) {
	streams := []struct {
		name       string
		hbd        bool
		subX, subY int
	}{
		{"chroma_444", false, 0, 0},
		{"chroma_422", false, 1, 0},
		{"chroma_440", false, 0, 1},
		{"chroma_srgb", false, 0, 0},
		{"chroma_444_p3", true, 0, 0},
		{"chroma_422_p3", true, 1, 0},
		{"chroma_440_p3", true, 0, 1},
		{"chroma_444_p3_12", true, 0, 0},
	}
	const frames = 24
	for _, s := range streams {
		t.Run(s.name, func(t *testing.T) {
			data, err := chromaStreams.ReadFile("testdata/" + s.name + ".ivf")
			if err != nil {
				t.Fatal(err)
			}
			sums, err := chromaStreams.ReadFile("testdata/" + s.name + ".md5")
			if err != nil {
				t.Fatal(err)
			}
			want := strings.Fields(string(sums))
			packets := readIVFPackets(t, data)
			if len(packets) != frames || len(want) != frames {
				t.Fatalf("%d packets, %d reference md5s, want %d of each", len(packets), len(want), frames)
			}
			d := New()
			for i, p := range packets {
				got, subX, subY, err := chromaFrameMD5(d, p, s.hbd)
				if err != nil {
					t.Fatalf("frame %d: %v", i, err)
				}
				if subX != s.subX || subY != s.subY {
					t.Fatalf("frame %d: subsampling %d,%d, want %d,%d", i, subX, subY, s.subX, s.subY)
				}
				if got != want[i] {
					t.Fatalf("frame %d: md5 %s, libvpx %s", i, got, want[i])
				}
			}
		})
	}
}

// An inter frame cannot predict from references of another chroma format
// (libvpx: "Referenced frame has incompatible color format"); it must be
// rejected, not decoded against mismatched planes.
func TestChromaFormatMismatchedReference(t *testing.T) {
	for _, hbd := range []bool{false, true} {
		sfx := ""
		if hbd {
			sfx = "_p3"
		}
		key, err := chromaStreams.ReadFile("testdata/chroma_444" + sfx + ".ivf")
		if err != nil {
			t.Fatal(err)
		}
		other, err := chromaStreams.ReadFile("testdata/chroma_422" + sfx + ".ivf")
		if err != nil {
			t.Fatal(err)
		}
		d := New()
		if _, _, _, err := chromaFrameMD5(d, readIVFPackets(t, key)[0], hbd); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := chromaFrameMD5(d, readIVFPackets(t, other)[1], hbd); err == nil {
			t.Errorf("hbd=%v: 4:2:2 inter frame decoded against 4:4:4 references without error", hbd)
		}
	}
}

// chromaFrameMD5 decodes one IVF packet and returns the md5 of the shown
// frame as raw planar data in the frame's own chroma format (16-bit
// little-endian samples when hbd), plus the frame's subsampling shifts.
func chromaFrameMD5(d *Decoder, packet []byte, hbd bool) (sum string, subX, subY int, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	h := md5.New()
	if hbd {
		f, err := d.DecodeFrame16(packet)
		if err != nil {
			return "", 0, 0, err
		}
		if f == nil {
			return "", 0, 0, fmt.Errorf("no frame shown")
		}
		var s [2]byte
		plane := func(pl []uint16, base, stride, w, hgt int) {
			for r := 0; r < hgt; r++ {
				for _, v := range pl[base+r*stride : base+r*stride+w] {
					binary.LittleEndian.PutUint16(s[:], v)
					h.Write(s[:])
				}
			}
		}
		plane(f.Y, f.YBase(), f.YStride, f.Width, f.Height)
		plane(f.U, f.UBase(), f.UVStride, f.UVWidth(), f.UVHeight())
		plane(f.V, f.VBase(), f.UVStride, f.UVWidth(), f.UVHeight())
		return fmt.Sprintf("%x", h.Sum(nil)), f.SubX, f.SubY, nil
	}
	f, err := d.DecodeFrame(packet)
	if err != nil {
		return "", 0, 0, err
	}
	if f == nil {
		return "", 0, 0, fmt.Errorf("no frame shown")
	}
	plane := func(pl []byte, base, stride, w, hgt int) {
		for r := 0; r < hgt; r++ {
			h.Write(pl[base+r*stride : base+r*stride+w])
		}
	}
	plane(f.Y, f.YBase(), f.YStride, f.Width, f.Height)
	plane(f.U, f.UBase(), f.UVStride, f.UVWidth(), f.UVHeight())
	plane(f.V, f.VBase(), f.UVStride, f.UVWidth(), f.UVHeight())
	return fmt.Sprintf("%x", h.Sum(nil)), f.SubX, f.SubY, nil
}
