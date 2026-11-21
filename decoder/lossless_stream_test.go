package decoder

import (
	"embed"
	"strings"
	"testing"
)

// Lossless frames (base_q_idx == 0 and all delta_q == 0; ffmpeg `-lossless 1`),
// pinned to libvpx's output.
//
// Each lossless_NAME.ivf is a small libvpx-vp9 encode (ffmpeg, lavfi testsrc2 /
// smptebars); lossless_NAME.md5 holds one md5 per shown frame of the reference
// decode `ffmpeg -c:v libvpx-vp9 -i NAME.ivf -pix_fmt yuv420p[10le|12le]
// -f rawvideo`. One shown frame per IVF packet. Exact generation commands are
// in testdata/lossless_streams.txt.
//
// What a lossless frame changes in the decoder (libvpx xd->lossless):
//   - the compressed header has no tx_mode syntax (tx_mode is ONLY_4X4);
//   - every transform block, luma and chroma, is a 4x4 inverse WHT
//     (vpx_iwht4x4_16_add / vpx_highbd_iwht4x4_16_add);
//   - intra luma blocks use the default coefficient scan (tx_type DCT_DCT)
//     whatever the intra mode.
//
// The streams cover key + inter frames, frame_parallel 0 and 1 (with and
// without backward adaptation), 8/10/12-bit, sizes that are multiples of 64
// (64x64, 320x192) and sizes that are only multiples of 8 (160x96, 72x40).
// Every stream decoded to wrong pixels from frame 0 (or desynced) before the
// lossless fixes.
//
//go:embed testdata/lossless_*.ivf testdata/lossless_*.md5
var losslessStreams embed.FS

func TestLosslessStreams(t *testing.T) {
	streams := []struct {
		name   string
		frames int
		hbd    bool
	}{
		{"lossless_64x64_fp0", 3, false},
		{"lossless_64x64_fp1", 3, false},
		{"lossless_160x96_fp0", 3, false},
		{"lossless_160x96_fp1", 3, false},
		{"lossless_72x40_fp0", 6, false},
		{"lossless_bars_320x192_fp0", 4, false},
		{"lossless_64x64_fp0_p2", 3, true},
		{"lossless_160x96_fp1_p2", 3, true},
		{"lossless_bars_320x192_fp1_p2", 4, true},
		{"lossless_64x64_fp0_p2_12bit", 3, true},
	}
	for _, s := range streams {
		t.Run(s.name, func(t *testing.T) {
			data, err := losslessStreams.ReadFile("testdata/" + s.name + ".ivf")
			if err != nil {
				t.Fatal(err)
			}
			sums, err := losslessStreams.ReadFile("testdata/" + s.name + ".md5")
			if err != nil {
				t.Fatal(err)
			}
			want := strings.Fields(string(sums))
			packets := readIVFPackets(t, data)
			if len(packets) != s.frames || len(want) != s.frames {
				t.Fatalf("%d packets, %d reference md5s, want %d of each", len(packets), len(want), s.frames)
			}
			d := New()
			for i, p := range packets {
				got, err := decodeFrameMD5(d, p, s.hbd)
				if err != nil {
					t.Fatalf("frame %d: %v", i, err)
				}
				if got != want[i] {
					t.Fatalf("frame %d: md5 %s, libvpx %s", i, got, want[i])
				}
			}
		})
	}
}
