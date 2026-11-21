package decoder

import (
	"embed"
	"strings"
	"testing"
)

// Frame sizes that are not a multiple of 8 (or whose 8×8 MI count is odd),
// pinned to libvpx's output.
//
// libvpx works on the MI grid, mi_cols × mi_rows whole 8×8 units
// (ceil(size/8)), and its frame buffer is that wide: the part of the last
// 8×8 column/row beyond the visible picture is coded, reconstructed, used as
// intra edge and loop-filtered like any other. Each NAME.ivf is a small
// libvpx-vp9 encode of lavfi testsrc2; NAME.md5 holds one md5 per shown frame
// of `ffmpeg -c:v libvpx-vp9 -i NAME.ivf -pix_fmt yuv420p[10le] -f rawvideo`.
//
//   - oddsize_66x66[_p2]: width and height ≡ 2 (mod 8), 9×9 MI. The grid was
//     sized ceil(size/4) 4×4 units (17 instead of 18), so partitions, coded
//     transform blocks and contexts at the right/bottom edge were wrong:
//     "entropy desync" on the key frame before the fix. Also needs the intra
//     edge and loop-filter extent to cover the 8-aligned area.
//   - oddsize_200x130[_p2]: 25×17 MI, both odd, two superblock rows/columns
//     with 1 MI left over (and the same grid error vertically: entropy desync
//     on the key frame before the fix). Exercises the chroma loop-filter
//     rules for the half-visible last chroma 8×8 column and row: 16-wide
//     filter replaced by the 8-wide one, internal 4×4 edges of the last
//     column dropped.
//   - oddsize_72x72[_p2]: multiple of 8 but 9×9 MI, so only the chroma
//     loop-filter rules apply (pixel mismatch from frame 2 / frame 0 before
//     the fix, no error).
//   - oddsize_10x10[_p2]: 2×2 MI of which 10 pixels are visible; intra
//     prediction must take its above-right / left edge from decoded pixels
//     up to the 8-aligned size 16, not replicate the pixel at the visible
//     edge (pixel mismatch on every frame before the fix, no error).
//
// Exact generation commands are in testdata/oddsize_streams.txt.
//
//go:embed testdata/oddsize_*.ivf testdata/oddsize_*.md5
var oddsizeStreams embed.FS

func TestOddSizeStreams(t *testing.T) {
	streams := []struct {
		name   string
		frames int
		hbd    bool
	}{
		{"oddsize_66x66", 16, false},
		{"oddsize_66x66_p2", 16, true},
		{"oddsize_200x130", 16, false},
		{"oddsize_200x130_p2", 16, true},
		{"oddsize_72x72", 16, false},
		{"oddsize_72x72_p2", 16, true},
		{"oddsize_10x10", 10, false},
		{"oddsize_10x10_p2", 10, true},
	}
	for _, s := range streams {
		t.Run(s.name, func(t *testing.T) {
			data, err := oddsizeStreams.ReadFile("testdata/" + s.name + ".ivf")
			if err != nil {
				t.Fatal(err)
			}
			sums, err := oddsizeStreams.ReadFile("testdata/" + s.name + ".md5")
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
