package decoder

import (
	"embed"
	"strings"
	"testing"
)

// Frame-level header features no libvpx encoder path reachable through ffmpeg
// emits (intra-only frames, reset_frame_context 1/2/3, a non-zero
// frame_context_idx on a frame that resets its contexts, sign-bias bits on an
// error-resilient frame, show_existing_frame), pinned to libvpx's output.
//
// The streams are ordinary libvpx-vp9 encodes (128x96, lavfi testsrc /
// testsrc2) whose uncompressed headers were rewritten bit by bit with
// testdata/intraonly_vp9hdr.py; compressed header and tile data are untouched.
// Every stream decodes without error in libvpx (ffmpeg -c:v libvpx-vp9 and
// libvpx's own vpxdec give identical output); NAME.md5 holds one md5 per shown
// frame of that reference decode. Exact commands: testdata/intraonly_streams.txt.
//
//   - intraonly_reset3_ctx1: key frame of picture A, then a hidden intra-only
//     frame of picture B (reset_frame_context=3, coded frame_context_idx=1,
//     refreshes all slots) in a superframe with B's first inter frame, then
//     more inter frames. Needs the intra-only header field order, key-frame
//     style mode info and frame_context_idx forced to 0.
//   - intraonly_reset2_keeprefs: same, but reset_frame_context=2 and the
//     intra-only frame refreshes only slots 1 and 2, so LAST still holds
//     picture A (an intra-only frame must not reset the reference slots).
//   - intraonly_reset1_adapt: reset_frame_context=1 (no reset), coded
//     frame_context_idx=3, frame_parallel=0, after a key frame with
//     refresh_frame_context=0: the intra-only frame's backward adaptation must
//     start from and be saved to context 0.
//   - intraonly_key_ctx2: key frame coding frame_context_idx=2; the following
//     inter frames use context 0, which the key frame must have refreshed.
//   - intraonly_er_signbias: error-resilient inter frames with the ALTREF
//     sign-bias bit set; libvpx zeroes the sign biases on such frames, so no
//     compound-reference syntax is present.
//   - intraonly_show_existing: a show_existing_frame packet after an inter
//     frame of a backward-adaptive stream. Control: already bit-exact before
//     the fixes above (show_existing_frame must leave all state untouched).
//
// The _p2 variants are the 10-bit profile 2 equivalents (16-bit pipeline; a
// profile 2 intra-only frame also codes bit depth / colour space, before
// refresh_frame_flags). All streams except the show_existing controls failed
// before the fixes.
//
//go:embed testdata/intraonly_*.ivf testdata/intraonly_*.md5
var intraOnlyStreams embed.FS

func TestIntraOnlyStreams(t *testing.T) {
	streams := []struct {
		name   string
		frames int
	}{
		{"intraonly_reset3_ctx1", 5},
		{"intraonly_reset2_keeprefs", 5},
		{"intraonly_reset1_adapt", 2},
		{"intraonly_key_ctx2", 5},
		{"intraonly_er_signbias", 5},
		{"intraonly_show_existing", 3},
	}
	for _, s := range streams {
		for _, hbd := range []bool{false, true} {
			name := s.name
			if hbd {
				name += "_p2"
			}
			t.Run(name, func(t *testing.T) {
				data, err := intraOnlyStreams.ReadFile("testdata/" + name + ".ivf")
				if err != nil {
					t.Fatal(err)
				}
				sums, err := intraOnlyStreams.ReadFile("testdata/" + name + ".md5")
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
					got, err := decodeFrameMD5(d, p, hbd)
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
}
