package decoder

import (
	"embed"
	"strings"
	"testing"
)

// Segmentation, pinned to libvpx's output.
//
// Each NAME.ivf is a small libvpx-vp9 encode; NAME.md5 holds one md5 per shown
// frame of the reference decode `ffmpeg -c:v libvpx-vp9 -i NAME.ivf -pix_fmt
// yuv420p[10le] -f rawvideo` (_p2 = profile 2, 10-bit, 16-bit pipeline). Exact
// generation commands are in testdata/segment_streams.txt. All of them failed
// before the segmentation fixes (entropy desync, or wrong pixels for
// segment_aq2_fp1).
//
//   - segment_aq1_fp0 / segment_aq1_fp1_p2 / segment_aq2_fp1 (-aq-mode 1 / 2):
//     ALT_Q deltas on up to 5 segments, map coded on the key frame, then
//     frames with segmentation on but neither map nor data coded (the map is
//     carried over, the features persist), then a temporally predicted map.
//   - segment_aq1_altref: the same with hidden alt-ref frames (the maps are
//     swapped after hidden frames too).
//   - segment_aq3_fp0[_p2] / segment_aq3_fp1 (-aq-mode 3, cyclic refresh):
//     segmentation off on the key frame, maps coded with and without temporal
//     prediction, segmentation switched off again at the end.
//   - segment_aq3_kf / segment_aq3_er: key frames inside the stream / every
//     frame error resilient (features and both maps are reset).
//   - segment_roi_fp0 / segment_roi_fp1_p2 (libvpx ROI map): all four features
//     at once - ALT_Q, ALT_LF (including -63, a segment filter level of 0 that
//     the ref deltas still raise), SKIP and REF_FRAME (intra / last / golden) -
//     on a moving map, with one frame in the middle coded without
//     segmentation (the maps are not swapped after it).
//   - segment_active_fp1 / segment_active_fp0_p2 (libvpx active map): SKIP +
//     ALT_LF -63 on segment 7.
//   - segment_aq1_abs / segment_roi_abs[_p2]: the above with
//     segmentation_abs_or_delta_update patched to 1 on some frames (absolute
//     quantizer index / filter level; the flag persists over frames that code
//     no data), segment_roi_abs additionally with non-default loop-filter ref
//     and mode deltas.
//   - segment_aq3_modedelta[_p2]: segment_aq3_fp0[_p2] with loop-filter mode
//     deltas patched in (-7 for ZEROMV, +6 for the other inter modes): the
//     delta is selected by the prediction mode, not by the MV being zero.
//
//go:embed testdata/segment_*.ivf testdata/segment_*.md5
var segmentStreams embed.FS

func TestSegmentationStreams(t *testing.T) {
	streams := []struct {
		name   string
		frames int
		hbd    bool
	}{
		{"segment_aq1_fp0", 12, false},
		{"segment_aq1_fp1_p2", 12, true},
		{"segment_aq2_fp1", 12, false},
		{"segment_aq1_altref", 24, false},
		{"segment_aq3_fp0", 30, false},
		{"segment_aq3_fp1", 30, false},
		{"segment_aq3_fp0_p2", 30, true},
		{"segment_aq3_kf", 20, false},
		{"segment_aq3_er", 14, false},
		{"segment_roi_fp0", 12, false},
		{"segment_roi_fp1_p2", 12, true},
		{"segment_active_fp1", 10, false},
		{"segment_active_fp0_p2", 10, true},
		{"segment_aq1_abs", 12, false},
		{"segment_roi_abs", 12, false},
		{"segment_roi_abs_p2", 12, true},
		{"segment_aq3_modedelta", 30, false},
		{"segment_aq3_modedelta_p2", 30, true},
	}
	for _, s := range streams {
		t.Run(s.name, func(t *testing.T) {
			data, err := segmentStreams.ReadFile("testdata/" + s.name + ".ivf")
			if err != nil {
				t.Fatal(err)
			}
			sums, err := segmentStreams.ReadFile("testdata/" + s.name + ".md5")
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
