package decoder

import (
	"crypto/md5"
	"embed"
	"encoding/binary"
	"fmt"
	"strings"
	"testing"
)

// Backward probability adaptation (frame_parallel_decoding_mode=0 with
// error_resilient_mode=0), pinned to libvpx's output.
//
// Each NAME.ivf is a small libvpx-vp9 encode (ffmpeg, lavfi testsrc2 /
// smptebars); NAME.md5 holds one md5 per shown frame of the reference decode
// `ffmpeg -c:v libvpx-vp9 -i NAME.ivf -pix_fmt yuv420p[10le] -f rawvideo`
// (ffmpeg's native vp9 decoder gives identical output). One shown frame per IVF
// packet; hidden alt-ref frames travel inside superframes.
//
//   - fp0_1tile_key_inter: 64x64, key + 1 inter. Minimal case: the context
//     saved after the key frame must carry the adapted coefficient probs.
//   - fp0_2tile_altref / fp0_4tile_altref: 512x288 / 1024x576, 2 / 4 tile
//     columns, 2-pass, hidden alt-refs in superframes, frame_context_idx 0 and
//     1. 288 and 576 are not multiples of 64, so the forced-partition counts at
//     the bottom edge matter; MVs exercise the implied-hp count.
//   - fp0_4tile_flat_altref: 1024x576 colour bars, 4 tile columns of a few
//     bytes each (tiles ending close to the bool decoder's over-read limit).
//   - fp0_2tile_altref_p2: 10-bit profile 2 variant (16-bit pipeline).
//   - fp0_qmax: base_q_idx 255 (AC quantizer table tail, q index >= 232).
//   - fp0_part_nocols_160x64 / fp0_part_edges_80x80[_p2]: widths (and for
//     80x80 both dimensions) that are not multiples of 64, so partitions are
//     forced at the right edge and in the corner as well.
//   - fp0_hpzero_320x192[_p2]: frames with allow_high_precision_mv=0 between
//     frames with 1 (hp probs must not adapt on the former).
//   - fp0_lf32_320x192[_p2]: low bitrate, loop_filter_level >= 32 (ref/mode
//     deltas are scaled by 1 << (level >> 5)).
//   - fp1_2tile_altref_control: same as fp0_2tile_altref with
//     frame_parallel=1 (no adaptation); guards the non-adaptive path.
//
// All fp0_* streams failed before the adaptation fixes (entropy desync or
// wrong pixels from the first frame after the key frame). Exact generation
// commands are in testdata/adapt_streams.txt.
//
//go:embed testdata/fp0_*.ivf testdata/fp0_*.md5 testdata/fp1_*.ivf testdata/fp1_*.md5
var adaptStreams embed.FS

func TestBackwardAdaptationStreams(t *testing.T) {
	streams := []struct {
		name   string
		frames int
		hbd    bool
	}{
		{"fp0_1tile_key_inter", 2, false},
		{"fp0_2tile_altref", 20, false},
		{"fp0_4tile_altref", 20, false},
		{"fp0_4tile_flat_altref", 20, false},
		{"fp0_2tile_altref_p2", 10, true},
		{"fp0_qmax", 6, false},
		{"fp0_part_nocols_160x64", 6, false},
		{"fp0_part_edges_80x80", 6, false},
		{"fp0_part_edges_80x80_p2", 6, true},
		{"fp0_hpzero_320x192", 12, false},
		{"fp0_hpzero_320x192_p2", 12, true},
		{"fp0_lf32_320x192", 30, false},
		{"fp0_lf32_320x192_p2", 30, true},
		{"fp1_2tile_altref_control", 20, false},
	}
	for _, s := range streams {
		t.Run(s.name, func(t *testing.T) {
			data, err := adaptStreams.ReadFile("testdata/" + s.name + ".ivf")
			if err != nil {
				t.Fatal(err)
			}
			sums, err := adaptStreams.ReadFile("testdata/" + s.name + ".md5")
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

// A tile cut short must still be rejected on an inter frame of an adaptive
// multi-tile stream (the over-read check is what catches it).
func TestBackwardAdaptationTruncatedTile(t *testing.T) {
	data, err := adaptStreams.ReadFile("testdata/fp0_2tile_altref.ivf")
	if err != nil {
		t.Fatal(err)
	}
	packets := readIVFPackets(t, data)
	// First plain (non-superframe) inter packet; dropping its tail truncates
	// the last tile.
	victim := 0
	for i := 1; i < len(packets); i++ {
		if len(splitSuperframe(packets[i])) == 1 {
			victim = i
			break
		}
	}
	if victim == 0 {
		t.Fatal("no plain inter packet")
	}
	for _, cut := range []int{1, 2, 8, len(packets[victim]) / 4} {
		d := New()
		for i := 0; i < victim; i++ {
			if _, err := decodeFrameMD5(d, packets[i], false); err != nil {
				t.Fatal(err)
			}
		}
		p := packets[victim][:len(packets[victim])-cut]
		if _, err := decodeFrameMD5(d, p, false); err == nil {
			t.Errorf("packet %d: last tile truncated by %d bytes decoded without error", victim, cut)
		}
	}
}

// decodeFrameMD5 decodes one IVF packet and returns the md5 of the shown
// frame as raw planar 4:2:0 (16-bit little-endian samples when hbd).
func decodeFrameMD5(d *Decoder, packet []byte, hbd bool) (sum string, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()
	h := md5.New()
	if hbd {
		f, err := d.DecodeFrame16(packet)
		if err != nil {
			return "", err
		}
		if f == nil {
			return "", fmt.Errorf("no frame shown")
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
		plane(f.U, f.UBase(), f.UVStride, f.Width/2, f.Height/2)
		plane(f.V, f.VBase(), f.UVStride, f.Width/2, f.Height/2)
		return fmt.Sprintf("%x", h.Sum(nil)), nil
	}
	f, err := d.DecodeFrame(packet)
	if err != nil {
		return "", err
	}
	if f == nil {
		return "", fmt.Errorf("no frame shown")
	}
	plane := func(pl []byte, base, stride, w, hgt int) {
		for r := 0; r < hgt; r++ {
			h.Write(pl[base+r*stride : base+r*stride+w])
		}
	}
	plane(f.Y, f.YBase(), f.YStride, f.Width, f.Height)
	plane(f.U, f.UBase(), f.UVStride, f.Width/2, f.Height/2)
	plane(f.V, f.VBase(), f.UVStride, f.Width/2, f.Height/2)
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}
