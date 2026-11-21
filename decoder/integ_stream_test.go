package decoder

import (
	"crypto/md5"
	"embed"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
)

// Regression streams for the bugs found by cross-verifying the merged topics
// (odd frame sizes, header corner cases, error-state handling), pinned to
// libvpx packet by packet.
//
// Each integ_NAME.ivf has an integ_NAME.ref with one line per IVF packet, the
// verdict of libvpx's vpx_codec_decode + vpx_codec_get_frame for that packet
// (testdata/robust_vpxpk.c.txt, index column and error text removed):
//
//	OK <md5> <width>x<height>   a frame was output; md5 of its planes in the
//	                            stream's own chroma format, 16-bit little-endian
//	                            samples for high bit depth
//	OK -                        decoded, no frame output
//	ERR                         the packet was rejected
//
// integLine produces the same line from this decoder. Generation commands and
// the before/after state of every stream are in testdata/integ_streams.txt.
//
//go:embed testdata/integ_*.ivf testdata/integ_*.ref
var integStreams embed.FS

// integLine decodes one packet and formats the outcome like an .ref line. The
// 8-bit or 16-bit entry point is chosen from the packet's profile bits.
func integLine(d *Decoder, packet []byte) (line string) {
	defer func() {
		if r := recover(); r != nil {
			line = fmt.Sprintf("PANIC %v", r)
		}
	}()
	h := md5.New()
	hbd := len(packet) > 0 && packet[0]&0x10 != 0 // profile high bit: profile 2/3
	if hbd {
		f, err := d.DecodeFrame16(packet)
		if err != nil {
			return "ERR"
		}
		if f == nil {
			return "OK -"
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
		return fmt.Sprintf("OK %x %dx%d", h.Sum(nil), f.Width, f.Height)
	}
	f, err := d.DecodeFrame(packet)
	if err != nil {
		return "ERR"
	}
	if f == nil {
		return "OK -"
	}
	plane := func(pl []byte, base, stride, w, hgt int) {
		for r := 0; r < hgt; r++ {
			h.Write(pl[base+r*stride : base+r*stride+w])
		}
	}
	plane(f.Y, f.YBase(), f.YStride, f.Width, f.Height)
	plane(f.U, f.UBase(), f.UVStride, f.UVWidth(), f.UVHeight())
	plane(f.V, f.VBase(), f.UVStride, f.UVWidth(), f.UVHeight())
	return fmt.Sprintf("OK %x %dx%d", h.Sum(nil), f.Width, f.Height)
}

// integLines decodes a whole embedded stream with a fresh decoder.
func integLines(t *testing.T, name string) (got, want []string) {
	t.Helper()
	data, err := integStreams.ReadFile("testdata/" + name + ".ivf")
	if err != nil {
		t.Fatal(err)
	}
	ref, err := integStreams.ReadFile("testdata/" + name + ".ref")
	if err != nil {
		t.Fatal(err)
	}
	want = strings.Split(strings.TrimSpace(string(ref)), "\n")
	d := New()
	for _, p := range readIVFPackets(t, data) {
		got = append(got, integLine(d, p))
	}
	return got, want
}

func runIntegStreams(t *testing.T, names []string) {
	t.Helper()
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			got, want := integLines(t, name)
			if len(got) != len(want) {
				t.Fatalf("%d packets, %d reference lines", len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("packet %d: got %q, libvpx %q", i, got[i], want[i])
				}
			}
		})
	}
}

// Frames with an odd width or height on a chroma-subsampled axis. The planes
// are allocated for the even-rounded size, but the picture - what the decoder
// reports and what later frames predict from - has the true size: libvpx
// replicates a reference's border from the true edge (y_crop_width /
// y_crop_height), so a motion vector reaching past the right or bottom edge
// must not see the extra decoded column or row.
//
// All are lossy 3-6 frame encodes of a rotating testsrc2 at 65x64 or 64x65
// whose inter blocks reach across the odd edge. Before the fix each one had
// luma mismatches on an inter frame and reported the size rounded up to even.
func TestIntegOddSizeStreams(t *testing.T) {
	runIntegStreams(t, []string{
		"integ_odd_420_65x64",
		"integ_odd_420_64x65",
		"integ_odd_422_65x64",
		"integ_odd_440_64x65",
		"integ_odd_420p10_65x64",
	})
}

// Coded quantizer deltas (delta_q_y_dc / uv_dc / uv_ac). libvpx's encoder never
// codes them, so the streams are libvpx encodes whose key-frame header was
// rewritten to carry deltas (-15, 15, 9 for integ_deltaq; -2, 3, -1 for the
// 10-bit integ_deltaq_p2); libvpx decodes both. A delta is a 4-bit magnitude
// followed by a sign bit. Reading it as 4-bit two's complement shifted every
// later header field by one bit: the key frame failed with "corrupt
// compressed header" and the rest of the stream was rejected.
func TestIntegDeltaQStreams(t *testing.T) {
	runIntegStreams(t, []string{"integ_deltaq", "integ_deltaq_p2"})
}

// integPairs are pairs of streams (from the other topics' testdata) decoded by
// two Decoder instances at once: different frame sizes, chroma formats and bit
// depths, plus one stream paired with itself.
var integPairs = [][2]string{
	{"oddsize_200x130", "chroma_444"},
	{"oddsize_66x66", "oddsize_200x130"},
	{"chroma_444_p3_12", "oddsize_66x66_p2"},
	{"oddsize_66x66_p2", "oddsize_200x130"},
	{"segment_aq3_fp0", "lossless_160x96_fp0"},
	{"tilerows_r1c1_p2", "fp0_2tile_altref"},
	{"oddsize_72x72", "oddsize_72x72"},
}

func integPackets(t *testing.T, name string) [][]byte {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name + ".ivf")
	if err != nil {
		t.Fatal(err)
	}
	return readIVFPackets(t, data)
}

// integAlone decodes a stream on its own decoder; every packet must decode.
func integAlone(t *testing.T, packets [][]byte) []string {
	t.Helper()
	d := New()
	out := make([]string, len(packets))
	for i, p := range packets {
		out[i] = integLine(d, p)
		if !strings.HasPrefix(out[i], "OK ") {
			t.Fatalf("packet %d decoded alone: %s", i, out[i])
		}
	}
	return out
}

// Two decoders fed alternately, one packet each, must each produce what they
// produce alone. The MI grid bounds used by reconstruction were package state
// written only when a decoder's own grid size changed, so the decoder that
// allocated last set the transform-block skip bounds, intra edge and MV clamp
// of both: wrong pixels or an entropy desync from packet 1 on.
func TestIntegInterleavedDecoders(t *testing.T) {
	for _, pr := range integPairs {
		t.Run(pr[0]+"+"+pr[1], func(t *testing.T) {
			pk := [2][][]byte{integPackets(t, pr[0]), integPackets(t, pr[1])}
			want := [2][]string{integAlone(t, pk[0]), integAlone(t, pk[1])}
			ds := [2]*Decoder{New(), New()}
			failed := [2]bool{}
			for i := 0; i < len(pk[0]) || i < len(pk[1]); i++ {
				for k := 0; k < 2; k++ {
					if i >= len(pk[k]) || failed[k] {
						continue
					}
					if got := integLine(ds[k], pk[k][i]); got != want[k][i] {
						t.Errorf("%s packet %d: interleaved %q, alone %q", pr[k], i, got, want[k][i])
						failed[k] = true
					}
				}
			}
		})
	}
}

// Two decoders running in separate goroutines must also match their solo
// output (run under -race to check for unsynchronised shared state). The eob
// of the last transform block used to be package state feeding the entropy
// contexts, so even two decoders on the same stream desynchronised.
func TestIntegConcurrentDecoders(t *testing.T) {
	for _, pr := range integPairs {
		t.Run(pr[0]+"+"+pr[1], func(t *testing.T) {
			pk := [2][][]byte{integPackets(t, pr[0]), integPackets(t, pr[1])}
			want := [2][]string{integAlone(t, pk[0]), integAlone(t, pk[1])}
			for round := 0; round < 3; round++ {
				var got [2][]string
				var wg sync.WaitGroup
				for k := 0; k < 2; k++ {
					wg.Add(1)
					go func(k int) {
						defer wg.Done()
						d := New()
						for _, p := range pk[k] {
							got[k] = append(got[k], integLine(d, p))
						}
					}(k)
				}
				wg.Wait()
				for k := 0; k < 2; k++ {
					for i := range want[k] {
						if got[k][i] != want[k][i] {
							t.Fatalf("round %d, %s packet %d: concurrent %q, alone %q", round, pr[k], i, got[k][i], want[k][i])
						}
					}
				}
			}
		})
	}
}

// A key frame rejected late in its uncompressed header has still reset the
// frame contexts: libvpx runs vp9_setup_past_independence as soon as
// frame_context_idx is read, before the errors for a truncated header tail,
// too many tile columns, a zero or out-of-packet first_partition_size. The
// streams are intraonly_reset1_adapt with packet 0 (the key frame) cut to 12
// bytes (inside the header, behind frame_context_idx) or 22 bytes (complete
// uncompressed header, compressed header cut short). Packet 1 is an
// intra-only frame with reset_frame_context=1, which uses context 0 as it
// finds it: libvpx decodes it; the decoder used to reject it with
// "uninitialized entropy context" because the failed key frame had not reset
// anything.
func TestIntegContextResetOnRejectedHeader(t *testing.T) {
	runIntegStreams(t, []string{
		"integ_ctxreset_trunc12",
		"integ_ctxreset_trunc22",
		"integ_ctxreset_trunc22_p2",
		// Key frame whose width field has a bit flipped: 16512x96, over this
		// decoder's default size limit (libvpx fails it on its tile sizes).
		// The limit is applied after the context reset for the same reason.
		"integ_ctxreset_bigsize",
	})
}

// Packets that decode without error but output nothing, as libvpx's
// vpx_codec_get_frame reports them; DecodeFrame returns (nil, nil).
//
//   - integ_lone_hidden[_p2]: packet 1 is a hidden intra-only frame alone in
//     its packet (not in a superframe with a shown frame). The decoder used
//     to return the hidden frame as a "best effort" picture, one more frame
//     than libvpx, vpxdec and ffmpeg output for the stream.
//   - integ_showexisting_resync[_p2]: key frame, an inter frame cut to 10
//     bytes (rejected), then a show_existing_frame packet. While libvpx waits
//     for a key or intra-only frame after the error it accepts the
//     show_existing_frame but outputs nothing; the decoder used to hand back
//     the reference slot's picture.
func TestIntegNoOutputPackets(t *testing.T) {
	runIntegStreams(t, []string{
		"integ_lone_hidden",
		"integ_lone_hidden_p2",
		"integ_showexisting_resync",
		"integ_showexisting_resync_p2",
	})
}

// An inter frame may only predict from references of its own bit depth
// (libvpx: "Referenced frame has incompatible color format"; the chroma
// subsampling half of that check is TestChromaFormatMismatchedReference).
// The stream is intraonly_reset2_keeprefs_p2 with the bit-depth bit of the
// intra-only frame in packet 1 flipped from 10 to 12 bit: that frame refreshes
// slots 1 and 2 only, so the inter frame behind it in the superframe finds a
// 10-bit LAST reference. libvpx rejects the packet and everything after it
// until a key frame; the decoder used to predict the 12-bit frame from the
// 10-bit reference and return pictures.
func TestIntegReferenceBitDepth(t *testing.T) {
	runIntegStreams(t, []string{"integ_refdepth_p2"})
}

// The two errors callers can test for: the resync state after a failed packet
// and the frame-size limit.
func TestIntegExportedErrors(t *testing.T) {
	packets := integPackets(t, "integ_showexisting_resync") // key, truncated inter, show_existing
	d := New()
	if _, err := d.DecodeFrame(packets[0]); err != nil {
		t.Fatal(err)
	}
	if _, err := d.DecodeFrame(packets[1]); err == nil {
		t.Fatal("truncated inter frame decoded")
	}
	inter := integPackets(t, "intraonly_show_existing")[1]
	if _, err := d.DecodeFrame(inter); !errors.Is(err, ErrNeedResync) {
		t.Fatalf("inter frame after an error: err = %v, want ErrNeedResync", err)
	}
	if f, err := d.DecodeFrame(packets[0]); err != nil || f == nil {
		t.Fatalf("key frame after an error: frame %v, err %v", f != nil, err)
	}
	if f, err := d.DecodeFrame(inter); err != nil || f == nil {
		t.Fatalf("inter frame after the key frame: frame %v, err %v", f != nil, err)
	}

	d = New()
	d.SetMaxFrameSize(64, 64)
	if _, err := d.DecodeFrame(packets[0]); !errors.Is(err, ErrFrameSize) {
		t.Fatalf("128x96 key frame with a 64x64 limit: err = %v, want ErrFrameSize", err)
	}
}
