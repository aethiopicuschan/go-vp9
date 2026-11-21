package decoder

import (
	"runtime"
	"strings"
	"testing"
)

// fuzzMaxDim is the frame size limit the fuzz target runs with (the seed
// streams are at most 1024x576); it bounds what one input can allocate.
const fuzzMaxDim = 1024

// fuzzSeedStreams are the streams FuzzDecodeFrame draws its seeds and its
// decoder states from: small frames first (cheap executions), then 2 and 4
// tile columns, superframes with hidden frames, and 10-bit.
var fuzzSeedStreams = []string{
	"fp0_1tile_key_inter.ivf",
	"fp0_part_edges_80x80.ivf",
	"fp0_part_edges_80x80_p2.ivf",
	"fp0_qmax.ivf",
	"fp0_hpzero_320x192.ivf",
	"fp0_lf32_320x192_p2.ivf",
	"robust_kf2_2tile.ivf",
	"robust_kf2_2tile_p2.ivf",
	"fp0_2tile_altref.ivf",
	"fp0_2tile_altref_p2.ivf",
	"fp0_4tile_flat_altref.ivf",
}

func loadFuzzStream(t testing.TB, name string) *robustStream {
	t.Helper()
	data, err := robustStreams.ReadFile("testdata/" + name)
	if err != nil {
		if data, err = adaptStreams.ReadFile("testdata/" + name); err != nil {
			t.Fatal(err)
		}
	}
	s := &robustStream{hbd: strings.HasSuffix(name, "_p2.ivf")}
	for _, p := range readIVFPacketsTB(t, data) {
		s.packets = append(s.packets, p)
	}
	d := New()
	for i, p := range s.packets {
		sum, err := robustDecodeMD5(d, p, s.hbd)
		if err != nil {
			t.Fatalf("%s: uncorrupted packet %d: %v", name, i, err)
		}
		s.orig = append(s.orig, sum)
		s.key = append(s.key, len(p) > 0 && p[0]&0xcc == 0x80)
	}
	return s
}

// readIVFPacketsTB is readIVFPackets for a testing.TB (fuzz targets have no
// *testing.T while seeding).
func readIVFPacketsTB(t testing.TB, data []byte) [][]byte {
	t.Helper()
	var out [][]byte
	for o := 32; o+12 <= len(data); {
		n := int(uint32(data[o]) | uint32(data[o+1])<<8 | uint32(data[o+2])<<16 | uint32(data[o+3])<<24)
		o += 12
		if n < 0 || o+n > len(data) {
			t.Fatalf("truncated IVF")
		}
		out = append(out, data[o:o+n])
		o += n
	}
	return out
}

// FuzzDecodeFrame feeds an arbitrary packet to a decoder that has decoded the
// first k packets of one of the seed streams (so the input is also tried as an
// inter frame against real references and adapted probabilities) and checks
// that the decoder
//
//   - does not panic, whatever the bytes are;
//   - never hands out a frame larger than its size limit (SetMaxFrameSize):
//     a few header bytes must not make it allocate without bound;
//   - is not poisoned: after two more packets of the stream (which may fail or
//     decode to anything), the stream's key frame decodes without error to
//     exactly the picture it gives on a fresh decoder.
//
// Run it with: go test ./decoder -run '^$' -fuzz FuzzDecodeFrame -fuzztime 3m
func FuzzDecodeFrame(f *testing.F) {
	var streams []*robustStream
	for si, name := range fuzzSeedStreams {
		s := loadFuzzStream(f, name)
		streams = append(streams, s)
		for k, p := range s.packets {
			// Every key frame and superframe, and the first inter frames.
			if k < 4 || s.key[k] || p[len(p)-1]&0xe0 == 0xc0 {
				f.Add(uint8(si), uint8(k), p)
			}
		}
	}
	f.Add(uint8(0), uint8(0), []byte{})
	f.Add(uint8(0), uint8(1), []byte{0x88})                                                       // show_existing_frame 0
	f.Add(uint8(0), uint8(0), []byte{0x82, 0x49, 0x83, 0x42, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00}) // truncated key frame header

	f.Fuzz(func(t *testing.T, si, k uint8, data []byte) {
		s := streams[int(si)%len(streams)]
		n := int(k) % len(s.packets)
		d := New()
		d.SetMaxFrameSize(fuzzMaxDim, fuzzMaxDim)
		for i := 0; i < n; i++ {
			if _, err := robustDecodeMD5(d, s.packets[i], s.hbd); err != nil {
				t.Fatalf("uncorrupted packet %d: %v", i, err)
			}
		}
		fuzzDecode(t, d, data, s.hbd)
		for i := n + 1; i < len(s.packets) && i <= n+2; i++ {
			fuzzDecode(t, d, s.packets[i], s.hbd)
		}
		got, err := robustDecodeMD5(d, s.packets[0], s.hbd)
		if err != nil {
			t.Fatalf("key frame after the fuzzed packet: %v", err)
		}
		if got != s.orig[0] {
			t.Fatalf("key frame after the fuzzed packet: md5 %s, want %s", got, s.orig[0])
		}
	})
}

// fuzzDecode decodes one packet, ignoring a decode error, and checks the
// size of whatever comes back.
func fuzzDecode(t *testing.T, d *Decoder, packet []byte, hbd bool) {
	t.Helper()
	w, h := 0, 0
	if hbd {
		f, err := d.DecodeFrame16(packet)
		if err != nil || f == nil {
			return
		}
		w, h = f.Width, f.Height
	} else {
		f, err := d.DecodeFrame(packet)
		if err != nil || f == nil {
			return
		}
		w, h = f.Width, f.Height
	}
	if w <= 0 || h <= 0 || w > fuzzMaxDim+1 || h > fuzzMaxDim+1 {
		t.Fatalf("decoded a %dx%d frame with a %dx%d limit", w, h, fuzzMaxDim, fuzzMaxDim)
	}
}

// A key frame whose header announces a size beyond the limit (bit 15 of the
// width or height set: 33280x64, 512x32832) must be turned down before
// anything of that size is allocated. libvpx only does so when built with
// --size-limit (see DefaultMaxFrameSize); without it, it allocates the frame
// and fails later, on the tile data.
func TestRobustOversizedFrameRejectedEarly(t *testing.T) {
	for _, c := range []struct{ stream, edit string }{
		{"robust_kf2_2tile.ivf", "xor:4:08"},
		{"robust_kf2_2tile.ivf", "xor:6:08"},
		{"robust_kf2_2tile.ivf", "xor:4:08;xor:6:08"},
		{"robust_kf2_2tile_p2.ivf", "xor:4:04"},
		{"robust_kf2_2tile_p2.ivf", "xor:6:04"},
	} {
		s := loadRobustStream(t, c.stream)
		p := robustApplyEdit(t, s.packets[0], c.edit)
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		_, err := robustDecodeMD5(New(), p, s.hbd)
		runtime.ReadMemStats(&after)
		if err == nil {
			t.Errorf("%s %s: decoded without error", c.stream, c.edit)
		} else if !strings.Contains(err.Error(), "outside the allowed 16384x16384") {
			t.Errorf("%s %s: rejected for another reason: %v", c.stream, c.edit, err)
		}
		if grew := after.TotalAlloc - before.TotalAlloc; grew > 4<<20 {
			t.Errorf("%s %s: allocated %d MB before rejecting the packet", c.stream, c.edit, grew>>20)
		}
		// With the limit raised the frame is accepted as far as its size goes
		// (and then fails on its tile data, like in libvpx).
		if strings.Contains(c.edit, ";") {
			continue // 33280x32832: not worth a gigabyte
		}
		d := New()
		d.SetMaxFrameSize(65536, 65536)
		if _, err := robustDecodeMD5(d, p, s.hbd); err == nil || strings.Contains(err.Error(), "outside the allowed") {
			t.Errorf("%s %s with a 65536x65536 limit: err = %v", c.stream, c.edit, err)
		}
	}
}
