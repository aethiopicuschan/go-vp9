package decoder

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"testing"
)

// Corrupt-input behaviour on multi-packet streams, pinned to libvpx.
//
// testdata/robust_corrupt.tsv lists byte-level corruptions of single packets
// of five streams (key frames, inter frames, superframes with a hidden
// alt-ref, 2 and 4 tile columns, 8 and 10 bit) together with what libvpx did
// with the corrupted packet AND with the packets after it: decode error,
// no output, or the md5 of the frame it output. The table is produced by
// testdata/robust_corrupt_gen.py.txt driving a small libvpx client
// (testdata/robust_vpxpk.c.txt); file formats and edit operations are
// described at the top of the generator.
//
// For every row this decoder must
//   - never panic,
//   - return an error for each packet libvpx fails,
//   - decode each packet libvpx decodes, to the same picture.
//
// That includes libvpx's packet rules (vp9_dx_iface.c decoder_decode): frames
// of a superframe get the sizes its index gives them; in a packet without
// index, whatever follows the end of the frame's last tile
// (vpx_reader_find_end) is skipped while it is zero and otherwise decoded as
// another frame; and its error state: after a failed frame every frame is
// rejected until a key frame (or intra-only frame) arrives.
//
// robust_kf2_2tile[_p2].ivf (12 frames, key frames at 0 and 6) are recorded
// to the end of the stream, so each of their rows also checks that decoding
// is bit-exact again from the next key frame on.
//
//go:embed testdata/robust_*.ivf testdata/robust_*.md5 testdata/robust_corrupt.tsv
//go:embed testdata/fp0_2tile_altref.ivf testdata/fp0_4tile_flat_altref.ivf testdata/fp0_2tile_altref_p2.ivf
var robustStreams embed.FS

// robustUndefinedPixels lists the rows ("stream/packet/edit") where libvpx
// decodes the corrupted packet and so does this decoder, with the same
// symbols, but to a different picture. All of them are streams whose
// coefficients overflow the inverse transform: the table marks them R
// (libvpx's own range checks fired, see the generator), the format leaves the
// result undefined, and libvpx itself is not consistent on them (its C and
// SIMD transforms differ). For these rows only the verdicts are checked, the
// pictures are not compared up to the next key frame. The test refuses an
// entry whose row is not marked R, and an entry that is not needed.
//
// (13 further R rows decode to libvpx's picture anyway.)
var robustUndefinedPixels = map[string]bool{
	// base_q_idx with its top bit flipped (quantizer far too large for the
	// coefficients that follow)
	"fp0_2tile_altref.ivf/0/xor:13:01":      true,
	"fp0_4tile_flat_altref.ivf/0/xor:13:01": true,
	"robust_kf2_2tile.ivf/0/xor:13:01":      true,
	"robust_kf2_2tile.ivf/6/xor:13:01":      true,
	"robust_kf2_2tile_p2.ivf/0/xor:14:80":   true,
	// second half of a key-frame tile overwritten with 0xff
	"robust_kf2_2tile.ivf/6/fill:1171:1014:ff": true,
	// single bit flips inside key-frame tile data
	"robust_kf2_2tile.ivf/0/xor:990:02":    true,
	"robust_kf2_2tile.ivf/0/xor:417:01":    true,
	"robust_kf2_2tile.ivf/0/xor:535:02":    true,
	"robust_kf2_2tile.ivf/6/xor:834:80":    true,
	"robust_kf2_2tile_p2.ivf/0/xor:450:02": true,
}

type robustStream struct {
	packets [][]byte
	hbd     bool
	orig    []string // md5 of each packet's output, uncorrupted stream
	key     []bool   // packet starts with a key frame
}

func loadRobustStream(t *testing.T, name string) *robustStream {
	t.Helper()
	data, err := robustStreams.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	s := &robustStream{packets: readIVFPackets(t, data), hbd: strings.HasSuffix(name, "_p2.ivf")}
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

// The two-key-frame streams themselves: one md5 per frame from
// `ffmpeg -c:v libvpx-vp9 -i NAME.ivf -f framemd5` (see robust_streams.txt).
func TestRobustStreams(t *testing.T) {
	for _, name := range []string{"robust_kf2_2tile", "robust_kf2_2tile_p2"} {
		t.Run(name, func(t *testing.T) {
			s := loadRobustStream(t, name+".ivf")
			sums, err := robustStreams.ReadFile("testdata/" + name + ".md5")
			if err != nil {
				t.Fatal(err)
			}
			want := strings.Fields(string(sums))
			if len(want) != len(s.orig) || len(want) != 12 {
				t.Fatalf("%d packets, %d reference md5s", len(s.orig), len(want))
			}
			for i := range want {
				if s.orig[i] != want[i] {
					t.Fatalf("frame %d: md5 %s, libvpx %s", i, s.orig[i], want[i])
				}
			}
		})
	}
}

func TestRobustCorruptStreamsMatchLibvpx(t *testing.T) {
	tsv, err := robustStreams.ReadFile("testdata/robust_corrupt.tsv")
	if err != nil {
		t.Fatal(err)
	}
	streams := map[string]*robustStream{}
	type tally struct{ rows, rejected, accepted, garbage, recovered int }
	total := tally{}
	used := map[string]bool{}
	sc := bufio.NewScanner(bytes.NewReader(tsv))
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "#") || sc.Text() == "" {
			continue
		}
		f := strings.Split(sc.Text(), "\t")
		if len(f) != 7 {
			t.Fatalf("malformed row %q", sc.Text())
		}
		stream, label, edit, outOfRange, vpxMsg := f[0], f[1], f[3], f[5] == "R", f[6]
		k, err := strconv.Atoi(f[2])
		if err != nil {
			t.Fatal(err)
		}
		verdicts := strings.Split(f[4], ",")
		s := streams[stream]
		if s == nil {
			s = loadRobustStream(t, stream)
			streams[stream] = s
		}
		if k+len(verdicts) > len(s.packets) {
			t.Fatalf("row %q: more verdicts than packets", sc.Text())
		}
		total.rows++
		if testing.Short() && total.rows%8 != 0 {
			continue
		}
		key := fmt.Sprintf("%s/%d/%s", stream, k, edit)
		name := key
		if len(name) > 80 {
			name = name[:80] + "..."
		}
		t.Run(name, func(t *testing.T) {
			undefinedPixels := robustUndefinedPixels[key]
			if undefinedPixels && !outOfRange {
				t.Fatalf("robustUndefinedPixels lists a row that libvpx does not flag as out-of-range")
			}
			// noPixels: pictures are not compared until the next key frame.
			noPixels := undefinedPixels
			mismatch := false
			d := New()
			for i := 0; i < k; i++ {
				if _, err := robustDecodeMD5(d, s.packets[i], s.hbd); err != nil {
					t.Fatalf("uncorrupted packet %d: %v", i, err)
				}
			}
			sawErr := false
			for j, v := range verdicts {
				i := k + j
				p := s.packets[i]
				if j == 0 {
					if v == "S" {
						continue
					}
					p = robustApplyEdit(t, p, edit)
				} else if s.key[i] {
					noPixels = false
					if sawErr && v == "=" {
						total.recovered++
					}
				}
				got, err := robustDecodeMD5(d, p, s.hbd) // a panic fails the test
				what := fmt.Sprintf("packet %d (%s)", i, label)
				if j > 0 {
					what = fmt.Sprintf("packet %d, %d after the corrupted one (%s)", i, j, label)
				}
				switch v {
				case "E":
					sawErr = true
					if err == nil {
						t.Fatalf("%s: decoded without error; libvpx rejects it (%s)", what, vpxMsg)
					}
					if j == 0 {
						total.rejected++
					}
					continue
				case "-":
					// libvpx decoded the packet and had no frame to output.
					if err != nil {
						t.Fatalf("%s: decode error %q; libvpx decodes it (no output)", what, err)
					}
					if got != "-" {
						t.Fatalf("%s: a frame was output; libvpx outputs none", what)
					}
					continue
				}
				if err != nil {
					t.Fatalf("%s: decode error %q; libvpx decodes it", what, err)
				}
				if got == "-" {
					t.Fatalf("%s: no frame output; libvpx outputs one", what)
				}
				want := v
				if v == "=" {
					want = s.orig[i]
				} else if j == 0 {
					total.garbage++
				}
				if j == 0 {
					total.accepted++
				}
				if got != want {
					if !noPixels {
						t.Fatalf("%s: output md5 %s, libvpx %s", what, got, want)
					}
					mismatch = true
				}
			}
			if undefinedPixels {
				used[key] = mismatch
			}
		})
	}
	if !testing.Short() {
		for key := range robustUndefinedPixels {
			if !used[key] {
				t.Errorf("robustUndefinedPixels[%q]: no such row, or its pictures match libvpx's (remove the entry)", key)
			}
		}
	}
	// Guard the table itself.
	t.Logf("%d rows: corrupted packet rejected in %d, decoded in %d (%d of them to a different picture); %d recoveries at a later key frame",
		total.rows, total.rejected, total.accepted, total.garbage, total.recovered)
	if testing.Short() {
		return
	}
	if total.rows < 1000 || total.rejected < 500 || total.accepted < 100 || total.garbage < 30 || total.recovered < 100 {
		t.Fatalf("table coverage too low")
	}
}

// A corrupt inter packet must not poison the decoder for good: it is an
// error, the inter frames that follow are refused, and from the next key
// frame on the output is bit-exact again. (Every row of the table for the
// robust_kf2 streams checks this against libvpx; this is the plain statement.)
func TestRobustRecoveryAtNextKeyFrame(t *testing.T) {
	for _, name := range []string{"robust_kf2_2tile.ivf", "robust_kf2_2tile_p2.ivf"} {
		t.Run(name, func(t *testing.T) {
			s := loadRobustStream(t, name)
			if !s.key[0] || !s.key[6] {
				t.Fatal("expected key frames at packets 0 and 6")
			}
			for _, edit := range []string{"trunc:40", "xor:0:80", "trunc:0", fmt.Sprintf("trunc:%d", len(s.packets[2])/2)} {
				d := New()
				for i, p := range s.packets {
					if i == 2 {
						if _, err := robustDecodeMD5(d, robustApplyEdit(t, p, edit), s.hbd); err == nil {
							t.Fatalf("%s: corrupted packet 2 decoded without error", edit)
						}
						continue
					}
					got, err := robustDecodeMD5(d, p, s.hbd)
					switch {
					case i < 2 || i >= 6:
						if err != nil {
							t.Fatalf("%s: packet %d: %v", edit, i, err)
						}
						if got != s.orig[i] {
							t.Fatalf("%s: packet %d: md5 %s, want %s", edit, i, got, s.orig[i])
						}
					default:
						// Between the error and the key frame libvpx refuses
						// every frame ("Keyframe / intra-only frame required to
						// reset decoder state"), and so does this decoder.
						if err == nil {
							t.Fatalf("%s: inter packet %d after a failed frame decoded without error", edit, i)
						}
					}
				}
			}
		})
	}
}

// robustApplyEdit applies the edit operations of robust_corrupt_gen.py.
func robustApplyEdit(t testing.TB, packet []byte, edit string) []byte {
	t.Helper()
	p := append([]byte(nil), packet...)
	num := func(s string) int {
		n, err := strconv.Atoi(s)
		if err != nil || n < 0 {
			t.Fatalf("bad edit %q", edit)
		}
		return n
	}
	for _, e := range strings.Split(edit, ";") {
		f := strings.Split(e, ":")
		switch {
		case f[0] == "trunc" && len(f) == 2:
			p = p[:num(f[1])]
		case f[0] == "xor" && len(f) == 3:
			m, err := strconv.ParseUint(f[2], 16, 8)
			if err != nil {
				t.Fatalf("bad edit %q", edit)
			}
			p[num(f[1])] ^= byte(m)
		case f[0] == "set" && len(f) == 3:
			v, err := hex.DecodeString(f[2])
			if err != nil {
				t.Fatalf("bad edit %q", edit)
			}
			if copy(p[num(f[1]):], v) != len(v) {
				t.Fatalf("edit %q runs past the packet", edit)
			}
		case f[0] == "fill" && len(f) == 4:
			v, err := strconv.ParseUint(f[3], 16, 8)
			if err != nil {
				t.Fatalf("bad edit %q", edit)
			}
			off, n := num(f[1]), num(f[2])
			for len(p) < off+n {
				p = append(p, 0)
			}
			for i := off; i < off+n; i++ {
				p[i] = byte(v)
			}
		case f[0] == "append" && len(f) == 2:
			v, err := hex.DecodeString(f[1])
			if err != nil {
				t.Fatalf("bad edit %q", edit)
			}
			p = append(p, v...)
		case f[0] == "dup" && len(f) == 1:
			p = append(p, packet...)
		default:
			t.Fatalf("unknown edit %q", edit)
		}
	}
	return p
}

// robustDecodeMD5 decodes one packet and returns the md5 of the output frame
// in the layout of vpxdec --i420 (16-bit little-endian samples when hbd), or
// "-" when the packet decodes without outputting a frame.
// Unlike decodeFrameMD5 it does not recover from panics: a panic on corrupt
// input is a bug and must fail the test.
func robustDecodeMD5(d *Decoder, packet []byte, hbd bool) (string, error) {
	h := md5.New()
	if hbd {
		f, err := d.DecodeFrame16(packet)
		if err != nil {
			return "", err
		}
		if f == nil {
			return "-", nil
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
		plane(f.U, f.UBase(), f.UVStride, (f.Width+1)/2, (f.Height+1)/2)
		plane(f.V, f.VBase(), f.UVStride, (f.Width+1)/2, (f.Height+1)/2)
		return fmt.Sprintf("%x", h.Sum(nil)), nil
	}
	f, err := d.DecodeFrame(packet)
	if err != nil {
		return "", err
	}
	if f == nil {
		return "-", nil
	}
	plane := func(pl []byte, base, stride, w, hgt int) {
		for r := 0; r < hgt; r++ {
			h.Write(pl[base+r*stride : base+r*stride+w])
		}
	}
	plane(f.Y, f.YBase(), f.YStride, f.Width, f.Height)
	plane(f.U, f.UBase(), f.UVStride, (f.Width+1)/2, (f.Height+1)/2)
	plane(f.V, f.VBase(), f.UVStride, (f.Width+1)/2, (f.Height+1)/2)
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}
