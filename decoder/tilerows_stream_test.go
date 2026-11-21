package decoder

import (
	"bufio"
	"bytes"
	"embed"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/aethiopicuschan/go-vp9/frame"
)

// Streams with more than one tile row (tile_rows_log2 > 0), pinned to
// libvpx's output.
//
// Tile rows are NOT independent in VP9. libvpx (decode_tiles) zeroes the
// above entropy / partition context once per frame, not per tile row, and
// set_mi_row_col makes the above neighbour available whenever mi_row != 0
// (only the left neighbour is tile-relative). So the first row of blocks of a
// tile row uses the last row of the tile row above for its entropy contexts,
// mode / MV-reference contexts and intra edge pixels. Only tile columns cut
// those dependencies.
//
// Each NAME.ivf is a small libvpx-vp9 encode (ffmpeg, lavfi testsrc2, encoder
// -threads 1: with more threads this libvpx drops tile rows as soon as there
// are tile columns); NAME.md5 holds one md5 per shown frame of the reference
// decode `ffmpeg -c:v libvpx-vp9 -i NAME.ivf -pix_fmt yuv420p[10le] -f
// rawvideo`. One shown frame per IVF packet; hidden alt-ref frames travel
// inside superframes.
//
//   - tilerows_r1c0_fp0: 256x256, 2 tile rows, 1 column, backward adaptation.
//   - tilerows_r2c1_fp1: 512x512, 4 tile rows x 2 columns, frame_parallel=1.
//   - tilerows_r2c1_640x384: 4 tile rows x 2 columns over 6 superblock rows
//     (uneven tile rows: 1, 2, 1, 2 superblock rows), backward adaptation.
//   - tilerows_r1c2_altref: 1024x576, 2 tile rows x 4 columns, 2-pass with
//     hidden alt-refs, backward adaptation.
//   - tilerows_r1c1_p2: 10-bit profile 2, 512x288, 2 tile rows x 2 columns.
//   - tilerows_r2c0_altref_p2: 10-bit profile 2, 320x384, 4 tile rows,
//     2-pass with hidden alt-refs, frame_parallel=1.
//
// Every one of them failed before the fix, on the key frame already:
// "entropy desync (bitstream over-read)" (above context reset / above
// neighbour treated as unavailable at each tile row); with only the entropy
// side fixed they decode but every frame differs (intra edge availability).
// Exact generation commands are in testdata/tilerows_streams.txt.
//
//go:embed testdata/tilerows_*.ivf testdata/tilerows_*.md5 testdata/tilerows_corrupt.tsv
var tileRowStreams embed.FS

func TestTileRowStreams(t *testing.T) {
	streams := []struct {
		name   string
		frames int
		hbd    bool
	}{
		{"tilerows_r1c0_fp0", 4, false},
		{"tilerows_r2c1_fp1", 4, false},
		{"tilerows_r2c1_640x384", 4, false},
		{"tilerows_r1c2_altref", 18, false},
		{"tilerows_r1c1_p2", 4, true},
		{"tilerows_r2c0_altref_p2", 18, true},
	}
	for _, s := range streams {
		t.Run(s.name, func(t *testing.T) {
			data, err := tileRowStreams.ReadFile("testdata/" + s.name + ".ivf")
			if err != nil {
				t.Fatal(err)
			}
			sums, err := tileRowStreams.ReadFile("testdata/" + s.name + ".md5")
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

// Corrupted key frames with several tile rows: the decoder must reach
// libvpx's verdict for every tile of the frame, not just the last one or the
// first tile row.
//
// testdata/tilerows_corrupt.tsv (stream, variant, verdict, md5, libvpx
// message) was recorded by running vpxdec on the variants this test builds
// from packet 0 of two of the streams above (TILEROWS_DUMP=dir makes the test
// write them as IVF files; see testdata/tilerows_streams.txt). The frame is
// [headers] then, per tile in raster order, a 4-byte big-endian size (absent
// for the last tile of the frame) and the tile bytes.
func TestTileRowCorruptTilesMatchLibvpxVerdict(t *testing.T) {
	tsv, err := tileRowStreams.ReadFile("testdata/tilerows_corrupt.tsv")
	if err != nil {
		t.Fatal(err)
	}
	type key struct{ stream, variant string }
	type verdict struct{ verdict, md5, msg string }
	table := map[key]verdict{}
	sc := bufio.NewScanner(bytes.NewReader(tsv))
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "#") || sc.Text() == "" {
			continue
		}
		f := strings.Split(sc.Text(), "\t")
		if len(f) != 5 {
			t.Fatalf("malformed row %q", sc.Text())
		}
		table[key{f[0], f[1]}] = verdict{f[2], f[3], f[4]}
	}

	dump := os.Getenv("TILEROWS_DUMP")
	rows, rejected, accepted := 0, 0, 0
	for _, s := range []struct {
		name  string
		tiles int
		hbd   bool
	}{
		{"tilerows_r2c1_640x384", 8, false},
		{"tilerows_r1c1_p2", 4, true},
	} {
		data, err := tileRowStreams.ReadFile("testdata/" + s.name + ".ivf")
		if err != nil {
			t.Fatal(err)
		}
		pay := readIVFPackets(t, data)[0]
		variants := tileRowCorruptVariants(t, pay, s.tiles)
		for _, v := range variants {
			if dump != "" {
				hdr := append([]byte(nil), data[:32]...)
				binary.LittleEndian.PutUint32(hdr[24:], 1)
				var fh [12]byte
				binary.LittleEndian.PutUint32(fh[:], uint32(len(v.packet)))
				out := append(append(hdr, fh[:]...), v.packet...)
				path := filepath.Join(dump, s.name+"."+v.name+".ivf")
				if err := os.WriteFile(path, out, 0o644); err != nil {
					t.Fatal(err)
				}
				continue
			}
			want, ok := table[key{s.name, v.name}]
			if !ok {
				t.Fatalf("%s/%s: no row in tilerows_corrupt.tsv", s.name, v.name)
			}
			rows++
			t.Run(s.name+"/"+v.name, func(t *testing.T) {
				got, derr := decodeFrameMD5(New(), v.packet, s.hbd)
				if derr != nil && strings.HasPrefix(derr.Error(), "panic") {
					t.Fatal(derr)
				}
				switch {
				case strings.Contains(want.msg, "Invalid frame marker"):
					// libvpx decoded the frame and then rejected the bytes
					// left behind the last tile as a further frame
					// (vp9_dx_iface.c decoder_decode); that under-read rule is
					// not implemented here (see overread_stream_test.go).
					t.Skipf("known divergence: libvpx %s (%s), this decoder err=%v", want.verdict, want.msg, derr)
				case want.verdict == "ERR":
					if derr == nil {
						t.Fatalf("decoded without error; libvpx rejects this frame (%s)", want.msg)
					}
					rejected++
				case want.verdict == "OK":
					if derr != nil {
						t.Fatalf("decode error %q; libvpx decodes this frame", derr)
					}
					if got != want.md5 {
						t.Fatalf("output md5 %s, libvpx %s", got, want.md5)
					}
					accepted++
				default:
					t.Fatalf("bad verdict %q", want.verdict)
				}
			})
		}
	}
	if dump != "" {
		t.Skipf("variants written to %s", dump)
	}
	// Guard the table itself: it must contain both kinds of rows.
	if rows < 80 || rejected < 40 || accepted < 10 {
		t.Fatalf("table coverage: %d rows, %d rejected, %d accepted", rows, rejected, accepted)
	}
}

type tileRowVariant struct {
	name   string
	packet []byte
}

// tileRowCorruptVariants locates the nTiles tiles of a frame and returns
// corrupted copies of it: per tile, the size field rewritten, the
// bool-decoder marker bit set, the tile's second half or last bytes zeroed and
// one bit flipped mid-tile; plus the packet cut short.
func tileRowCorruptVariants(t *testing.T, pay []byte, nTiles int) []tileRowVariant {
	t.Helper()
	h, err := frame.Parse(pay, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if n := 1 << uint(h.TileColsLog2+h.TileRowsLog2); n != nTiles {
		t.Fatalf("%d tiles in the frame header, want %d", n, nTiles)
	}
	// start[i], size[i]: tile i's bytes; the size field (if any) is the 4
	// bytes before start[i].
	start := make([]int, nTiles)
	size := make([]int, nTiles)
	pos := h.Size + h.CompressedHeaderSize
	for i := 0; i < nTiles; i++ {
		if i == nTiles-1 {
			start[i], size[i] = pos, len(pay)-pos
			break
		}
		if pos+4 > len(pay) {
			t.Fatalf("tile %d: size field outside the packet", i)
		}
		sz := int(binary.BigEndian.Uint32(pay[pos:]))
		start[i], size[i] = pos+4, sz
		pos += 4 + sz
		if pos > len(pay) {
			t.Fatalf("tile %d: outside the packet", i)
		}
	}
	if size[nTiles-1] <= 0 {
		t.Fatal("empty last tile")
	}

	var out []tileRowVariant
	add := func(name string, edit func(p []byte) []byte) {
		out = append(out, tileRowVariant{name, edit(append([]byte(nil), pay...))})
	}
	add("orig", func(p []byte) []byte { return p })
	for i := 0; i < nTiles; i++ {
		if i < nTiles-1 {
			rest := len(pay) - start[i]
			for _, v := range []struct {
				tag string
				sz  int64
			}{
				{"zero", 0}, {"minus1", int64(size[i]) - 1}, {"plus1", int64(size[i]) + 1},
				{"half", int64(size[i]) / 2}, {"rest", int64(rest)}, {"rest_plus1", int64(rest) + 1},
				{"huge", 0xffffffff},
			} {
				add(fmt.Sprintf("size_%s_t%d", v.tag, i), func(p []byte) []byte {
					binary.BigEndian.PutUint32(p[start[i]-4:], uint32(v.sz))
					return p
				})
			}
		}
		add("marker_t"+strconv.Itoa(i), func(p []byte) []byte {
			p[start[i]] |= 0x80
			return p
		})
		add("zero_half_t"+strconv.Itoa(i), func(p []byte) []byte {
			for j := start[i] + size[i]/2; j < start[i]+size[i]; j++ {
				p[j] = 0
			}
			return p
		})
		for _, k := range []int{1, 2, 3} {
			add(fmt.Sprintf("zero_tail%d_t%d", k, i), func(p []byte) []byte {
				for j := start[i] + size[i] - k; j < start[i]+size[i]; j++ {
					p[j] = 0
				}
				return p
			})
		}
		add("flip_mid_t"+strconv.Itoa(i), func(p []byte) []byte {
			p[start[i]+size[i]/2] ^= 0x10
			return p
		})
	}
	last := size[nTiles-1]
	for _, k := range []int{1, 2, 4, 8, last / 2, last - 1, last, last + 1, last + 4, last + 5, last + 4 + size[nTiles-2]/2} {
		add("trunc_"+strconv.Itoa(k), func(p []byte) []byte { return p[:len(p)-k] })
	}
	return out
}
