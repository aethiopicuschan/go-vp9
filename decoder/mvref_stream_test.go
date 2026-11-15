package decoder

import (
	"crypto/sha256"
	"embed"
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/aethiopicuschan/go-vp9/refbuf"
)

// These two-frame IVF inputs were encoded with libvpx. Their second frame
// reaches sub-8x8 block 0 with no MV candidates in both decode paths.
//
//go:embed testdata/sub8x8_profile0.ivf testdata/sub8x8_profile2.ivf
var sub8x8Streams embed.FS

func TestSub8x8NoMVRefsStreams(t *testing.T) {
	tests := []struct {
		name   string
		decode func(*Decoder, []byte) ([3]string, error)
		want   [][3]string
	}{
		{
			name: "sub8x8_profile0.ivf",
			decode: func(d *Decoder, packet []byte) ([3]string, error) {
				f, err := d.DecodeFrame(packet)
				if err != nil {
					return [3]string{}, err
				}
				return hashVisibleFrame8(f), nil
			},
			want: [][3]string{
				{"b45bb84fbd4a313560dfd2139e9ac286c122f98671d50579ff5b500a2bdf6801", "c3cb5323d5ea1d5fb0b2693e2c2b34e946cb9d4280cbac16a7b6dd65aa2be7a3", "e308be72e67d9be850fa22d4f454dd456503570f523166fc7d269fe001106f85"},
				{"bf9114a8b9e872ef90245a5fed3774b0ead0b785b261e7ee6c78748644cf97af", "5b7fb83b97392634c60bbd89980f4bba0770b828aa41c598c43d0f75feea709a", "fc5b91f03fc5b38a984220ef7cb746ac24fcdd5f4ef826e243a0bfb2c203b5c7"},
			},
		},
		{
			name: "sub8x8_profile2.ivf",
			decode: func(d *Decoder, packet []byte) ([3]string, error) {
				f, err := d.DecodeFrame16(packet)
				if err != nil {
					return [3]string{}, err
				}
				return hashVisibleFrame16(f), nil
			},
			want: [][3]string{
				{"4df736d29c5b2915b9a09efb9b721eec04d96533bda6a63d2c57ddb735304d2d", "2150a835b42f577b35f93318a4006287aac5fb28dbf544551b40e56e7fb75bbf", "21c5beb719364c61d60e2fcee36a293716238805681ba858e67f0a1affa9cb71"},
				{"29a8eb5ff3dac838baa70a4679b826354fe25a8f0834e1a6c11763a1e5a72927", "03e1f703a03e9a28ced3684a150ee885a1a7b65ecef3d6f880e9bce3a571869c", "2450a0cdbea524d4d0bcd3b71434904ca7fff28b4a991ed7a6d536fa03e88e3b"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := sub8x8Streams.ReadFile("testdata/" + tt.name)
			if err != nil {
				t.Fatal(err)
			}
			packets := readIVFPackets(t, data)
			if len(packets) != len(tt.want) {
				t.Fatalf("stream has %d frames, want %d", len(packets), len(tt.want))
			}
			d := New()
			for i, packet := range packets {
				got, err := tt.decode(d, packet)
				if err != nil {
					t.Fatalf("frame %d decode: %v", i, err)
				}
				if got != tt.want[i] {
					t.Errorf("frame %d Y/U/V hashes = %v, want %v", i, got, tt.want[i])
				}
			}
		})
	}
}

func readIVFPackets(t *testing.T, data []byte) [][]byte {
	t.Helper()
	if len(data) < 32 || string(data[:4]) != "DKIF" {
		t.Fatal("embedded stream is not IVF")
	}
	var packets [][]byte
	for off := 32; off < len(data); {
		if off+12 > len(data) {
			t.Fatalf("truncated IVF frame header at byte %d", off)
		}
		size := int(binary.LittleEndian.Uint32(data[off:]))
		off += 12
		if size > len(data)-off {
			t.Fatalf("truncated IVF frame payload at byte %d", off)
		}
		packets = append(packets, data[off:off+size])
		off += size
	}
	return packets
}

func hashVisibleFrame8(f *refbuf.Frame) [3]string {
	return [3]string{
		hashVisiblePlane8(f.Y, f.YBase(), f.YStride, f.Width, f.Height),
		hashVisiblePlane8(f.U, f.UBase(), f.UVStride, f.Width/2, f.Height/2),
		hashVisiblePlane8(f.V, f.VBase(), f.UVStride, f.Width/2, f.Height/2),
	}
}

func hashVisiblePlane8(plane []byte, base, stride, width, height int) string {
	h := sha256.New()
	for row := 0; row < height; row++ {
		_, _ = h.Write(plane[base+row*stride : base+row*stride+width])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func hashVisibleFrame16(f *refbuf.Frame16) [3]string {
	return [3]string{
		hashVisiblePlane16(f.Y, f.YBase(), f.YStride, f.Width, f.Height),
		hashVisiblePlane16(f.U, f.UBase(), f.UVStride, f.Width/2, f.Height/2),
		hashVisiblePlane16(f.V, f.VBase(), f.UVStride, f.Width/2, f.Height/2),
	}
}

func hashVisiblePlane16(plane []uint16, base, stride, width, height int) string {
	h := sha256.New()
	var sample [2]byte
	for row := 0; row < height; row++ {
		for _, pixel := range plane[base+row*stride : base+row*stride+width] {
			binary.LittleEndian.PutUint16(sample[:], pixel)
			_, _ = h.Write(sample[:])
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}
