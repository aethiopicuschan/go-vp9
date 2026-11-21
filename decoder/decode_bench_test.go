package decoder

import (
	"os"
	"testing"
)

// Whole-stream decode benchmarks on embedded regression streams; run with
//
//	go test ./decoder -run '^$' -bench Decode -benchmem
//
// to track time and allocation per stream (allocation is a large part of the
// cost: every frame allocates its output planes and mode-info grid).
func benchmarkDecode(b *testing.B, name string) {
	data, err := os.ReadFile("testdata/" + name + ".ivf")
	if err != nil {
		b.Fatal(err)
	}
	packets := readIVFPacketsTB(b, data)
	hbd := len(packets) > 0 && packets[0][0]&0x10 != 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d := New()
		for k, p := range packets {
			var err error
			if hbd {
				_, err = d.DecodeFrame16(p)
			} else {
				_, err = d.DecodeFrame(p)
			}
			if err != nil {
				b.Fatalf("packet %d: %v", k, err)
			}
		}
	}
	b.ReportMetric(float64(len(packets)), "packets/op")
}

func BenchmarkDecode640x384(b *testing.B)      { benchmarkDecode(b, "tilerows_r2c1_640x384") }
func BenchmarkDecode320x192(b *testing.B)      { benchmarkDecode(b, "fp0_hpzero_320x192") }
func BenchmarkDecode320x192HBD(b *testing.B)   { benchmarkDecode(b, "fp0_hpzero_320x192_p2") }
func BenchmarkDecode104x72Chroma(b *testing.B) { benchmarkDecode(b, "chroma_444") }
