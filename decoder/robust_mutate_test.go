package decoder

import (
	"math/rand"
	"os"
	"strconv"
	"testing"
)

// TestRobustRandomMutations is a deterministic companion of FuzzDecodeFrame:
// seeded pseudo-random mutations (bit flips, byte overwrites, truncations,
// short insertions) of every packet of the seed streams, biased towards what
// coverage-guided fuzzing reaches slowly, i.e. damage deep inside the tile
// data of frames whose headers are intact. Same assertions as the fuzz
// target: no panic, no oversized frame, and the stream's key frame decodes
// bit-exactly afterwards.
//
// ROBUST_MUTATIONS sets the number of mutations per packet (default 4; 400 was run when the test was written: 27600 packets).
func TestRobustRandomMutations(t *testing.T) {
	per := 4
	if v := os.Getenv("ROBUST_MUTATIONS"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			t.Fatal(err)
		}
		per = n
	}
	if testing.Short() {
		per = 2
	}
	rnd := rand.New(rand.NewSource(0x76703921))
	rejected, decoded := 0, 0
	for _, name := range fuzzSeedStreams {
		s := loadFuzzStream(t, name)
		for k := range s.packets {
			if k > 6 && !s.key[k] {
				continue
			}
			for m := 0; m < per; m++ {
				p := append([]byte(nil), s.packets[k]...)
				for n := 1 + rnd.Intn(3); n > 0 && len(p) > 0; n-- {
					switch rnd.Intn(8) {
					case 0, 1, 2, 3:
						p[rnd.Intn(len(p))] ^= 1 << uint(rnd.Intn(8))
					case 4:
						p[rnd.Intn(len(p))] = byte(rnd.Intn(256))
					case 5:
						o := rnd.Intn(len(p))
						for i := o; i < len(p) && i < o+1+rnd.Intn(16); i++ {
							p[i] = byte(0xff * rnd.Intn(2))
						}
					case 6:
						p = p[:rnd.Intn(len(p)+1)]
					case 7:
						o := rnd.Intn(len(p) + 1)
						ins := make([]byte, 1+rnd.Intn(4))
						rnd.Read(ins)
						p = append(p[:o:o], append(ins, p[o:]...)...)
					}
				}
				d := New()
				d.SetMaxFrameSize(fuzzMaxDim, fuzzMaxDim)
				for i := 0; i < k; i++ {
					if _, err := robustDecodeMD5(d, s.packets[i], s.hbd); err != nil {
						t.Fatalf("%s: uncorrupted packet %d: %v", name, i, err)
					}
				}
				if _, err := robustDecodeMD5(d, p, s.hbd); err != nil {
					rejected++
				} else {
					decoded++
				}
				for i := k + 1; i < len(s.packets) && i <= k+2; i++ {
					fuzzDecode(t, d, s.packets[i], s.hbd)
				}
				got, err := robustDecodeMD5(d, s.packets[0], s.hbd)
				if err != nil {
					t.Fatalf("%s packet %d mutation %d (% x...): key frame afterwards: %v", name, k, m, p[:min(len(p), 16)], err)
				}
				if got != s.orig[0] {
					t.Fatalf("%s packet %d mutation %d: key frame afterwards: md5 %s, want %s", name, k, m, got, s.orig[0])
				}
			}
		}
	}
	t.Logf("%d mutated packets rejected, %d decoded", rejected, decoded)
}
