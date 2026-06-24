// Package boolcoder implements the VP9 boolean arithmetic coder — the entropy
// layer underneath every VP9 syntax element. This is the decoder (Reader) plus a
// matching encoder (Writer, see encoder.go) so the two can be roundtrip-tested
// bit-exactly without an external reference.
//
// Probabilities are 8-bit: prob = P(bit == 0) scaled to 256 (so 128 ≈ 50/50).
// Original implementation of the standard range/probability arithmetic decoding
// described in the public VP9 specification (§9.2).
package boolcoder

// norm8[r] = how many left shifts bring an 8-bit range r (1..255) up to bit-7 set
// (i.e. >=128). norm8[0] is unused (range is never 0).
var norm8 [256]uint8

func init() {
	for i := 1; i < 256; i++ {
		v, n := i, 0
		for v < 128 {
			v <<= 1
			n++
		}
		norm8[i] = uint8(n)
	}
}

// Reader decodes booleans from a VP9-coded byte slice.
type Reader struct {
	buf     []byte
	pos     int    // next byte index
	value   uint64 // high bits hold the coded window
	rng     uint32 // current range, 128..255 after renorm
	bitsAvl int    // number of valid value bits beyond the top byte boundary
}

// NewReader initialises a boolean decoder over buf.
func NewReader(buf []byte) *Reader {
	r := &Reader{buf: buf, rng: 255}
	r.fill()
	return r
}

// fill tops up value with bytes from the buffer (big-endian, MSB-first).
func (r *Reader) fill() {
	// Keep ~56 bits of window so single reads never run dry mid-renorm.
	for r.bitsAvl <= 56 && r.pos < len(r.buf) {
		r.value |= uint64(r.buf[r.pos]) << (56 - r.bitsAvl)
		r.pos++
		r.bitsAvl += 8
	}
}

// ReadBool decodes one boolean with probability prob (1..255 = P(0)*256).
func (r *Reader) ReadBool(prob uint8) int {
	split := 1 + (((r.rng - 1) * uint32(prob)) >> 8)
	bigSplit := uint64(split) << 56
	var bit int
	if r.value >= bigSplit {
		r.rng -= split
		r.value -= bigSplit
		bit = 1
	} else {
		r.rng = split
	}
	// Renormalise: shift range/value left until range >= 128.
	shift := norm8[r.rng]
	r.rng <<= shift
	r.value <<= shift
	r.bitsAvl -= int(shift)
	if r.bitsAvl <= 56 {
		r.fill()
	}
	return bit
}

// ReadBit decodes one equiprobable bit (prob 128).
func (r *Reader) ReadBit() int { return r.ReadBool(128) }

// ReadLiteral decodes an n-bit unsigned value MSB-first (equiprobable bits).
func (r *Reader) ReadLiteral(n int) uint32 {
	var x uint32
	for i := 0; i < n; i++ {
		x = x<<1 | uint32(r.ReadBit())
	}
	return x
}

// ReadSigned decodes an n-bit magnitude followed by a sign bit (1 = negative).
func (r *Reader) ReadSigned(n int) int32 {
	v := int32(r.ReadLiteral(n))
	if r.ReadBit() == 1 {
		return -v
	}
	return v
}

// ReadTree walks a VP9 token tree: tree is the flattened left/right index array,
// probs are the per-node probabilities. Returns the decoded (negated-leaf) token.
func (r *Reader) ReadTree(tree []int8, probs []uint8) int {
	i := int8(0)
	for {
		i = tree[int(i)+r.ReadBool(probs[i>>1])]
		if i <= 0 {
			return int(-i)
		}
	}
}

// ReadTreeCount is ReadTree with per-node binary count accumulation.
// counts[nodeIdx][bit]++ for each tree node visited.
// Pass nil counts to skip counting (identical to ReadTree).
func (r *Reader) ReadTreeCount(tree []int8, probs []uint8, counts [][2]uint32) int {
	i := int8(0)
	for {
		bit := r.ReadBool(probs[i>>1])
		if counts != nil {
			counts[i>>1][bit]++
		}
		i = tree[int(i)+bit]
		if i <= 0 {
			return int(-i)
		}
	}
}
