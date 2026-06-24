package boolcoder

// Writer is a VP9 boolean arithmetic encoder. It exists so the decoder can be
// roundtrip-tested bit-exactly (encode a sequence, decode it back, compare) with
// no external dependency. It implements the same split/renormalise arithmetic as
// the decoder, with standard carry-propagating byte emission.
type Writer struct {
	out      []byte
	low      uint32 // 24 active bits + carry headroom
	rng      uint32 // 128..255 after renorm
	count    int    // bits buffered beyond a byte boundary
	finished bool
}

// NewWriter returns an empty encoder.
func NewWriter() *Writer { return &Writer{rng: 255, count: -24} }

// WriteBool encodes one boolean with probability prob (1..255 = P(0)*256).
func (w *Writer) WriteBool(prob uint8, bit int) {
	split := 1 + (((w.rng - 1) * uint32(prob)) >> 8)
	rng := split
	if bit != 0 {
		w.low += split
		rng = w.rng - split
	}
	shift := int(norm8[rng])
	rng <<= shift
	w.count += shift
	if w.count >= 0 {
		offset := shift - w.count
		if w.low<<(offset-1)&0x80000000 != 0 { // carry into already-emitted bytes
			x := len(w.out) - 1
			for x >= 0 && w.out[x] == 0xff {
				w.out[x] = 0
				x--
			}
			w.out[x]++
		}
		w.out = append(w.out, byte((w.low>>(24-offset))&0xff))
		w.low = (w.low << offset) & 0xffffff
		shift = w.count
		w.count -= 8
	}
	w.low <<= shift
	w.rng = rng
}

// WriteBit encodes one equiprobable bit.
func (w *Writer) WriteBit(bit int) { w.WriteBool(128, bit) }

// WriteLiteral encodes an n-bit value MSB-first (equiprobable bits).
func (w *Writer) WriteLiteral(v uint32, n int) {
	for i := n - 1; i >= 0; i-- {
		w.WriteBit(int((v >> i) & 1))
	}
}

// Bytes flushes the coder and returns the encoded buffer.
func (w *Writer) Bytes() []byte {
	if !w.finished {
		for i := 0; i < 32; i++ { // flush trailing bits
			w.WriteBit(0)
		}
		w.finished = true
	}
	return w.out
}
