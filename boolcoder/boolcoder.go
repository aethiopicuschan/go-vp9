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
	bitsAvl int    // real (buffer) bits currently held in value, top byte included; negative = zero-fill bits already shifted out
	marker  bool   // NewReaderVP9: the leading marker bit was 1 (libvpx init error)

	// Shadow of libvpx's lazy fill schedule (vpx_reader.count / .buffer), kept
	// only so FindEnd can reproduce vpx_reader_find_end. vpxCount is libvpx's
	// `count` while data remains; vpxPos its `buffer` as an offset into buf.
	// Once a fill has loaded the last byte (libvpx adds LOTS_OF_BITS to count)
	// vpxCount is parked at vpxExhausted and never consulted again.
	vpxCount int
	vpxPos   int

	// optional tracing (debug only)
	traceN     int
	traceF     TraceFunc
	traceLabel string
}

// NewReader initialises a boolean decoder over buf.
func NewReader(buf []byte) *Reader {
	r := &Reader{buf: buf, rng: 255, vpxCount: -8}
	r.fill()
	r.vpxFill()
	return r
}

// vpxExhausted parks vpxCount after libvpx's fill has consumed the whole
// buffer (stands in for count + LOTS_OF_BITS; any value that renormalisation
// shifts cannot bring below zero or into find_end's rewind range works).
const vpxExhausted = 1 << 62

// vpxFill mirrors the bookkeeping of libvpx vpx_reader_fill (no data is moved,
// the real window is maintained by fill): called when count < 0, it loads as
// many whole bytes as fit into the 64-bit register, or everything that is left.
func (r *Reader) vpxFill() {
	shift := 48 - r.vpxCount // BD_VALUE_SIZE - CHAR_BIT - (count + CHAR_BIT)
	left := len(r.buf) - r.vpxPos
	if left <= 8 && shift+8 >= left*8 {
		r.vpxPos = len(r.buf)
		r.vpxCount = vpxExhausted
		return
	}
	n := shift>>3 + 1
	r.vpxPos += n
	r.vpxCount += 8 * n
}

// FindEnd returns the offset in the reader's buffer at which libvpx considers
// the coded data to end: vpx_reader_find_end (vpx_dsp/bitreader.c),
//
//	while (r->count > CHAR_BIT && r->count < BD_VALUE_SIZE) {
//	  r->count -= CHAR_BIT;
//	  r->buffer--;
//	}
//	return r->buffer;
//
// i.e. the read pointer with the look-ahead bytes that were loaded into the
// register but not needed given back. The VP9 decoder uses the value of the
// frame's last tile as the end of the frame (vp9_decodeframe.c decode_tiles);
// vp9_dx_iface.c then treats what follows in the packet as another frame.
// When the register has already swallowed the last byte of the buffer the
// result is len(buf), whatever was consumed. The result depends on libvpx's
// lazy refill schedule (a refill happens at the start of the read that finds
// count < 0), which is why it is tracked rather than derived from BitPos.
func (r *Reader) FindEnd() int {
	c, p := r.vpxCount, r.vpxPos
	for c > 8 && c < 64 {
		c -= 8
		p--
	}
	return p
}

// NewReaderVP9 initialises a boolean decoder over a real VP9 bitstream partition
// (the compressed header or a tile). VP9's init_bool consumes one leading marker
// bit after loading the value window (libvpx vpx_reader_init ends with
// vpx_read_bit) — this constructor reads and discards it so subsequent reads
// align with the stream. Use NewReader (no marker) for the encoder roundtrip.
func NewReaderVP9(buf []byte) *Reader {
	r := NewReader(buf)
	r.marker = r.ReadBit() != 0 // marker bit (always 0 in a valid stream)
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
	if r.vpxCount < 0 {
		r.vpxFill()
	}
	split := 1 + (((r.rng - 1) * uint32(prob)) >> 8)
	bigSplit := uint64(split) << 56
	rngBefore := r.rng
	topBefore := uint8(r.value >> 56)
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
	r.vpxCount -= int(shift)
	if r.bitsAvl <= 56 {
		r.fill()
	}
	if r.traceF != nil && r.traceN > 0 {
		lbl := r.traceLabel
		r.traceLabel = ""
		r.traceN--
		r.traceF(lbl, prob, rngBefore, topBefore, split, bit)
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

// State returns the current range and top byte of value for debugging.
func (r *Reader) State() (rng uint32, topByte uint8) {
	return r.rng, uint8(r.value >> 56)
}

// BitPos returns the number of bits shifted out of the value register so far
// (renormalisation shifts; a symbol may consume zero bits). It keeps counting
// past the end of the buffer, where the register is fed with zeros.
func (r *Reader) BitPos() int {
	return r.pos*8 - r.bitsAvl
}

// overreadSlackBits is the number of bits at the tail of a partition that the
// decoder must NOT have shifted out: libvpx requires the 8-bit arithmetic
// window (the top byte of its value register) to consist of real buffer bits
// at all times. See Overread.
const overreadSlackBits = 8

// Overread reports whether the decoder has shifted zero-fill (bits past the
// end of the partition) into its 8-bit arithmetic window. It is the exact
// equivalent of libvpx's vpx_reader_has_error (vpx_dsp/bitreader.h), which the
// VP9 decoder evaluates after every block (vp9_decodeframe.c decode_block:
// "xd->corrupted |= vpx_reader_has_error(r)") and turns into the fatal
// "Failed to decode tile data" / "Decode failed. Frame data is corrupted."
//
// libvpx keeps `count` = (real bits in the value register) - 8 and, on the
// fill that exhausts the buffer, adds LOTS_OF_BITS to it; has_error is
// `count > BD_VALUE_SIZE && count < LOTS_OF_BITS`, i.e. "the buffer is
// exhausted and fewer than 8 real bits remain in the register". Expressed in
// stream position, with S = bits shifted out of the register so far (BitPos)
// and N = 8*len(buf):
//
//	no error  <=>  S <= N-8
//	error     <=>  S >  N-8
//
// So the bits the decoder may consume stop 8 bits short of the end of the
// buffer: reading zeros beyond the buffer into the *look-ahead* part of the
// register is fine (and routine: the register is topped up with zeros as soon
// as the data runs out), but as soon as one zero-fill bit enters the window
// that is compared against the split, the partition is corrupt. A libvpx
// encoder never produces that: vpx_stop_encode flushes 32 padding bits, which
// makes every tile that decodes in sync end at or before the limit (measured
// on libvpx-encoded tiles: 1..6 bits before it, so the limit is tight and the
// "8" must not be approximated). An empty partition is always an error, a
// one-byte partition as soon as a single bit has been consumed.
//
// The condition is monotonic (S only grows), so evaluating it once after the
// last block of a tile gives the same verdict as libvpx's per-block sticky
// flag. (libvpx's own flag would clear again after 2^30 further bits because
// `count` wraps through a second LOTS_OF_BITS fill; that is not reproduced.)
//
// Callers use this after a tile/partition to reject a corrupt or desynced
// frame instead of reconstructing garbage pixels.
func (r *Reader) Overread() bool {
	return r.BitPos() > len(r.buf)*8-overreadSlackBits
}

// MarkerError reports whether the leading marker bit consumed by NewReaderVP9
// was 1. libvpx's vpx_reader_init returns that bit as its error status and the
// VP9 decoder aborts the frame when it is set (setup_token_decoder for tiles,
// read_compressed_header for the compressed header). Always false for readers
// made with NewReader.
func (r *Reader) MarkerError() bool { return r.marker }

// TraceFunc is called for each ReadBool when tracing is active.
// label: caller-supplied context; prob: boolean prob; rng: range before read;
// top: value[63:56] before read; split: threshold; bit: decoded bit (0 or 1).
type TraceFunc func(label string, prob uint8, rng uint32, top uint8, split uint32, bit int)

// TraceN sets the number of subsequent ReadBool calls to trace via fn.
// Set fn=nil to stop tracing.
func (r *Reader) TraceN(n int, fn TraceFunc) {
	r.traceN = n
	r.traceF = fn
}

// traceLabel is the next-op label (set by SetLabel, consumed on ReadBool).
func (r *Reader) SetLabel(s string) { r.traceLabel = s }

// ReadBoolT is ReadBool with mandatory tracing (for explicit call sites).
func (r *Reader) ReadBoolT(prob uint8, label string) int {
	old := r.traceLabel
	r.traceLabel = label
	bit := r.ReadBool(prob)
	r.traceLabel = old
	return bit
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
