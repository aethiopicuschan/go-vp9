package decoder

import (
	"errors"
	"fmt"

	"github.com/aethiopicuschan/go-vp9/frame"
)

// Packet-level and header-level error handling shared by the 8-bit and the
// high-bit-depth paths. It follows libvpx (vp9_dx_iface.c decoder_decode /
// decode_one, vp9_decoder.c vp9_parse_superframe_index, vp9_decodeframe.c
// read_uncompressed_header / vp9_decode_frame) so that a corrupt packet gets
// the verdict libvpx gives it.

var (
	errEmptyPacket     = errors.New("vp9/decoder: empty packet")
	errSuperframeIndex = errors.New("vp9/decoder: corrupt superframe index")
	errSuperframeSize  = errors.New("vp9/decoder: invalid frame size in superframe index")
	errRefSlotEmpty    = errors.New("vp9/decoder: invalid reference frame index (empty slot)")
	errRefFormat       = errors.New("vp9/decoder: reference frame has incompatible color format (bit depth or chroma subsampling)")
	errRefScaled       = errors.New("vp9/decoder: reference frame size differs from the frame size (scaled references are not supported)")
	errHeaderSize      = errors.New("vp9/decoder: invalid header size (compressed header size 0)")
	errHeaderTruncated = errors.New("vp9/decoder: truncated packet or corrupt header length")
	errNoContext       = errors.New("vp9/decoder: uninitialized entropy context")
	errCompHdrMarker   = errors.New("vp9/decoder: corrupt compressed header (bool-decoder marker bit set)")
	errInvalidMV       = errors.New("vp9/decoder: corrupt frame (motion vector out of range)")
	errCompHdrOverread = errors.New("vp9/decoder: corrupt compressed header (bitstream over-read)")
)

// Errors a caller may want to react to (test with errors.Is).
var (
	// ErrNeedResync is returned for every packet that is not a key frame or
	// an intra-only frame while the decoder is waiting for one: after any
	// decode error the references are unusable until such a frame arrives
	// (libvpx need_resync, "Keyframe / intra-only frame required to reset
	// decoder state"). A caller can keep feeding packets, or ask the sender
	// for a key frame.
	ErrNeedResync = errors.New("vp9/decoder: key frame or intra-only frame required to reset decoder state")

	// ErrFrameSize is wrapped by the error for a frame whose width or height
	// exceeds the decoder's limit (SetMaxFrameSize, DefaultMaxFrameSize).
	ErrFrameSize = errors.New("vp9/decoder: frame size outside the allowed range")
)

// DefaultMaxFrameSize is the default limit on frame width and height. libvpx
// has no limit unless it is configured with --size-limit (Chromium and others
// build it with 16384x16384, the value used here); a decoder that must not be
// made to allocate gigabytes by a ten-byte packet needs one.
const DefaultMaxFrameSize = 16384

// SetMaxFrameSize sets the largest frame width and height the decoder accepts
// (DefaultMaxFrameSize each unless changed). A frame whose header announces a
// larger size is rejected before anything is allocated for it.
func (d *Decoder) SetMaxFrameSize(width, height int) {
	d.maxWidth, d.maxHeight = width, height
}

// parseSuperframeIndex is libvpx vp9_parse_superframe_index: when the last
// byte of the packet looks like a superframe marker (0b110xxxxx) the packet
// must end with a well-formed index, whose frame sizes are returned. A packet
// without the marker is a plain packet (sizes == nil).
func parseSuperframeIndex(data []byte) (sizes []int, err error) {
	marker := data[len(data)-1]
	if marker&0xe0 != 0xc0 {
		return nil, nil
	}
	nFrames := int(marker&0x7) + 1
	mag := int((marker>>3)&0x3) + 1
	indexSz := 2 + mag*nFrames
	// Marked as having an index but too short for it, or the matching marker
	// byte at the front of the index is missing: an invalid chunk.
	if len(data) < indexSz || data[len(data)-indexSz] != marker {
		return nil, errSuperframeIndex
	}
	x := data[len(data)-indexSz+1:]
	sizes = make([]int, nFrames)
	for i := range sizes {
		for j := 0; j < mag; j++ {
			sizes[i] |= int(x[i*mag+j]) << (8 * j)
		}
	}
	return sizes, nil
}

// decodePacket is libvpx decoder_decode. one decodes a single coded frame from
// the start of data and returns the offset at which that frame's data ends.
//
// Superframe: every frame gets exactly the bytes its index entry gives it (the
// entries are only checked against the end of the packet, index included, as
// libvpx does).
//
// Plain packet: after a frame, whatever lies behind the end of its last tile
// (boolcoder.Reader.FindEnd) is skipped while it is zero ("suboptimal
// termination by the encoder") and otherwise decoded as a further frame. So
// trailing zero padding is harmless, a hidden frame followed by a shown one
// without an index decodes as two frames, and trailing garbage is an error
// (normally "bad frame marker").
func (d *Decoder) decodePacket(payload []byte, one func(data []byte) (end int, err error)) error {
	if len(payload) == 0 {
		return errEmptyPacket
	}
	sizes, err := parseSuperframeIndex(payload)
	if err != nil {
		return err
	}
	if sizes != nil {
		off := 0
		for _, sz := range sizes {
			if sz > len(payload)-off {
				return errSuperframeSize
			}
			if _, err := d.decodeOne(payload[off:off+sz], one); err != nil {
				return err
			}
			off += sz
		}
		return nil
	}
	for off := 0; off < len(payload); {
		end, err := d.decodeOne(payload[off:], one)
		if err != nil {
			return err
		}
		if end <= 0 || end > len(payload)-off {
			end = len(payload) - off
		}
		off += end
		for off < len(payload) && payload[off] == 0 {
			off++
		}
	}
	return nil
}

// decodeOne is libvpx decode_one: a frame that fails to decode leaves the
// decoder waiting for a key frame or an intra-only frame (need_resync).
func (d *Decoder) decodeOne(data []byte, one func(data []byte) (end int, err error)) (int, error) {
	end, err := one(data)
	if err != nil {
		d.needResync = true
		return 0, err
	}
	return end, nil
}

// refSlots abstracts the 8-bit / 16-bit reference buffers for beginFrame.
type refSlots struct {
	size  func(slot int) (w, h int) // 0, 0 for an empty slot
	clear func()                    // empty all eight slots
	// format returns the chroma subsampling shifts and bit depth of the frame
	// in a slot; depth is the bit depth an inter frame inherits (8 on the
	// 8-bit path, the last intra frame's on the 16-bit path).
	format func(slot int) (ssX, ssY, bitDepth int)
	depth  int
}

// beginFrame applies what libvpx does between parsing the uncompressed
// header and reading the compressed header, in terms of the decoder state and
// in libvpx's order:
//
//   - need_resync: after a failed frame only a key frame or an intra-only frame
//     is accepted; it restarts with an empty reference map
//     (read_uncompressed_header).
//   - an inter frame may only name reference slots that hold a frame
//     ("Invalid reference frame index"), of the frame's own bit depth and
//     chroma format ("Referenced frame has incompatible color format").
//   - vp9_setup_past_independence: intra and error-resilient frames reset the
//     saved frame contexts (resetFrameContexts). This precedes the remaining
//     header errors, so a frame rejected by them has still reset the contexts.
//   - first_partition_size must be non-zero and lie inside the packet
//     ("Invalid header size", "Truncated packet or corrupt header length").
//
// The frame context itself is checked later (frameContextIndex).
//
// Two restrictions of this decoder are reported here as errors too, before
// any allocation: a frame larger than the size limit (checked after the
// context reset), and references whose
// size differs from the frame's (libvpx scales them).
func (d *Decoder) beginFrame(h *frame.Header, payloadLen int, refs refSlots) error {
	if err := d.checkFrameState(h, refs); err != nil {
		return err
	}
	d.resetFrameContexts(h)
	// The size limit is this decoder's own restriction. libvpx would go on
	// with such a frame (and, for the corrupt headers that typically announce
	// such sizes, fail further down), so it is applied only after the state
	// changes above.
	maxW, maxH := d.maxWidth, d.maxHeight
	if maxW <= 0 {
		maxW = DefaultMaxFrameSize
	}
	if maxH <= 0 {
		maxH = DefaultMaxFrameSize
	}
	if h.FrameWidth <= 0 || h.FrameHeight <= 0 || h.FrameWidth > maxW || h.FrameHeight > maxH {
		return fmt.Errorf("%w: frame size %dx%d outside the allowed %dx%d", ErrFrameSize, h.FrameWidth, h.FrameHeight, maxW, maxH)
	}
	if h.CompressedHeaderSize == 0 {
		return errHeaderSize
	}
	if h.Size+h.CompressedHeaderSize > payloadLen {
		return errHeaderTruncated
	}
	return nil
}

// rejectedHeader handles a header frame.ParseRefs rejected. When the error
// lies behind frame_context_idx (h != nil: truncation in the later fields, or
// too many tile columns) libvpx has made the same state checks and reset the
// frame contexts before failing; mirror that so the frames that follow start
// from the same contexts.
func (d *Decoder) rejectedHeader(h *frame.Header, refs refSlots) {
	if h == nil || h.ShowExistingFrame {
		return
	}
	if d.checkFrameState(h, refs) == nil {
		d.resetFrameContexts(h)
	}
}

// checkFrameState is the first part of beginFrame: the resync state and the
// reference slots.
func (d *Decoder) checkFrameState(h *frame.Header, refs refSlots) error {
	frameIsIntra := h.FrameType == frame.FrameKey || h.IntraOnly
	if frameIsIntra {
		if d.needResync {
			refs.clear()
			d.needResync = false
		}
	} else {
		if d.needResync {
			return ErrNeedResync
		}
		for _, slot := range h.RefFrameIdx {
			w, hgt := refs.size(slot)
			if w <= 0 || hgt <= 0 {
				return errRefSlotEmpty
			}
			if w != h.FrameWidth || hgt != h.FrameHeight {
				return errRefScaled
			}
		}
		// libvpx setup_frame_size_with_refs: every reference must have the
		// frame's bit depth and chroma subsampling ("Referenced frame has
		// incompatible color format"). An inter frame inherits both from the
		// last key or intra-only frame, which need not have refreshed all the
		// slots this frame names.
		for _, slot := range h.RefFrameIdx {
			sx, sy, bd := refs.format(slot)
			if sx != d.ssX || sy != d.ssY || bd != refs.depth {
				return errRefFormat
			}
		}
	}
	return nil
}

// resetFrameContexts is the frame-context part of libvpx
// vp9_setup_past_independence, run for intra and error-resilient frames as
// soon as frame_context_idx has been read: all contexts are reset to the
// defaults (key frame, error-resilient, reset_frame_context 3) or just the
// selected one (reset_frame_context 2).
func (d *Decoder) resetFrameContexts(h *frame.Header) {
	if h.FrameType != frame.FrameKey && !h.IntraOnly && !h.ErrorResilient {
		return
	}
	if h.FrameType == frame.FrameKey || h.ErrorResilient || h.ResetFrameContext == 3 {
		for i := range d.frameContexts {
			d.resetFrameContext(i)
		}
	} else if h.ResetFrameContext == 2 {
		d.resetFrameContext(h.FrameContextIdx)
	}
}

// frameContextIndex returns the frame context a frame works with: intra and
// error-resilient frames always use context 0, whatever frame_context_idx
// says (vp9_setup_past_independence; the coded index only selects the context
// reset_frame_context 2 resets, see resetFrameContexts). The context must have
// been initialised ("Uninitialized entropy context"): an intra-only frame that
// does not reset the contexts cannot be the first frame of a stream.
func (d *Decoder) frameContextIndex(h *frame.Header) (int, error) {
	idx := h.FrameContextIdx
	if h.FrameType == frame.FrameKey || h.IntraOnly || h.ErrorResilient {
		idx = 0
	}
	if !d.fcInit[idx] {
		return 0, errNoContext
	}
	return idx, nil
}
