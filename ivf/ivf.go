// Package ivf reads the IVF container format used by libvpx test vectors.
// IVF is a trivial wrapper: a 32-byte file header followed by 12-byte frame
// headers each preceding a raw compressed frame (VP8, VP9, AV1, …).
// Spec: https://wiki.multimedia.cx/index.php/IVF
package ivf

import (
	"encoding/binary"
	"errors"
	"io"
)

// FileHeader is the 32-byte IVF file header.
type FileHeader struct {
	FourCC      [4]byte // codec identifier, e.g. "VP90"
	Width       uint16
	Height      uint16
	TimebaseNum uint32 // timebase numerator
	TimebaseDen uint32 // timebase denominator
	NumFrames   uint32
}

// FrameHeader is the 12-byte per-frame header.
type FrameHeader struct {
	Size      uint32
	Timestamp uint64
}

// Reader reads IVF frames sequentially from an io.Reader.
type Reader struct {
	r      io.Reader
	Header FileHeader
}

var errMagic = errors.New("ivf: not an IVF file (bad magic)")
var errVersion = errors.New("ivf: unsupported IVF version")

// NewReader reads and validates the IVF file header, then returns a Reader
// positioned at the first frame.
func NewReader(r io.Reader) (*Reader, error) {
	var raw [32]byte
	if _, err := io.ReadFull(r, raw[:]); err != nil {
		return nil, err
	}
	if string(raw[0:4]) != "DKIF" {
		return nil, errMagic
	}
	if binary.LittleEndian.Uint16(raw[4:6]) != 0 {
		return nil, errVersion
	}
	rd := &Reader{r: r}
	copy(rd.Header.FourCC[:], raw[8:12])
	rd.Header.Width = binary.LittleEndian.Uint16(raw[12:14])
	rd.Header.Height = binary.LittleEndian.Uint16(raw[14:16])
	rd.Header.TimebaseNum = binary.LittleEndian.Uint32(raw[16:20])
	rd.Header.TimebaseDen = binary.LittleEndian.Uint32(raw[20:24])
	rd.Header.NumFrames = binary.LittleEndian.Uint32(raw[24:28])
	return rd, nil
}

// ErrEOF is returned when there are no more frames.
var ErrEOF = io.EOF

// Next reads the next frame and returns its header and payload.
// Returns (zero, nil, io.EOF) when the stream is exhausted.
func (rd *Reader) Next() (FrameHeader, []byte, error) {
	var hdr [12]byte
	_, err := io.ReadFull(rd.r, hdr[:])
	if err == io.EOF || err == io.ErrUnexpectedEOF {
		return FrameHeader{}, nil, io.EOF
	}
	if err != nil {
		return FrameHeader{}, nil, err
	}
	fh := FrameHeader{
		Size:      binary.LittleEndian.Uint32(hdr[0:4]),
		Timestamp: binary.LittleEndian.Uint64(hdr[4:12]),
	}
	payload := make([]byte, fh.Size)
	if _, err := io.ReadFull(rd.r, payload); err != nil {
		return FrameHeader{}, nil, errors.New("ivf: truncated frame payload")
	}
	return fh, payload, nil
}

// WriteFile serialises frames into IVF format (fourcc e.g. "VP90").
// Useful for constructing test vectors programmatically.
func WriteFile(w io.Writer, fourcc string, width, height uint16, frames [][]byte) error {
	hdr := make([]byte, 32)
	copy(hdr[0:4], "DKIF")
	binary.LittleEndian.PutUint16(hdr[4:6], 0)  // version
	binary.LittleEndian.PutUint16(hdr[6:8], 32) // header size
	copy(hdr[8:12], fourcc)
	binary.LittleEndian.PutUint16(hdr[12:14], width)
	binary.LittleEndian.PutUint16(hdr[14:16], height)
	binary.LittleEndian.PutUint32(hdr[16:20], 1)  // timebase num
	binary.LittleEndian.PutUint32(hdr[20:24], 30) // timebase den (30 fps)
	binary.LittleEndian.PutUint32(hdr[24:28], uint32(len(frames)))
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	for i, f := range frames {
		fh := make([]byte, 12)
		binary.LittleEndian.PutUint32(fh[0:4], uint32(len(f)))
		binary.LittleEndian.PutUint64(fh[4:12], uint64(i))
		if _, err := w.Write(fh); err != nil {
			return err
		}
		if _, err := w.Write(f); err != nil {
			return err
		}
	}
	return nil
}
