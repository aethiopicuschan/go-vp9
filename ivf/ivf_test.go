package ivf

import (
	"bytes"
	"io"
	"testing"
)

func TestRoundtrip(t *testing.T) {
	payloads := [][]byte{
		{0x01, 0x02, 0x03},
		bytes.Repeat([]byte{0xab}, 300),
		{},
		bytes.Repeat([]byte{0xff}, 65536),
	}

	var buf bytes.Buffer
	if err := WriteFile(&buf, "VP90", 320, 240, payloads); err != nil {
		t.Fatal(err)
	}

	rd, err := NewReader(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if string(rd.Header.FourCC[:]) != "VP90" {
		t.Fatalf("fourcc = %q", rd.Header.FourCC)
	}
	if rd.Header.Width != 320 || rd.Header.Height != 240 {
		t.Fatalf("dims = %dx%d", rd.Header.Width, rd.Header.Height)
	}
	if rd.Header.NumFrames != uint32(len(payloads)) {
		t.Fatalf("num_frames = %d", rd.Header.NumFrames)
	}

	for i, want := range payloads {
		fh, got, err := rd.Next()
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if fh.Timestamp != uint64(i) {
			t.Fatalf("frame %d: timestamp = %d", i, fh.Timestamp)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("frame %d: payload mismatch (len got=%d want=%d)", i, len(got), len(want))
		}
	}

	_, _, err = rd.Next()
	if err != io.EOF {
		t.Fatalf("expected EOF, got %v", err)
	}
}

func TestBadMagic(t *testing.T) {
	buf := bytes.Repeat([]byte{0}, 32)
	copy(buf, "XXXX")
	_, err := NewReader(bytes.NewReader(buf))
	if err == nil {
		t.Fatal("expected error for bad magic")
	}
}

func TestTruncatedFrameHeader(t *testing.T) {
	var buf bytes.Buffer
	_ = WriteFile(&buf, "VP90", 1, 1, [][]byte{{1, 2, 3}})
	data := buf.Bytes()

	// Cut inside the 12-byte frame header (at byte 37 = 32 file hdr + 5 frame hdr bytes).
	rd, _ := NewReader(bytes.NewReader(data[:37]))
	_, _, err := rd.Next()
	if err != io.EOF {
		t.Fatalf("truncated frame header: expected EOF, got %v", err)
	}

	// Cut inside the payload — should be a non-nil, non-EOF error.
	rd2, _ := NewReader(bytes.NewReader(data[:len(data)-1]))
	_, _, err = rd2.Next()
	if err == nil || err == io.EOF {
		t.Fatalf("truncated payload: expected error, got %v", err)
	}
}
