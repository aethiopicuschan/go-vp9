package tile

import (
	"encoding/binary"
	"testing"
)

// buildFrame assembles a fake frame payload with the given tile payloads.
// uncompHdrBytes and compHdrBytes bytes of zeros precede the tile data.
func buildFrame(uncompHdrBytes, compHdrBytes int, tileColsLog2, tileRowsLog2 int, tilePayloads [][]byte) []byte {
	numTiles := (1 << uint(tileColsLog2)) * (1 << uint(tileRowsLog2))
	if len(tilePayloads) != numTiles {
		panic("tile count mismatch")
	}

	hdr := make([]byte, uncompHdrBytes+compHdrBytes)

	var tileData []byte
	for i, tp := range tilePayloads {
		last := i == numTiles-1
		if !last {
			// 4-byte big-endian tile_size (exact byte count)
			var szBuf [4]byte
			binary.BigEndian.PutUint32(szBuf[:], uint32(len(tp)))
			tileData = append(tileData, szBuf[:]...)
		}
		tileData = append(tileData, tp...)
	}
	return append(hdr, tileData...)
}

func TestSingleTile(t *testing.T) {
	tile0 := []byte{0xAA, 0xBB, 0xCC}
	payload := buildFrame(10, 5, 0, 0, [][]byte{tile0})

	r, err := NewReader(payload, 10, 5, 0, 0)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if r.NumCols() != 1 || r.NumRows() != 1 {
		t.Fatalf("dims = %dx%d", r.NumCols(), r.NumRows())
	}
	got := r.TileData(0, 0)
	if string(got) != string(tile0) {
		t.Fatalf("tile(0,0) = %v, want %v", got, tile0)
	}
}

func TestMultiTile2x1(t *testing.T) {
	// 2 tile columns, 1 tile row
	tile0 := []byte{0x11, 0x22, 0x33}
	tile1 := []byte{0x44, 0x55}
	payload := buildFrame(8, 3, 1, 0, [][]byte{tile0, tile1})

	r, err := NewReader(payload, 8, 3, 1, 0)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if r.NumCols() != 2 || r.NumRows() != 1 {
		t.Fatalf("dims = %dx%d", r.NumCols(), r.NumRows())
	}
	if string(r.TileData(0, 0)) != string(tile0) {
		t.Fatalf("tile(0,0) mismatch: %v", r.TileData(0, 0))
	}
	if string(r.TileData(0, 1)) != string(tile1) {
		t.Fatalf("tile(0,1) mismatch: %v", r.TileData(0, 1))
	}
}

func TestMultiTile2x2(t *testing.T) {
	// 2 tile cols × 2 tile rows = 4 tiles, row-major order
	tiles := [][]byte{
		{0x01},
		{0x02, 0x03},
		{0x04, 0x05, 0x06},
		{0x07},
	}
	payload := buildFrame(4, 4, 1, 1, tiles)

	r, err := NewReader(payload, 4, 4, 1, 1)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	for row := 0; row < 2; row++ {
		for col := 0; col < 2; col++ {
			want := tiles[row*2+col]
			got := r.TileData(row, col)
			if string(got) != string(want) {
				t.Errorf("tile(%d,%d) = %v, want %v", row, col, got, want)
			}
		}
	}
}

func TestTruncatedSizeField(t *testing.T) {
	// 2 tiles but only 3 bytes of tile data (need ≥4 for the size prefix).
	// Treated as a "copy-from-reference" frame: tiles are nil, no error.
	payload := make([]byte, 10+3)
	r, err := NewReader(payload, 10, 0, 1, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.TileData(0, 0) != nil {
		t.Fatal("expected nil tile data when size field is truncated")
	}
}

func TestTruncatedTilePayload(t *testing.T) {
	// size field claims 100 bytes but only 5 are available.
	// Treated as a "copy-from-reference" frame: tiles are nil, no error.
	payload := make([]byte, 10+4+5)
	binary.BigEndian.PutUint32(payload[10:], 100) // sz = 100 (exact size)
	r, err := NewReader(payload, 10, 0, 1, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.TileData(0, 0) != nil {
		t.Fatal("expected nil tile data when payload is truncated")
	}
}

func TestOffsetBeyondPayload(t *testing.T) {
	// tileStart beyond payload: treated as copy-from-reference, single tile is nil.
	payload := make([]byte, 5)
	r, err := NewReader(payload, 3, 10, 0, 0) // offset 13 > len 5
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.TileData(0, 0) != nil {
		t.Fatal("expected nil tile data when offset is beyond payload")
	}
}
