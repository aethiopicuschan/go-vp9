// Package tile parses VP9 tile data from a frame payload (spec §5.2.2).
// A VP9 frame carries (1<<TileColsLog2)*(1<<TileRowsLog2) tiles, ordered
// row-major. All tiles except the last carry a big-endian 4-byte size prefix;
// the last tile extends to the end of the frame payload.
package tile

import (
	"encoding/binary"
	"errors"
)

// Reader holds the byte slices for every tile in the frame.
type Reader struct {
	tiles [][]byte
	nCols int
	nRows int
	// lastOff is the offset in the frame payload at which the last tile's
	// data starts (len(payload) when the tile data is cut short before it).
	lastOff int
}

// NewReader parses the tile data from payload.
//
//   - uncompHdrBytes: bytes consumed by the uncompressed header
//     (frame.Header.Size).
//   - compHdrBytes: size of the compressed header in bytes
//     (frame.Header.CompressedHeaderSize).
//   - tileColsLog2, tileRowsLog2: from frame.Header.TileColsLog2/TileRowsLog2.
//
// Tile data starts at payload[uncompHdrBytes + compHdrBytes:].
func NewReader(payload []byte, uncompHdrBytes, compHdrBytes, tileColsLog2, tileRowsLog2 int) (*Reader, error) {
	tileStart := uncompHdrBytes + compHdrBytes
	if tileStart < 0 {
		return nil, errors.New("tile: negative tile data offset")
	}
	var data []byte
	if tileStart <= len(payload) {
		data = payload[tileStart:]
	}

	nCols := 1 << uint(tileColsLog2)
	nRows := 1 << uint(tileRowsLog2)
	numTiles := nCols * nRows

	tiles := make([][]byte, numTiles)
	lastOff := len(payload)
	pos := 0
	for i := 0; i < numTiles; i++ {
		last := i == numTiles-1
		if last {
			if pos > len(data) {
				// No tile data available; leave as nil (empty).
				break
			}
			tiles[i] = data[pos:]
			lastOff = tileStart + pos
		} else {
			// 4-byte big-endian tile size (exact byte count, no +1 adjustment).
			if pos+4 > len(data) {
				// No tile data available; leave remaining tiles as nil.
				break
			}
			sz := int(binary.BigEndian.Uint32(data[pos:]))
			pos += 4
			if pos+sz > len(data) {
				// Tile payload extends past end; leave remaining tiles as nil.
				break
			}
			tiles[i] = data[pos : pos+sz]
			pos += sz
		}
	}
	return &Reader{tiles: tiles, nCols: nCols, nRows: nRows, lastOff: lastOff}, nil
}

// NumCols returns the number of tile columns.
func (r *Reader) NumCols() int { return r.nCols }

// NumRows returns the number of tile rows.
func (r *Reader) NumRows() int { return r.nRows }

// LastTileOffset returns the offset in the frame payload of the last tile's
// data, so that an end position inside that tile (boolcoder.Reader.FindEnd)
// can be turned into the end of the frame.
func (r *Reader) LastTileOffset() int { return r.lastOff }

// TileData returns the raw byte slice for tile at (row, col).
// The caller passes these bytes to boolcoder.NewReader to decode the tile.
func (r *Reader) TileData(row, col int) []byte {
	return r.tiles[row*r.nCols+col]
}
