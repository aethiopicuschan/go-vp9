// Package mvpred implements VP9 motion vector prediction (spec §8.3.4).
//
// VP9 selects candidate MVs from up to 4 neighbouring blocks in raster order:
// above, above-right, left, above-left.  Each candidate that references the
// same frame as the current block is added to a short list.  Candidates are
// tracked with a usage count so that the most-common MV becomes "nearest"
// and the second-most-common becomes "near".  Any remaining slots are filled
// with the zero MV.
//
// This package does not perform the temporal-MV scan (requires a full decoded
// reference frame structure); temporal candidates are left as a future
// extension.
package mvpred

import "github.com/mgvs/go-vp9/mv"

// RefFrame constants used to identify which reference frame a block uses.
const (
	RefNone   = 0
	RefIntra  = 0 // intra block — no MV
	RefLast   = 1
	RefGolden = 2
	RefAltRef = 3
)

// BlockMV holds the motion vector and reference frame of one decoded block.
// Intra blocks have Ref == RefIntra and MV == {0,0}.
type BlockMV struct {
	MV  mv.MV
	Ref int // one of Ref* constants
}

// zeroMV is the zero-displacement motion vector.
var zeroMV = mv.MV{}

// candidate is an entry in the internal dedup list.
type candidate struct {
	mv    mv.MV
	count int
}

// List accumulates MV candidates for one block prediction.
// Call Add for each neighbouring BlockMV, then call BestMVs.
type List struct {
	entries [4]candidate
	n       int
}

// Add considers one neighbour block.  If its reference frame matches
// targetRef the MV is folded into the candidate list (incrementing the
// count of an existing equal MV or appending a new entry).
func (l *List) Add(b BlockMV, targetRef int) {
	if b.Ref != targetRef || targetRef == RefNone {
		return
	}
	// Search for existing equal MV to increment its count.
	for i := 0; i < l.n; i++ {
		if l.entries[i].mv == b.MV {
			l.entries[i].count++
			return
		}
	}
	if l.n < len(l.entries) {
		l.entries[l.n] = candidate{mv: b.MV, count: 1}
		l.n++
	}
}

// sort does a simple insertion sort by count descending (list is at most 4 entries).
func (l *List) sort() {
	for i := 1; i < l.n; i++ {
		key := l.entries[i]
		j := i - 1
		for j >= 0 && l.entries[j].count < key.count {
			l.entries[j+1] = l.entries[j]
			j--
		}
		l.entries[j+1] = key
	}
}

// BestMVs returns (nearest, near) — the two top candidates by usage count.
// Missing slots are filled with the zero MV.
func (l *List) BestMVs() (nearest, near mv.MV) {
	l.sort()
	nearest = zeroMV
	near = zeroMV
	if l.n >= 1 {
		nearest = l.entries[0].mv
	}
	if l.n >= 2 {
		near = l.entries[1].mv
	}
	return
}

// FindBestRefMVs is a convenience wrapper: given the four canonical VP9
// spatial neighbours (above, aboveRight, left, aboveLeft), an optional
// temporal co-located block, and the target reference frame, it returns
// (nearest, near).
//
// Pass BlockMV{} (zero value, Ref==RefNone) for any neighbour that does not
// exist (edge of frame or not yet decoded).
func FindBestRefMVs(above, aboveRight, left, aboveLeft, temporal BlockMV, targetRef int) (nearest, near mv.MV) {
	var l List
	// VP9 scans spatial candidates first, then temporal.
	l.Add(above, targetRef)
	l.Add(aboveRight, targetRef)
	l.Add(left, targetRef)
	l.Add(aboveLeft, targetRef)
	l.Add(temporal, targetRef)
	return l.BestMVs()
}

// ClampMV clamps a motion vector to the valid range for the current tile/frame.
// minRow/maxRow/minCol/maxCol are the bounds in VP9-native units
// (8 units = 1 pixel, matching the convention in the mv package).
func ClampMV(m mv.MV, minRow, maxRow, minCol, maxCol int16) mv.MV {
	if m.Row < minRow {
		m.Row = minRow
	} else if m.Row > maxRow {
		m.Row = maxRow
	}
	if m.Col < minCol {
		m.Col = minCol
	} else if m.Col > maxCol {
		m.Col = maxCol
	}
	return m
}
