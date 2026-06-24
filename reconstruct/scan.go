package reconstruct

import "github.com/mgvs/go-vp9/transform"

// VP9 scan orders for each TX size × TX type combination.
// Each table maps scan position → raster index (row*n + col).
//
// Three scan patterns (per VP9 spec §8.6.1 / libvpx vp9_scan_orders.h):
//   - Diagonal (default): even diagonals r↑, odd diagonals r↓ — used for DCT_DCT and ADST_ADST
//   - Row scan: left-to-right, top-to-bottom — used for DCT_ADST (ADST in rows)
//   - Col scan: top-to-bottom, left-to-right — used for ADST_DCT (ADST in cols)

var (
	DefaultScan4x4   = genScan(4)
	DefaultScan8x8   = genScan(8)
	DefaultScan16x16 = genScan(16)
	DefaultScan32x32 = genScan(32)

	RowScan4x4   = genRowScan(4)
	RowScan8x8   = genRowScan(8)
	RowScan16x16 = genRowScan(16)

	ColScan4x4   = genColScan(4)
	ColScan8x8   = genColScan(8)
	ColScan16x16 = genColScan(16)
)

// ScanFor returns the scan table for the given TX size and TX type.
// TX32x32 always uses the diagonal scan (ADST not used at 32×32).
func ScanFor(txSize uint8, txType transform.TxType) []int {
	if txSize >= 3 { // TX32x32
		return DefaultScan32x32
	}
	switch txType {
	case transform.DCT_ADST: // ADST in rows → row scan
		switch txSize {
		case 0:
			return RowScan4x4
		case 1:
			return RowScan8x8
		default:
			return RowScan16x16
		}
	case transform.ADST_DCT: // ADST in cols → col scan
		switch txSize {
		case 0:
			return ColScan4x4
		case 1:
			return ColScan8x8
		default:
			return ColScan16x16
		}
	default: // DCT_DCT, ADST_ADST → diagonal
		return DefaultScanFor(txSize)
	}
}

// DefaultScanFor returns the diagonal scan table for the given TX size.
// txSize: 0=TX4×4, 1=TX8×8, 2=TX16×16, 3=TX32×32.
func DefaultScanFor(txSize uint8) []int {
	switch txSize {
	case 0:
		return DefaultScan4x4
	case 1:
		return DefaultScan8x8
	case 2:
		return DefaultScan16x16
	default:
		return DefaultScan32x32
	}
}

// genRowScan generates a row-major scan: left-to-right within each row,
// top row first. Used for DCT_ADST (ADST applied in the horizontal/row direction).
func genRowScan(n int) []int {
	out := make([]int, n*n)
	pos := 0
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			out[pos] = r*n + c
			pos++
		}
	}
	return out
}

// genColScan generates a column-major scan: top-to-bottom within each column,
// left column first. Used for ADST_DCT (ADST applied in the vertical/col direction).
func genColScan(n int) []int {
	out := make([]int, n*n)
	pos := 0
	for c := 0; c < n; c++ {
		for r := 0; r < n; r++ {
			out[pos] = r*n + c
			pos++
		}
	}
	return out
}

// genScan generates the VP9 diagonal zigzag scan for an n×n block.
func genScan(n int) []int {
	out := make([]int, n*n)
	pos := 0
	for d := 0; d < 2*n-1; d++ {
		rMin := d - (n - 1)
		if rMin < 0 {
			rMin = 0
		}
		rMax := d
		if rMax > n-1 {
			rMax = n - 1
		}
		if d%2 == 1 {
			// odd diagonal: r decreases
			for r := rMax; r >= rMin; r-- {
				out[pos] = r*n + (d - r)
				pos++
			}
		} else {
			// even diagonal: r increases
			for r := rMin; r <= rMax; r++ {
				out[pos] = r*n + (d - r)
				pos++
			}
		}
	}
	return out
}
