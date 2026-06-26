package intra16

import "github.com/mgvs/go-vp9/modeinfo"

// Reference-sample requirement flags (libvpx extend_modes).
const (
	needLeft       = 1
	needAbove      = 2
	needAboveRight = 4
)

// extendModes[mode] gives which neighbours a mode reads (libvpx extend_modes;
// IntraMode order DC,V,H,D45,D135,D117,D153,D207,D63,TM).
var extendModes = [10]uint16{
	needAbove | needLeft, // DC
	needAbove,            // V
	needLeft,             // H
	needAboveRight,       // D45
	needLeft | needAbove, // D135
	needLeft | needAbove, // D117
	needLeft | needAbove, // D153
	needLeft,             // D207
	needAboveRight,       // D63
	needLeft | needAbove, // TM
}

// PredictBlock builds the bordered above/left reference rows the way libvpx
// build_intra_predictors does (the in-frame "faster path": padding unavailable
// above with 127, left with 129, the above-left corner with the real pixel /
// 129 / 127, and the above-right with the actual pixels only for 4x4 blocks
// whose right neighbour is available, else replicating above[bs-1]), then runs
// the predictor. realAbove[0..2bs-1] / realLeft[0..bs-1] / corner are the actual
// neighbour pixels (already edge-extended by the caller); they are consulted
// only where haveTop / haveLeft are set.
func PredictBlock(mode modeinfo.IntraMode, dst []uint16, stride, bs int,
	realAbove, realLeft []uint16, corner uint16, haveTop, haveLeft, haveRight bool) {

	var aboveBuf [2*32 + 1]uint16
	above := aboveBuf[1:] // above[-1] == aboveBuf[0]
	var leftCol [32]uint16
	em := extendModes[mode]

	if em&needLeft != 0 {
		if haveLeft {
			copy(leftCol[:bs], realLeft[:bs])
		} else {
			for i := 0; i < bs; i++ {
				leftCol[i] = uint16(Base + 1)
			}
		}
	}

	setCorner := func() {
		if haveLeft {
			aboveBuf[0] = corner
		} else {
			aboveBuf[0] = uint16(Base + 1)
		}
	}

	if em&needAbove != 0 {
		if haveTop {
			copy(above[:bs], realAbove[:bs])
			setCorner()
		} else {
			for i := 0; i < bs; i++ {
				above[i] = uint16(Base - 1)
			}
			aboveBuf[0] = uint16(Base - 1)
		}
	}

	if em&needAboveRight != 0 {
		if haveTop {
			copy(above[:bs], realAbove[:bs])
			if bs == 4 && haveRight {
				copy(above[bs:2*bs], realAbove[bs:2*bs])
			} else {
				for i := bs; i < 2*bs; i++ {
					above[i] = above[bs-1]
				}
			}
			setCorner()
		} else {
			for i := 0; i < 2*bs; i++ {
				above[i] = uint16(Base - 1)
			}
			aboveBuf[0] = uint16(Base - 1)
		}
	}

	if mode == modeinfo.DC {
		dcVariant(dst, stride, bs, above, leftCol[:], haveTop, haveLeft)
		return
	}
	predictMode(mode, dst, stride, bs, above, leftCol[:], aboveBuf[0])
}

// dcVariant is libvpx dc_pred[left_available][up_available]: average the
// available sides only (both / left-only / top-only / neither→128).
func dcVariant(dst []uint16, stride, bs int, above, left []uint16, haveTop, haveLeft bool) {
	var avg uint16
	switch {
	case haveTop && haveLeft:
		sum := 0
		for i := 0; i < bs; i++ {
			sum += int(above[i]) + int(left[i])
		}
		avg = uint16((sum + bs) / (2 * bs))
	case haveLeft:
		sum := 0
		for i := 0; i < bs; i++ {
			sum += int(left[i])
		}
		avg = uint16((sum + bs/2) / bs)
	case haveTop:
		sum := 0
		for i := 0; i < bs; i++ {
			sum += int(above[i])
		}
		avg = uint16((sum + bs/2) / bs)
	default:
		avg = uint16(Base)
	}
	for r := 0; r < bs; r++ {
		row := dst[r*stride : r*stride+bs]
		for i := range row {
			row[i] = avg
		}
	}
}

// predictMode dispatches the non-DC predictors on already-bordered samples.
func predictMode(mode modeinfo.IntraMode, dst []uint16, stride, bs int, above, left []uint16, topLeft uint16) {
	switch mode {
	case modeinfo.V:
		vPredict(dst, stride, bs, above)
	case modeinfo.H:
		hPredict(dst, stride, bs, left)
	case modeinfo.TM:
		tmPredict(dst, stride, bs, above, left, topLeft)
	case modeinfo.D45:
		d45Predict(dst, stride, bs, above)
	case modeinfo.D63:
		d63Predict(dst, stride, bs, above)
	case modeinfo.D117:
		d117Predict(dst, stride, bs, above, left, topLeft)
	case modeinfo.D135:
		d135Predict(dst, stride, bs, above, left, topLeft)
	case modeinfo.D153:
		d153Predict(dst, stride, bs, above, left, topLeft)
	case modeinfo.D207:
		d207Predict(dst, stride, bs, left)
	}
}
