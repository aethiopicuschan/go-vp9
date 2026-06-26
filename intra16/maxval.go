package intra16

// MaxVal is the maximum pixel value (1<<bitDepth - 1) for high-bit-depth intra
// prediction clamping. Set it before calling PredictBlock. Default 255 (8-bit).
var MaxVal = 255

// Base is the midpoint 1<<(bitDepth-1) used for unavailable-neighbour padding
// (above=Base-1, left=Base+1) and the no-neighbour DC value. Default 128 (8-bit).
var Base = 128
