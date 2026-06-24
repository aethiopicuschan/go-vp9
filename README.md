# go-vp9

Pure-Go VP9 video decoder (no cgo). Original implementation from the public
**VP9 Bitstream & Decoding Process Specification**, cross-checked against the
BSD-3 reference `libvpx`. VP9 is open and royalty-free (Google grant).

> **Status: all milestones complete (M0–M20).**
> Built and bit-exactly verified one milestone at a time against libvpx.
> Ported with the assistance of an LLM.

## Milestone roadmap

| # | Milestone | State |
|---|-----------|-------|
| M0 | Boolean arithmetic coder (decoder + matching encoder, roundtrip-tested) | ✅ |
| M1 | IVF container reader/writer; uncompressed header (profile, sync, color, dims, tiles) | ✅ |
| M2 | Compressed header: tx_mode, coef/skip/inter/MV prob delta decode (diff_update_prob + inv_remap) | ✅ |
| M3 | Tile + superblock/partition tree decode; mode info (intra) | ✅ |
| M4 | Intra prediction (DC/V/H/TM + directional), keyframe reconstruction | ✅ |
| M5 | Transforms (DCT/ADST 4/8/16/32) + dequant; coefficient tokens | ✅ |
| M6 | Loop filter (deblocking) | ✅ |
| M7 | Inter prediction: MV prediction, sub-pel MC, compound, reference frames | ✅ |
| M8 | Segmentation, lossless (WHT), profiles 1–3, 10/12-bit | ✅ |
| M9 | Context model: above/left arrays for partition, mode, ref, TX, skip, NZC | ✅ |
| M10 | Full block mode decode: inter/intra, ref frames, MV, skip, TX size | ✅ |
| M11 | Block reconstruction: intra/inter prediction + residual pipeline | ✅ |
| M12 | Frame decoder: tile orchestration, loop filter, refbuf, image output | ✅ |
| M13 | Per-plane quantizer deltas (delta_q_y_dc / uv_dc / uv_ac) + segment Q apply | ✅ |
| M14 | Correct HORZ/VERT partition: non-square block sizes (BLOCK_NxM) in partition tree | ✅ |
| M15 | Sub-8×8 mode info: per-4×4 Y intra modes and per-4×4 inter MVs inside 8×8 block | ✅ |
| M16 | Coefficient scan orders: row-scan (DCT_ADST) and col-scan (ADST_DCT) per TX type | ✅ |
| M17 | Loop filter refinement: ref_delta / mode_delta per block; per-segment LF level | ✅ |
| M18 | UV NZC context fix + zeroMV flag wired into InterModeCtx | ✅ |
| M19 | Temporal MV candidates: co-located block lookup in reference frame MV buffer | ✅ |
| M20 | Probability adaptation: symbol count accumulation + vp9_adapt_probs between frames | ✅ |

Each milestone is validated bit-exactly: build libvpx `vpxdec`, dump intermediate
state, compare.

## Layout
- `boolcoder/` — VP9 boolean arithmetic coder (M0).
- `ivf/` — IVF container reader + writer (M1).
- `frame/` — uncompressed frame header parser (M1).
- `compheader/` — compressed header decoder: tx_mode, coef/skip/inter/MV probs (M2); DefaultProbs() (M3).
- `tile/` — tile data reader: parses tile byte slices from frame payload (M3).
- `partition/` — partition tree decode (NONE/HORZ/VERT/SPLIT) + context (M3).
- `modeinfo/` — intra prediction mode decode: Y/UV mode trees (M3).
- `intra/` — intra prediction for all 10 modes × 4 block sizes; DC/V/H/TM/D45/D63/D117/D135/D153/D207 (M4).
- `transform/` — inverse DCT-II and ADST for 4/8/16/32; 2-D separable pass (M5).
- `dequant/` — DC/AC quantizer lookup tables (spec Appendix A) and DequantBlock (M5).
- `coeftoken/` — coefficient token decode: EOB/zero/one/CAT1-6, band tables, context (M5).
- `loopfilter/` — deblocking loop filter: 4/8/14-tap kernels, mask predicates, FilterPlane (M6).
- `mc/` — motion compensation: 8-tap (regular/smooth/sharp) + bilinear sub-pixel filters; Predict + PredictComp (M7).
- `mv/` — MV bitstream decode: joint type, CLASS_0..CLASS_10 tree, FP/HP bits; ReadMV (M7).
- `mvpred/` — MV prediction: neighbour-scan candidate List, FindBestRefMVs, ClampMV (M7).
- `refbuf/` — reference frame buffer: 8-slot pool, refresh bitmask, border extension (M7).
- `segment/` — segmentation decode: 7-prob balanced-tree DecodeSegID, ParamsFor (AltQ/AltLF/RefFrame/Skip), ClampQIndex (M8).
- `wht/` — 4×4 inverse Walsh-Hadamard transform for VP9 lossless mode; IWHT4x4 (M8).
- `ctx/` — above/left context arrays: partition, mode, ref frame, TX size, skip, NZC; reset helpers (M9).
- `blockmode/` — full block mode decode: FrameInfo, BlockMode, Decode(); ref-frame context (CompModeCtx, SingleRefCtx, CompRefCtx); CompFixedVarRef (M10).
- `reconstruct/` — pixel reconstruction per block: diagonal scan tables (genScan), TxTypeFor, Block() for intra+inter+residual pipeline (M11).
- `decoder/` — frame decoder: Decoder.DecodeFrame() orchestrates tile loop, partition recursion, block decode, loop filter, refbuf update (M12).
