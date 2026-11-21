# go-vp9

Pure-Go VP9 video decoder (no cgo). Original implementation from the public
**VP9 Bitstream & Decoding Process Specification**, cross-checked against the
BSD-3 reference `libvpx`. VP9 is open and royalty-free (Google grant).

> **Status — real-stream verification against libvpx:**
> - **Key frames: 100% bit-exact** with libvpx on a real ffmpeg/libvpx VP9
>   stream (Y/U/V all 0 diffs, PSNR ≡ ∞ / MSE 0, including the loop filter).
> - **Inter frames: 100% bit-exact** (all planes) with libvpx on the real
>   320×240 ffmpeg/libvpx test stream — every one of the 6 frames (key + 5 inter)
>   matches Y/U/V exactly (0 diffs). Covers mode info, references, motion vectors
>   (incl. temporal prediction), residual tokens, sub-pixel motion compensation,
>   and the inter loop filter. Multi-tile, compound (alt-ref),
>   non-64-multiple sizes, and high-bit-depth (10/12-bit profile 2) are all
>   bit-exact too.
>
> The milestone table below (M0–M20) tracks the original implementation, which
> was validated only by the decoder's *own* encoder↔decoder roundtrip. That is
> self-consistent but does **not** catch divergences from real libvpx/ffmpeg
> streams; the real-stream verification (next section) is the authoritative
> status. Ported with the assistance of an LLM.

## Real-stream verification (vs libvpx) — authoritative status

Method: build libvpx `vpxdec`, decode a real VP9 stream to a reference YUV, and
compare go-vp9's output pixel-for-pixel. For desync hunting, instrument both
decoders to dump per-block bool-coder state / mode-info and diff.

**Key frames — DONE, 100% bit-exact (all planes, incl. loop filter).** Real
streams exposed ~43 bugs the self-roundtrip missed; all fixed. Highlights:
entropy contexts (TxSizeCtx, NZC span, skip reset, left-context full-tile
indexing), forced partition at frame edges, inverse transforms (axis-swap
ADST/DCT, idct/iadst ports — oracle 984/984), intra prediction (DSP 480/480,
per-TX reconstruct, border padding), **TX_32X32 dequant `dq_shift`**, and the
loop filter (per-SB order; kernels oracle 12000/12000; the
`vp9_adjust_mask` TX4→TX8 promotion on 32-px grid lines). Regression oracles
live in the session scratchpad (txcmp/intracmp/borderpred/lfcmp; env-gated
per-block dumps).

**Inter frames — DONE, 100% bit-exact (single-tile).** Test rig:
`decoder/inter_test.go` (`TestInterStream`, env `VP9_IVF`+`VP9_REF`) decodes a
multi-frame IVF and diffs each frame; all 6 frames of the 320×240 stream match
Y/U/V exactly. The first inter frame was oracle-walked block-by-block against an
instrumented `vpxdec` (per-block bool-coder range/value + bit-position dumps)
until **all 246 blocks match**; pixel reconstruction was then driven to zero
diff via per-edge loop-filter and per-block residual oracles.

*Verified bit-exact (entropy / bool-coder), first inter frame:*
- Uncompressed header: `frame_size_with_refs` always reads `render_size`.
- Probability **frame-context model**: 4 saved contexts, pre-fc load → forward
  update → backward adapt; **coef-prob adaptation factors** (COEF_COUNT_SAT 24 /
  update-factor 112, or 128 right after a key frame).
- Compressed header: correct section order (**no phantom `uv_mode` update**),
  the **MV-prob update mechanism** (`(read_literal(7)<<1)|1`, not the subexp
  update), `default_partition_probs`, and the **inter-frame default mode
  tables** (`default_if_y_probs` / `default_if_uv_probs`, which differ from the
  key-frame tables).
- Mode-info **read order** (is_inter before tx_size; inter_mode before
  interp_filter) and contexts (intra/inter, single_ref p1/p2, interp, skip, tx).
- **Motion-vector decode**: joint/class/fp trees, magnitude, sign; the full
  **`find_mv_refs` candidate list** (8-neighbour scan + dedup + sign-bias + clamp
  + precision-lowering), **`use_mv_hp`** gating, whole-block MV resolution, and
  the **sub-8×8 path** (`append_sub8x8_mvs_for_idx` + per-sub-block `bmi`
  storage + `get_sub_block_mv`).
- Residual: coefficient-token decode with the correct intra/inter `refType`.
- **Inter motion compensation, sub-pixel filtering** — luma uses the 1/16-pel
  filter at phase `(mv&7)<<1` with integer offset `mv>>3`; chroma (4:2:0) uses
  phase `mv&15`, offset `mv>>4`; sub-8×8 blocks do per-4×4 MC (chroma uses the
  q4-rounded average of the four sub-MVs). The **filter coefficient table** was
  found wrong at EIGHTTAP phases 7/8/9 and SMOOTH phases 5–11 (dumped libvpx's
  real MC source + kernel and diffed against `vp9_filter.c`).
- **DC-only residual fast path** now applies the `TX_32X32` `>>1` dq_shift (it
  was skipped, so chroma 32×32 DC-only blocks — from 64×64 prediction blocks —
  were ±1 off).
- **Inter loop filter** — a skipped inter block only filters its prediction-block
  boundaries (libvpx `build_masks` returns before the TX / internal-4×4 masks),
  and the chroma masks **and filter level** are taken from the top-left 8×8 of
  each 16×16 region (`vp9_filter_block_plane_ss11` reads `lfl_y[(r<<3)+(c<<1)]`),
  not per-position.

*Remaining to verify / fix (point by point):*
1. ~~Chroma / inter loop filter~~ — **DONE** (all planes bit-exact, above).
2. ~~**2-D sub-pixel motion compensation**~~ — **DONE** (filter-table fix above).
3. ~~**Temporal MV prediction** (`use_prev_frame_mvs`)~~ — **DONE**: co-located
   previous-frame MV candidates in `find_mv_refs`.
4. ~~**Multi-tile** (>1 tile column)~~ — **DONE**: the 1272×720 / 4-tile clip is
   100 % bit-exact on every frame (tile-relative neighbour availability incl.
   `find_mv_refs`; 2-D MC 8-bit-clipped intermediate).
5. ~~**Non-64-multiple frame sizes** (e.g. 1920×1080)~~ — **DONE**: transform
   blocks past the frame edge are not coded (libvpx `max_blocks_wide/high`); the
   reference border was widened (8→96 px) for inter MVs that read past the edge.
6. ✅ **Compound prediction** (alt-ref / dual-ref) — **DONE**: a 2-pass alt-ref
   stream (90 frames, incl. invisible alt-refs in superframes + compound blocks)
   decodes **100 % bit-exact on every frame**. Six latent bugs that no single-ref
   stream exercised: superframe splitting (alt-ref is an invisible sub-frame),
   the compressed-header `reference_mode_probs` order (single_ref before
   comp_ref), `CompModeCtx`/`CompRefCtx` (rewritten to libvpx
   `vp9_get_reference_mode_context`/`comp_ref_p`), the default `comp_inter`
   probs, the compound ref-frame slot order (`ref[sign_bias[fixed]]`), and — the
   one that hid all the pixel error — the **LAST/GOLDEN/ALTREF→slot mapping was
   applied at end-of-frame instead of before each frame's own decode**, so a
   frame whose `ref_frame_idx` differed from the previous (e.g. an alt-ref with
   `[0 2 1]`) read the wrong reference slots.
7. ✅ **High-bit-depth** (10/12-bit, profile 2) — **DONE**. A parallel 16-bit
   (`uint16`) / 64-bit (`int64`) pipeline so the bit-exact 8-bit `[]byte` path
   stays untouched. **10-bit AND 12-bit decode 100 % bit-exact vs libvpx — key +
   inter frames, every plane Y/U/V = 0** (6-frame 320×240 streams, profile 2).
   Pixel packages (`_16`): `refbuf.Frame16`, `intra16` (`uint16`, padding
   `Base±1`/`Base`=`1<<(bd-1)`), `mc16` (sub-pel single+compound, clip to MaxVal),
   `loopfilter16` (midpoint/`sclampHigh`/`<<(bd-8)` thresholds). Residual chain is
   **`int64`** (libvpx `tran_high_t`): `coeftoken.DecodeBlock16` (bd-extended CAT6),
   `dequant` 10/12-bit `dc/ac_qlookup` + `DequantBlock16`, `transform16`
   (`HIGHBD_WRAPLOW(bd)` butterflies, `int64` cospi products), `wht16` — int32
   overflows at 12-bit (dequant coeff ≈ 2³³, products ≈ 2³⁴). Each `_16` was first
   verified `== ` its 8-bit counterpart at bd=8 on random inputs, then end-to-end
   vs libvpx (rebuilt `--enable-vp9-highbitdepth`). Six bugs found+fixed:
   `intra16` padding; default quant (`qStepsForPlaneBD`/`DCQuantBD`); **`bit_depth`
   only coded on key/intra-only frames → inter had `h.BitDepth==0`** (bd-globals
   garbage + 8-bit quant table — inter now inherits `d.bitDepth`); and the `int64`
   residual chain for 12-bit. 8-bit stays 100 % bit-exact (all `_16` is additive).
8. Pathological MVs more than 96 px past the frame edge (on-demand edge extension).
9. Remove the env-gated oracle instrumentation.

**Backward adaptation (`frame_parallel_decoding_mode=0`) — DONE.** The streams
above were all encoded with frame-parallel mode on, so adaptation never ran;
YouTube-style streams (frame_parallel=0, multi-tile, hidden alt-refs) desynced on
the first frame after a key frame. Fixed against an instrumented `vpxdec`
(per-frame probability/count table dumps): coefficient counts were not collected
at all; forced partitions at frame edges must count the whole symbol; the implied
`hp=1` bit of non-high-precision MVs is counted (and hp probs adapt only when the
frame allows hp); intra frames adapt only coef probs and keep the other
forward-updated tables. Found on the way: the 8-bit AC quantizer table tail
(q index 232–255), loop-filter ref/mode delta scaling for `filter_level >= 32`,
and the MV border clamp (`clamp_mv_to_umv_border_sb`) in inter prediction. The
tile over-read check now equals libvpx `vpx_reader_has_error`. Regression streams:
`decoder/adapt_stream_test.go`, `decoder/overread_stream_test.go`.

Status snapshot: **six streams — 320×240 single-tile, 1272×720 4-tile, 1920×1080,
a 2-pass alt-ref/compound stream (90 frames), and 10-bit + 12-bit profile-2
streams (key + inter) — all decode 100 % bit-exact vs libvpx on every frame, all
planes.**

## What's left to do (plain list)

The 320×240 single-tile test stream is fully bit-exact (key + inter, all
planes). The open work, in priority order:

1. ✅ **Inter loop filter / chroma** — done (skip blocks filter only prediction
   boundaries; chroma masks + level from the 16×16-region top-left 8×8).
2. ✅ **2-D sub-pixel motion compensation** — done (filter coefficient table was
   wrong at EIGHTTAP 7/8/9 and SMOOTH 5–11; DC-only TX32 dq_shift).
3. ✅ **Temporal MV prediction** (`use_prev_frame_mvs`) — done.
4. ✅ **Multi-tile** frames (>1 tile column) — **DONE**: the 1272×720 / 4-tile
   clip decodes **100 % bit-exact on every frame** (key + 7 inter, all planes).
   Fixes: superblock-aligned tile boundaries (`get_tile_offset`), row-interleaved
   decode across tile columns (libvpx `decode_tiles` order), and tile-relative
   neighbour availability everywhere — entropy contexts, intra-prediction border
   pixels (`gatherBorders`), `PredictBlock` have-above/have-left flags, **and the
   MV-reference search** (`find_mv_refs` restricts spatial neighbours to the tile
   columns). Also corrected the 2-D sub-pixel MC to clip the horizontal pass to
   8-bit before the vertical pass (matching `vpx_convolve8`).
5. ✅ **Non-multiple-of-64 frame sizes** — **DONE**: a 1920×1080 clip is 100 %
   bit-exact on every frame. Transform blocks whose top-left lies past the frame
   edge are not coded (libvpx `max_blocks_wide/high`); the reference border was
   widened (8→96 px) so inter MVs reading past the edge hit replicated edge
   pixels (matches `extend_and_predict`).
6. **Compound (two-reference) prediction** — implemented but never hit by the
   single-ref test streams; needs a B-frame stream to verify. **← next gap.**
7. **Untested feature paths** — segmentation, lossless (WHT), profiles 1–3, and
   10/12-bit. Code exists, but no real-stream check yet.
8. Strip the env-gated oracle instrumentation.

## Milestone roadmap

This table tracks the **original implementation** (each milestone validated by
the decoder's own encoder↔decoder roundtrip). The *real-stream* state column is
the one that matters — it's what's been re-checked bit-for-bit against libvpx
`vpxdec` on a real ffmpeg stream (the verification section above is the
authority). Legend: **✅** real-stream bit-exact · **🔶** code done, entropy
checked, pixels/behaviour still being verified · **⬜** code written but not yet
real-stream-tested.

| # | Milestone | Real-stream state |
|---|-----------|-------|
| M0 | Boolean arithmetic coder | ✅ |
| M1 | IVF container; uncompressed header | ✅ |
| M2 | Compressed header (prob delta decode) | ✅ |
| M3 | Tile + partition tree; intra mode info | ✅ |
| M4 | Intra prediction; keyframe reconstruction | ✅ |
| M5 | Transforms (DCT/ADST 4–32) + dequant; coef tokens | ✅ |
| M6 | Loop filter (deblocking) | ✅ |
| M7 | Inter prediction: MV pred, sub-pel MC, refs, compound | ✅ (single + multi-tile + compound/alt-ref) |
| M8 | Segmentation, lossless (WHT), profiles 1–3, 10/12-bit | 🔶 10/12-bit profile 2 ✅ real-stream bit-exact; segmentation/lossless/profile 3 (4:2:2/4:4:4) untested |
| M9 | Context model (above/left arrays) | ✅ |
| M10 | Full block mode decode (inter/intra) | ✅ |
| M11 | Block reconstruction (pred + residual) | ✅ |
| M12 | Frame decoder: tiles, loop filter, refbuf, output | ✅ single + multi-tile |
| M13 | Per-plane quantizer deltas + segment Q | ✅ |
| M14 | HORZ/VERT non-square partitions | ✅ |
| M15 | Sub-8×8 mode info (per-4×4 Y modes / inter MVs) | ✅ |
| M16 | Coefficient scan orders (row/col per TX type) | ✅ |
| M17 | Loop-filter ref_delta / mode_delta + per-segment level | ✅ |
| M18 | UV NZC context + zeroMV in InterModeCtx | ✅ |
| M19 | Temporal MV candidates (co-located lookup) | ✅ |
| M20 | Probability adaptation between frames | ✅ coef / mode / mv backward adaptation (`frame_parallel_decoding_mode=0`), see `decoder/adapt_stream_test.go` |

## Using the decoder (behaviour notes)

`decoder.Decoder` mirrors libvpx (`vpx_codec_decode` + `vpx_codec_get_frame`)
packet by packet: same accept/reject verdict, same picture, same "no output".
Details are in the `decoder` package documentation; the points that matter to
callers:

- **Call `DecodeFrame` for profile 0/1 (8-bit) and `DecodeFrame16` for profile
  2/3 (10/12-bit).** Both take one IVF packet payload (possibly a superframe).
- **`(nil, nil)` means "no frame for this packet"**: the packet decoded but
  its last frame is hidden (a lone alt-ref or intra-only frame), or it is a
  `show_existing_frame` that arrived while the decoder is waiting for a key
  frame after an error. libvpx outputs nothing for these packets either.
- **Frame size is the true size.** `Frame.Width`/`Height` may be odd; the
  chroma planes are `UVWidth()` x `UVHeight()` = `(Width+SubX)>>SubX` by
  `(Height+SubY)>>SubY`.
- **Not every stream is 4:2:0.** Profile 1 and 3 frames are 4:4:4, 4:2:2 or
  4:4:0; read `Frame.SubX`/`SubY` (or `UVWidth`/`UVHeight`) instead of
  assuming half-size chroma.
- **Errors are sticky until a key frame.** After any decode error every packet
  fails with `decoder.ErrNeedResync` until a key frame or an intra-only frame
  arrives (libvpx `need_resync`). Test with `errors.Is`.
- **Frame size limit.** Frames wider or taller than
  `decoder.DefaultMaxFrameSize` (16384) are rejected with an error wrapping
  `decoder.ErrFrameSize`; `SetMaxFrameSize` changes the limit.
- **Trailing data.** Non-zero bytes behind the last tile of a plain packet are
  decoded as a further frame (and usually fail), as in libvpx; zero padding is
  ignored.
- **Reference scaling is not implemented**: an inter frame whose references
  have a different size is rejected.
- **Concurrency.** One `Decoder` must not be used from two goroutines at once.
  Separate `Decoder` instances can be interleaved or used from different
  goroutines, but decoding is serialised process-wide (the lower-level
  packages keep per-frame parameters in package variables), so they do not
  run in parallel.

### API stability

Only `decoder` (with `refbuf.Frame`/`Frame16`, the frame types it returns) and
`ivf` are meant for applications. The other packages are the decoder's
building blocks; they are importable but their signatures change with the
decoder and carry no compatibility promise. Source-incompatible changes in
them since commit 14ee559:

- `blockmode.FrameInfo` has slice fields (segmentation maps) and is no longer
  comparable with `==`.
- `loopfilter.LevelForBlock` takes an additional `bool` parameter.
- `reconstruct.Block` and `reconstruct.Block16` return the block's eob total
  (`int`) instead of nothing.
- `frame.ParseRefs`/`Parse` may return a non-nil partial header together with
  an error (header rejected after `frame_context_idx`).
- `refbuf.Frame`/`Frame16`: `Width`/`Height` are the true size (previously
  rounded up to even on subsampled axes); `UVWidth`/`UVHeight` round up.

Behaviour changes of `decoder` itself since 14ee559: the `(nil, nil)` result,
true (odd) frame sizes, non-4:2:0 output for profile 1/3, the frame size
limit, the sticky error state and the trailing-data rule, all described above.

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
