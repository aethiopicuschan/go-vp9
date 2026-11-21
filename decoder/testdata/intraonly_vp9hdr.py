#!/usr/bin/env python3
# Bit-level editor for VP9 uncompressed frame headers inside IVF files.
# Used to derive header variants no libvpx encoder path emits (intra-only
# frames, non-zero frame_context_idx on key frames, sign-bias bits on
# error-resilient frames, extra show_existing_frame frames) from ordinary
# libvpx encodes. The payload after the uncompressed header (compressed header
# + tile data) is always copied verbatim. Every derived stream is validated by
# decoding it with libvpx (see intraonly_streams.txt).
#
# usage: vp9hdr.py dump X.ivf
#        vp9hdr.py build RECIPE OUT.ivf IN1.ivf [IN2.ivf ...]   (recipes below)
import struct, sys


def read_ivf(path):
    d = open(path, 'rb').read()
    hdr, off, pkts = d[:32], 32, []
    while off + 12 <= len(d):
        sz = struct.unpack('<I', d[off:off + 4])[0]
        pkts.append(d[off + 12:off + 12 + sz])
        off += 12 + sz
    return hdr, pkts


def write_ivf(path, hdr, pkts):
    out = bytearray(hdr)
    out[24:28] = struct.pack('<I', len(pkts))
    for i, p in enumerate(pkts):
        out += struct.pack('<IQ', len(p), i) + p
    open(path, 'wb').write(out)


def split_superframe(p):
    m = p[-1]
    if m & 0xe0 != 0xc0:
        return [p]
    bps, n = ((m >> 3) & 3) + 1, (m & 7) + 1
    isz = 2 + n * bps
    if len(p) < isz or p[-isz] != m:
        return [p]
    out, off, q = [], 0, len(p) - isz + 1
    for _ in range(n):
        sz = int.from_bytes(p[q:q + bps], 'little')
        q += bps
        out.append(p[off:off + sz])
        off += sz
    return out


def make_superframe(frames):
    if len(frames) == 1:
        return frames[0]
    bps = 4
    m = 0xc0 | ((bps - 1) << 3) | (len(frames) - 1)
    idx = bytes([m]) + b''.join(len(f).to_bytes(bps, 'little') for f in frames) + bytes([m])
    return b''.join(frames) + idx


class Hdr:
    """Uncompressed header as an ordered list of [name, nbits, value]."""

    def __init__(self, frame):
        self.f, self.rest = [], b''
        self.data, self.pos = frame, 0
        self._parse()
        nbytes = (self.pos + 7) // 8
        self.rest = frame[nbytes:]
        del self.data

    def _r(self, name, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        self.f.append([name, n, v])
        return v

    def _color(self, profile):
        if profile >= 2:
            self._r('ten_or_twelve_bit', 1)
        cs = self._r('color_space', 3)
        if cs != 7:
            self._r('color_range', 1)
            if profile in (1, 3):
                self._r('subsampling_x', 1); self._r('subsampling_y', 1); self._r('reserved_zero', 1)
        elif profile in (1, 3):
            self._r('reserved_zero', 1)

    def _size(self):
        self._r('frame_width_minus_1', 16); self._r('frame_height_minus_1', 16)

    def _render(self):
        if self._r('render_and_frame_size_different', 1):
            self._r('render_width_minus_1', 16); self._r('render_height_minus_1', 16)

    def _parse(self):
        r = self._r
        r('frame_marker', 2)
        profile = r('profile_low_bit', 1) | (r('profile_high_bit', 1) << 1)
        self.profile = profile
        if profile == 3:
            r('reserved_zero', 1)
        if r('show_existing_frame', 1):
            r('frame_to_show_map_idx', 3)
            return
        ftype = r('frame_type', 1)
        show = r('show_frame', 1)
        er = r('error_resilient_mode', 1)
        width = None
        if ftype == 0:
            r('sync_code', 24)
            self._color(profile)
            self._size(); self._render()
        else:
            intra_only = 0 if show else r('intra_only', 1)
            if not er:
                r('reset_frame_context', 2)
            if intra_only:
                r('sync_code', 24)
                if profile > 0:
                    self._color(profile)
                r('refresh_frame_flags', 8)
                self._size(); self._render()
            else:
                r('refresh_frame_flags', 8)
                for i in range(3):
                    r('ref_frame_idx_%d' % i, 3); r('ref_frame_sign_bias_%d' % i, 1)
                found = 0
                for i in range(3):
                    found = r('found_ref_%d' % i, 1)
                    if found:
                        break
                if not found:
                    self._size()
                self._render()
                r('allow_high_precision_mv', 1)
                if not r('is_filter_switchable', 1):
                    r('raw_interpolation_filter', 2)
        if not er:
            r('refresh_frame_context', 1); r('frame_parallel_decoding_mode', 1)
        r('frame_context_idx', 2)
        self.tail_start = len(self.f)  # loop filter .. header_size: copied as-is
        r('loop_filter_level', 6); r('loop_filter_sharpness', 3)
        if r('loop_filter_delta_enabled', 1):
            if r('loop_filter_delta_update', 1):
                for i in range(4):
                    if r('update_ref_delta', 1):
                        r('ref_delta', 7)
                for i in range(2):
                    if r('update_mode_delta', 1):
                        r('mode_delta', 7)
        r('base_q_idx', 8)
        for n in ('y_dc', 'uv_dc', 'uv_ac'):
            if r('delta_coded_' + n, 1):
                r('delta_q_' + n, 5)
        if r('segmentation_enabled', 1):
            if r('segmentation_update_map', 1):
                for i in range(7):
                    if r('tree_prob_coded', 1):
                        r('tree_prob', 8)
                if r('segmentation_temporal_update', 1):
                    for i in range(3):
                        if r('pred_prob_coded', 1):
                            r('pred_prob', 8)
            if r('segmentation_update_data', 1):
                r('segmentation_abs_or_delta_update', 1)
                for s in range(8):
                    for bits, signed in ((8, 1), (6, 1), (2, 0), (0, 0)):
                        if r('feature_enabled', 1):
                            if bits:
                                r('feature_value', bits)
                            if signed:
                                r('feature_sign', 1)
        w = self.get('frame_width_minus_1')
        if w is None:
            w = self.inherit_width - 1  # inter frame using a reference's size
        sb = (w + 1 + 63) >> 6
        mn = 0
        while (64 << mn) < sb:
            mn += 1
        mx = 1
        while (sb >> mx) >= 4:
            mx += 1
        mx -= 1
        n = mn
        while n < mx:
            if not r('increment_tile_cols_log2', 1):
                break
            n += 1
        if r('tile_rows_log2', 1):
            r('increment_tile_rows_log2', 1)
        r('header_size_in_bytes', 16)

    inherit_width = 64  # set by callers before parsing inter frames (tile info needs it)

    def get(self, name):
        for n, _, v in self.f:
            if n == name:
                return v
        return None

    def set(self, name, value):
        for e in self.f:
            if e[0] == name:
                assert 0 <= value < (1 << e[1]), (name, value)
                e[2] = value
                return self
        raise KeyError(name)

    def bytes(self):
        bits = ''.join(format(v, '0%db' % n) for _, n, v in self.f if n)
        bits += '0' * (-len(bits) % 8)
        return int(bits, 2).to_bytes(len(bits) // 8, 'big') + self.rest

    def describe(self):
        skip = ('frame_marker', 'sync_code', 'reserved_zero')
        return ' '.join('%s=%d' % (n, v) for n, _, v in self.f[:self.tail_start if hasattr(self, 'tail_start') else None]
                        if n not in skip)


def key_to_intra_only(frame, reset, rff, ctx=None, rfc=None):
    """Re-header a key frame as a hidden intra-only frame (payload untouched)."""
    k = Hdr(frame)
    assert k.get('frame_type') == 0 and k.get('show_existing_frame') == 0
    er = k.get('error_resilient_mode')
    names = [e[0] for e in k.f]
    i_sync = names.index('sync_code')
    i_size = names.index('frame_width_minus_1')
    color = k.f[i_sync + 1:i_size]
    out = k.f[:names.index('frame_type')]
    out += [['frame_type', 1, 1], ['show_frame', 1, 0], ['error_resilient_mode', 1, er], ['intra_only', 1, 1]]
    if not er:
        out.append(['reset_frame_context', 2, reset])
    out.append(['sync_code', 24, 0x498342])
    if k.profile > 0:
        out += color
    out.append(['refresh_frame_flags', 8, rff])
    out += k.f[i_size:]
    k.f = out
    if ctx is not None:
        k.set('frame_context_idx', ctx)
    if rfc is not None:
        k.set('refresh_frame_context', rfc)
    return k.bytes()


def patch(frame, **kv):
    h = Hdr(frame)
    for name, v in kv.items():
        h.set(name, v)
    return h.bytes()


def show_existing(profile, idx):
    assert profile in (0, 2)
    return bytes([0x80 | ((profile & 1) << 5) | ((profile >> 1) << 4) | 0x08 | idx])


def frames_of(path):
    hdr, pkts = read_ivf(path)
    Hdr.inherit_width = struct.unpack('<H', hdr[12:14])[0]
    return hdr, pkts


# ---- recipes ---------------------------------------------------------------
# A = IVF whose first packet is a key frame of picture A (shown first).
# B = IVF "key + inter frames" of a different picture: its key frame becomes the
#     hidden intra-only frame, packed in a superframe with B's first inter frame.

def r_intra(ins, reset, rff, ctx, nB, a_rfc=None):
    hdr, a = frames_of(ins[0])
    _, b = frames_of(ins[1])
    k1 = a[0] if a_rfc is None else patch(a[0], refresh_frame_context=a_rfc)
    io = key_to_intra_only(b[0], reset=reset, rff=rff, ctx=ctx)
    return hdr, [k1, make_superframe([io, b[1]])] + b[2:1 + nB]


RECIPES = {
    # intra-only, reset_frame_context=3, header frame_context_idx=1 (libvpx forces 0), refreshes all slots
    'io_reset3_ctx1': lambda ins: r_intra(ins, 3, 0xff, 1, 4),
    # intra-only, reset_frame_context=2 idx 0, refreshes only slots 1,2: LAST keeps A's key frame
    'io_reset2_keeprefs': lambda ins: r_intra(ins, 2, 0x06, 0, 4),
    # intra-only, reset_frame_context=1 (no reset; A's key frame patched to refresh_frame_context=0 so
    # context 0 is still the default), header idx 3, backward adaptation on (B is frame_parallel=0)
    'io_reset1_adapt': lambda ins: r_intra(ins, 1, 0xff, 3, 1, a_rfc=0),
    # key frame with header frame_context_idx=2 (libvpx forces 0)
    'key_ctx2': lambda ins: (lambda hdr, p: (hdr, [patch(p[0], frame_context_idx=2)] + p[1:]))(*frames_of(ins[0])),
    # error-resilient inter frames with ALTREF sign_bias bit set (libvpx zeroes sign biases on such frames)
    'er_signbias': lambda ins: (lambda hdr, p: (hdr, [p[0]] + [patch(x, ref_frame_sign_bias_2=1) for x in p[1:]]))(*frames_of(ins[0])),
    # a show_existing_frame (slot 0) packet inserted after every 2nd packet
    'show_existing': lambda ins: (lambda hdr, p: (hdr, [y for i, x in enumerate(p) for y in ([x, show_existing(Hdr(split_superframe(x)[0]).profile, 0)] if i % 2 == 1 else [x])]))(*frames_of(ins[0])),
    # (exploration only) intra-only frame first in the stream / alone in its packet
    'io_first': lambda ins: (lambda hdr, b: (hdr, [make_superframe([key_to_intra_only(b[0], reset=3, rff=0xff), b[1]])] + b[2:]))(*frames_of(ins[0])),
    'io_lone': lambda ins: (lambda hdr, b: (hdr, [frames_of(ins[1])[1][0], key_to_intra_only(b[0], reset=3, rff=0xff)] + b[1:]))(*frames_of(ins[0])),
    'show_existing_alt': lambda ins: (lambda hdr, p: (hdr, [y for i, x in enumerate(p) for y in ([x, show_existing(0, 2), show_existing(0, 1)] if i in (3, 4, 9) else [x])]))(*frames_of(ins[0])),
}

if __name__ == '__main__':
    if sys.argv[1] == 'dump':
        hdr, pkts = frames_of(sys.argv[2])
        for i, p in enumerate(pkts):
            for j, f in enumerate(split_superframe(p)):
                print('pkt %d.%d (%d B): %s' % (i, j, len(f), Hdr(f).describe()))
    elif sys.argv[1] == 'build':
        hdr, pkts = RECIPES[sys.argv[2]](sys.argv[4:])
        write_ivf(sys.argv[3], hdr, pkts)
    else:
        sys.exit('unknown command')
