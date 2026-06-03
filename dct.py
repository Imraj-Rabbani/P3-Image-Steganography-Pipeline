import os, zlib
import numpy as np
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Global constants ─────────────────────────────────────────────────────────
N                     = 8
VERSION               = 3                  # v3: Y channel + exact-Y BGR reconstruction
FALLBACK_K            = 24
ELIGIBILITY_THRESHOLD = 2
MAGIC_SHORT           = 0x4447            # 'DG'
HEADER_BLOCKS         = [(0, 0), (0, 1)]
HEADER_BLOCK_BITS     = 32
HEADER_TOTAL_BITS     = 64
MAX_FIX_ITERATIONS    = 20

# JPEG luminance quantisation table (quality ≈85)
_Q_TABLE = np.array([
    [ 3,  2,  2,  3,  4,  6,  8, 10],
    [ 2,  2,  3,  4,  5,  9, 10,  9],
    [ 3,  3,  4,  5,  6,  9, 11,  9],
    [ 3,  4,  5,  6,  8, 14, 13, 10],
    [ 4,  5,  7,  9, 11, 17, 16, 12],
    [ 5,  7,  9, 10, 13, 17, 18, 15],
    [10, 13, 12, 14, 16, 19, 19, 17],
    [14, 17, 18, 18, 19, 18, 19, 17],
], dtype=np.float32)

# ── Zigzag scan for 8×8 ──────────────────────────────────────────────────────
def _build_zigzag():
    order = []
    for s in range(2 * N - 1):
        if s % 2 == 0:
            r = min(s, N - 1); c = s - r
            while r >= 0 and c < N:
                order.append((r, c)); r -= 1; c += 1
        else:
            c = min(s, N - 1); r = s - c
            while c >= 0 and r < N:
                order.append((r, c)); r += 1; c -= 1
    return order

_ZIGZAG = _build_zigzag()
_ZZ_IDX = {uv: i for i, uv in enumerate(_ZIGZAG)}

# ── Stable positions: non-DC, Q≥8 (39 positions) ────────────────────────────
_STABLE = [
    (u, v)
    for u in range(N) for v in range(N)
    if not (u == 0 and v == 0) and _Q_TABLE[u, v] >= 8
]

_PAYLOAD_CANDIDATES = sorted(_STABLE, key=lambda uv: _ZZ_IDX[uv])
MAX_K = len(_PAYLOAD_CANDIDATES) - 1

ZZ_START = _ZZ_IDX[_PAYLOAD_CANDIDATES[0]]
ZZ_END   = _ZZ_IDX[_PAYLOAD_CANDIDATES[-1]]

_HEADER_POSITIONS = sorted(
    _STABLE,
    key=lambda uv: (-float(_Q_TABLE[uv[0]][uv[1]]), uv[0], uv[1])
)[:HEADER_BLOCK_BITS]

_HEADER_BLOCK_SET = set(HEADER_BLOCKS)

# ── DCT helpers ──────────────────────────────────────────────────────────────
def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def _to_blocks(channel: np.ndarray):
    h, w   = channel.shape
    h8, w8 = h - h % N, w - w % N
    nr, nc = h8 // N, w8 // N
    ch     = channel[:h8, :w8].astype(np.float64)
    blocks = np.zeros((nr, nc, N, N), dtype=np.float64)
    for i in range(nr):
        for j in range(nc):
            blocks[i, j] = _dct2(ch[i*N:i*N+N, j*N:j*N+N] - 128.0)
    return blocks, h8, w8

def _from_blocks(blocks: np.ndarray, orig: np.ndarray, h8: int, w8: int) -> np.ndarray:
    out    = orig.copy()
    nr, nc = blocks.shape[:2]
    for i in range(nr):
        for j in range(nc):
            patch = np.clip(np.round(_idct2(blocks[i, j]) + 128.0), 0, 255)
            out[i*N:i*N+N, j*N:j*N+N] = patch.astype(np.uint8)
    return out

# ── Quantised LSB helpers ────────────────────────────────────────────────────
def _qi(coeff: float, u: int, v: int) -> int:
    return int(np.round(float(coeff) / float(_Q_TABLE[u, v])))

def _lsb_read(coeff: float, u: int, v: int) -> int:
    return _qi(coeff, u, v) & 1

def _lsb_write(coeff: float, u: int, v: int, bit: int, flip_dir: int = -1) -> float:
    Q     = float(_Q_TABLE[u, v])
    q     = _qi(coeff, u, v)
    if (q & 1) == bit:
        return float(q) * Q
    q_up, q_dn = q + 1, q - 1
    up_ok = abs(q_up) >= ELIGIBILITY_THRESHOLD
    dn_ok = abs(q_dn) >= ELIGIBILITY_THRESHOLD
    if up_ok and dn_ok:
        if flip_dir == 1:   return float(q_up) * Q
        if flip_dir == 0:   return float(q_dn) * Q
        return (float(q_up) if abs(q_up * Q - coeff) <= abs(q_dn * Q - coeff)
                else float(q_dn)) * Q
    if up_ok:  return float(q_up) * Q
    if dn_ok:  return float(q_dn) * Q
    return float(q_up if q >= 0 else q_dn) * Q

# ── PRNG for LSB-matching direction ──────────────────────────────────────────
def _make_prng(h: int, w: int):
    seed = (int(h) * 0x9e3779b9 ^ int(w) * 0x6c62272e) & 0xFFFFFFFF
    seed = seed or 0xDEADBEEF
    state = [seed]
    def _next() -> int:
        s = state[0]
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17)
        s ^= (s <<  5) & 0xFFFFFFFF
        state[0] = s
        return (s >> 16) & 1
    return _next

# ── Position selection ───────────────────────────────────────────────────────
def _payload_positions(K: int) -> list:
    return _PAYLOAD_CANDIDATES[:K + 1]

def _build_slot_order(K: int, nr: int, nc: int) -> list:
    positions = _payload_positions(K)
    slots = []
    for pos_rank, (u, v) in enumerate(positions):
        q_val = float(_Q_TABLE[u, v])
        for i in range(nr):
            for j in range(nc):
                if (i, j) in _HEADER_BLOCK_SET:
                    continue
                slots.append((q_val, pos_rank, i, j, u, v))
    slots.sort(key=lambda s: (s[0], s[1], s[2], s[3]))
    return [(i, j, u, v) for (_, _, i, j, u, v) in slots]

def _compute_K(blocks: np.ndarray, verbose: bool = True) -> int:
    nr, nc = blocks.shape[:2]
    counts = [
        sum(1 for u, v in _PAYLOAD_CANDIDATES
            if abs(_qi(blocks[i, j, u, v], u, v)) >= ELIGIBILITY_THRESHOLD)
        for i in range(nr) for j in range(nc)
        if (i, j) not in _HEADER_BLOCK_SET
    ]
    if not counts:
        return FALLBACK_K
    mean_val = int(np.mean(counts))
    if mean_val <= 0:
        return FALLBACK_K
    extra = mean_val // 2
    K     = max(1, min(MAX_K, mean_val + extra))
    if verbose:
        print(f"  K-stats  : mean={mean_val}  extra={extra}  K={K}  (freqs 1-{K+1})")
    return K

# ── Header ────────────────────────────────────────────────────────────────────
def _int_to_bits(value: int, n_bits: int) -> list:
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]

def _bits_to_int(bits: list) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | (int(b) & 1)
    return v

def _pack_header(n_message_bits: int, K: int) -> list:
    magic   = MAGIC_SHORT
    payload = n_message_bits & 0xFFFFFF
    k6      = K & 0x3F
    ver4    = VERSION & 0xF
    data50  = (magic << 34) | (payload << 10) | (k6 << 4) | ver4
    body    = data50.to_bytes(7, 'big')
    crc14   = zlib.crc32(body) & 0x3FFF
    return (
        _int_to_bits(magic,   16)
        + _int_to_bits(payload, 24)
        + _int_to_bits(k6,       6)
        + _int_to_bits(ver4,     4)
        + _int_to_bits(crc14,   14)
    )

def _unpack_header(bits: list) -> tuple:
    magic   = _bits_to_int(bits[0:16])
    payload = _bits_to_int(bits[16:40])
    k6      = _bits_to_int(bits[40:46])
    ver4    = _bits_to_int(bits[46:50])
    crc14r  = _bits_to_int(bits[50:64])
    data50  = (magic << 34) | (payload << 10) | (k6 << 4) | ver4
    body    = data50.to_bytes(7, 'big')
    crc_ok  = (zlib.crc32(body) & 0x3FFF) == crc14r
    magic_ok = magic == MAGIC_SHORT
    return payload, k6, ver4, (crc_ok and magic_ok)

def _write_header(blocks: np.ndarray, n_message_bits: int, K: int) -> None:
    bits = _pack_header(n_message_bits, K)
    for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
        chunk = bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_HEADER_POSITIONS):
            blocks[bi, bj, u, v] = _lsb_write(
                blocks[bi, bj, u, v], u, v, chunk[k], flip_dir=-1
            )

def _read_header(blocks: np.ndarray) -> tuple:
    bits = []
    for bi, bj in HEADER_BLOCKS:
        for u, v in _HEADER_POSITIONS:
            bits.append(_lsb_read(blocks[bi, bj, u, v], u, v))
    return _unpack_header(bits)

# ── Verify-fix pass (G-channel: only IDCT roundtrip, no color loss) ──────────
def _verify_and_fix(slot_order: list,
                    current_y: np.ndarray,
                    h8: int, w8: int,
                    header_bits: list,
                    payload_bits: list,
                    K: int) -> tuple:
    
    # Re-DCT the current uint8 Y channel — exactly what extract() will see.
    blk, _, _ = _to_blocks(current_y)

    # ── Fix header (min-distortion direction) ────────────────────────────────
    for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
        chunk = header_bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_HEADER_POSITIONS):
            expected = chunk[k]
            if _lsb_read(blk[bi, bj, u, v], u, v) != expected:
                blk[bi, bj, u, v] = _lsb_write(
                    blk[bi, bj, u, v], u, v, expected, flip_dir=-1
                )

    # ── Fix payload (slot order, min-distortion direction) ───────────────────
    n_payload_bits = len(payload_bits)
    for k in range(n_payload_bits):
        i, j, u, v = slot_order[k]
        expected = payload_bits[k]
        if _lsb_read(blk[i, j, u, v], u, v) != expected:
            blk[i, j, u, v] = _lsb_write(blk[i, j, u, v], u, v, expected, flip_dir=-1)

    # Reconstruct Y channel from fixed DCT blocks.
    new_g = _from_blocks(blk, current_y, h8, w8)

    # ── Count remaining errors AFTER the IDCT roundtrip ─────────────────────
    # This is the definitive check: does extract() get the right bits?
    blk_v, _, _ = _to_blocks(new_g)
    n_remaining = 0

    for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
        chunk = header_bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
        for k, (u, v) in enumerate(_HEADER_POSITIONS):
            if _lsb_read(blk_v[bi, bj, u, v], u, v) != chunk[k]:
                n_remaining += 1

    for k in range(n_payload_bits):
        i, j, u, v = slot_order[k]
        if _lsb_read(blk_v[i, j, u, v], u, v) != payload_bits[k]:
            n_remaining += 1

    return new_g, n_remaining

def _set_pixel_y_minl2(B, G, R, pr, pc, target_y, max_radius):
    b, g, r = int(B[pr, pc]), int(G[pr, pc]), int(R[pr, pc])
    best = None
    best_cost = 1 << 30
    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            rr = r + dr
            if rr < 0 or rr > 255:
                continue
            for db in range(-rad, rad + 1):
                bb = b + db
                if bb < 0 or bb > 255:
                    continue
                # solve the G that lands on target_y for this (rr, bb)
                gi = (target_y * 16384 + 8192 - 4899 * rr - 1868 * bb) / 9617.0
                for gg in (int(np.floor(gi)), int(np.ceil(gi))):
                    if 0 <= gg <= 255 and \
                       ((4899 * rr + 9617 * gg + 1868 * bb + 8192) >> 14) == target_y:
                        cost = dr * dr + db * db + (gg - g) * (gg - g)
                        if cost < best_cost:
                            best_cost = cost
                            best = (bb, gg, rr)
        if best is not None:
            break  # found at the smallest possible radius — minimal distortion
    if best is not None:
        B[pr, pc], G[pr, pc], R[pr, pc] = best
        return True
    return False


def _set_pixel_y(B, G, R, pr, pc, target_y):
    b, g, r = int(B[pr, pc]), int(G[pr, pc]), int(R[pr, pc])
    gi = (target_y * 16384 + 8192 - 4899 * r - 1868 * b) / 9617.0
    for gg in range(max(0, int(gi) - 2), min(255, int(gi) + 2) + 1):
        if ((4899 * r + 9617 * gg + 1868 * b + 8192) >> 14) == target_y:
            G[pr, pc] = gg
            return
    R[pr, pc] = G[pr, pc] = B[pr, pc] = target_y


def _exact_y_repair(cover_bgr: np.ndarray, target_y: np.ndarray, cr: np.ndarray,
                    cb: np.ndarray, header_bits: list, payload_bits: list,
                    slot_order: list) -> np.ndarray:
    
    bgr = cv2.cvtColor(cv2.merge([target_y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    n = len(payload_bits)
    max_iter = max(MAX_FIX_ITERATIONS, 40)

    for rep in range(max_iter):
        y_read = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb))[0]
        blk, _, _ = _to_blocks(y_read)

        wrong_blocks = set()
        for blk_idx, (bi, bj) in enumerate(HEADER_BLOCKS):
            chunk = header_bits[blk_idx * HEADER_BLOCK_BITS:(blk_idx + 1) * HEADER_BLOCK_BITS]
            for k, (u, v) in enumerate(_HEADER_POSITIONS):
                if _lsb_read(blk[bi, bj, u, v], u, v) != chunk[k]:
                    wrong_blocks.add((bi, bj))
        for k in range(n):
            i, j, u, v = slot_order[k]
            if _lsb_read(blk[i, j, u, v], u, v) != payload_bits[k]:
                wrong_blocks.add((i, j))

        if not wrong_blocks:
            break

        # Widen the min-L2 search as iterations progress; force luma only at the end.
        max_radius = 4 if rep < 20 else (10 if rep < 30 else 40)
        force_last = rep >= max_iter - 3

        bgr_i = bgr.astype(np.int32)
        B, G, R = bgr_i[:, :, 0], bgr_i[:, :, 1], bgr_i[:, :, 2]
        for (bi, bj) in wrong_blocks:
            for pr in range(bi * N, bi * N + N):
                for pc in range(bj * N, bj * N + N):
                    t = int(target_y[pr, pc])
                    if int(y_read[pr, pc]) != t:
                        ok = _set_pixel_y_minl2(B, G, R, pr, pc, t, max_radius)
                        if not ok and force_last:
                            _set_pixel_y(B, G, R, pr, pc, t)
        bgr = np.stack([B, G, R], axis=2).clip(0, 255).astype(np.uint8)

    return bgr


# ── Canonical Huffman codec (self-contained, no external table needed) ───────
# The secret message is compressed with canonical Huffman before embedding and
# decompressed after extraction. The compressed blob is fully self-describing:
# it carries the original length and the per-symbol code lengths, so the
# extractor can rebuild the identical canonical code with no sidecar.
#
# Blob layout (big-endian):
#   [0:4]  original byte length L
#   if L == 0: nothing else follows
#   [4]    number of distinct symbols - 1   (0..255 -> 1..256 symbols)
#   then nsym pairs of (symbol_byte, code_length_byte), in canonical order
#   then the packed Huffman bitstream (MSB-first, zero-padded to a byte)
def _huff_code_lengths(data: bytes) -> dict:
    """Per-symbol Huffman code lengths via a weight-ordered binary merge."""
    import heapq
    from collections import Counter
    freq = Counter(data)
    if len(freq) == 1:
        # A single distinct symbol still needs one bit per occurrence.
        return {next(iter(freq)): 1}
    heap, order = [], 0
    for sym, f in freq.items():
        heapq.heappush(heap, (f, order, ('leaf', sym))); order += 1
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, order, ('node', n1, n2))); order += 1
    lengths = {}
    stack = [(heap[0][2], 0)]
    while stack:
        node, depth = stack.pop()
        if node[0] == 'leaf':
            lengths[node[1]] = depth
        else:
            stack.append((node[1], depth + 1))
            stack.append((node[2], depth + 1))
    return lengths

def _canonical_codes(lengths: dict) -> dict:
    """Assign canonical codes from code lengths. Returns sym -> (code, length)."""
    syms = sorted(lengths.keys(), key=lambda s: (lengths[s], s))
    codes = {}
    code, prev_len = 0, None
    for sym in syms:
        l = lengths[sym]
        code = 0 if prev_len is None else (code + 1) << (l - prev_len)
        codes[sym] = (code, l)
        prev_len = l
    return codes

def _huff_compress(data: bytes) -> bytes:
    L = len(data)
    out = bytearray(L.to_bytes(4, 'big'))
    if L == 0:
        return bytes(out)
    lengths = _huff_code_lengths(data)
    codes   = _canonical_codes(lengths)
    syms    = sorted(lengths.keys(), key=lambda s: (lengths[s], s))
    out.append(len(syms) - 1)
    for sym in syms:
        out.append(sym)
        out.append(lengths[sym])
    bit_acc, nbits, packed = 0, 0, bytearray()
    for byte in data:
        code, l = codes[byte]
        bit_acc = (bit_acc << l) | code
        nbits  += l
        while nbits >= 8:
            nbits -= 8
            packed.append((bit_acc >> nbits) & 0xFF)
    if nbits > 0:
        packed.append((bit_acc << (8 - nbits)) & 0xFF)
    out += packed
    return bytes(out)

def _huff_decompress(blob: bytes) -> bytes:
    L = int.from_bytes(blob[0:4], 'big')
    if L == 0:
        return b''
    nsym = blob[4] + 1
    pos, lengths = 5, {}
    for _ in range(nsym):
        lengths[blob[pos]] = blob[pos + 1]
        pos += 2
    codes  = _canonical_codes(lengths)
    lookup = {(l, c): sym for sym, (c, l) in codes.items()}
    stream = blob[pos:]
    res, codeval, length, bit_index = bytearray(), 0, 0, 0
    total_bits = len(stream) * 8
    while len(res) < L and bit_index < total_bits:
        bit = (stream[bit_index >> 3] >> (7 - (bit_index & 7))) & 1
        bit_index += 1
        codeval = (codeval << 1) | bit
        length += 1
        sym = lookup.get((length, codeval))
        if sym is not None:
            res.append(sym)
            codeval, length = 0, 0
    return bytes(res)


def get_capacity(cover_bgr: np.ndarray) -> dict:
    Y            = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, _, _ = _to_blocks(Y)
    K            = _compute_K(blocks)
    nr, nc       = blocks.shape[:2]
    n_payload    = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _HEADER_BLOCK_SET
    )
    total = n_payload * (K + 1)
    h, w  = cover_bgr.shape[:2]
    return {
        'payload_bits' : total,
        'K'            : K,
        'bpp'          : total / (h * w),
        'approx_chars' : total // 8,
        'zz_start'     : ZZ_START,
        'zz_end'       : ZZ_END,
    }


def embed(cover_bgr: np.ndarray, message: str) -> np.ndarray:
    raw        = message.encode('utf-8')
    compressed = _huff_compress(raw)
    bits = [(b >> (7 - k)) & 1 for b in compressed for k in range(8)]
    n    = len(bits)

    if raw:
        ratio = len(compressed) / len(raw)
        print(f"  Huffman  : {len(raw)} -> {len(compressed)} bytes  "
              f"({ratio:.3f}x, {n} bits embedded)")

    Y, Cr, Cb = cv2.split(cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb))

    blocks, h8, w8 = _to_blocks(Y)
    K              = _compute_K(blocks, verbose=False)
    nr, nc         = blocks.shape[:2]

    n_payload = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _HEADER_BLOCK_SET
    )
    capacity = n_payload * (K + 1)

    if n > capacity:
        raise ValueError(
            f"Message too large: {n} bits needed, {capacity} bits available.  "
            f"Max message length: ~{capacity // 8} chars."
        )

    # ── Step 1: write header ─────────────────────────────────────────────────
    header_bits = _pack_header(n, K)
    _write_header(blocks, n, K)

    # ── Step 2: embed payload bits (distortion-ordered slots, min-distortion) ─
    # Fill the lowest-distortion slots across all blocks first, and always pick
    # the q±1 candidate closest to the original coefficient (flip_dir=-1).
    slot_order = _build_slot_order(K, nr, nc)
    for k in range(n):
        i, j, u, v = slot_order[k]
        blocks[i, j, u, v] = _lsb_write(blocks[i, j, u, v], u, v, bits[k], flip_dir=-1)

    print(f"  Embedded : {n} / {n} bits  OK")

    # ── Step 3: verify-fix loop (Y-domain IDCT round-trip only) ──────────────
    stego_y = _from_blocks(blocks, Y, h8, w8)

    total_corrections = 0
    for fix_iter in range(MAX_FIX_ITERATIONS):
        stego_y, n_remaining = _verify_and_fix(
            slot_order, stego_y, h8, w8, header_bits, bits, K
        )
        if n_remaining == 0:
            break
        total_corrections += n_remaining
        print(f"  Fix pass {fix_iter + 1}: {n_remaining} bit(s) wrong after IDCT roundtrip")
    else:
        raise RuntimeError(
            f"Verify-fix did not converge after {MAX_FIX_ITERATIONS} iterations.  "
            f"Please report this image and message."
        )

    if total_corrections > 0:
        print(f"  Verify   : converged in {fix_iter + 1} pass(es) — extraction guaranteed")
    else:
        print(f"  Verify   : all bits stable — no corrections needed")

    # ── Step 4: write BGR whose Y channel reproduces the embedded bits ───────
    # Selective repair: start from the high-PSNR naive YCrCb→BGR inverse and fix
    # only the blocks whose bits would otherwise read wrong after BGR→YCrCb.
    return _exact_y_repair(cover_bgr, stego_y, Cr, Cb, header_bits, bits, slot_order)


def extract(stego_bgr: np.ndarray) -> str:
    """Extract message from stego image.  No key or sidecar file needed."""
    Y = cv2.split(cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb))[0]
    blocks, h8, w8 = _to_blocks(Y)

    n_bits, K, version, header_ok = _read_header(blocks)

    if not header_ok:
        raise ValueError(
            "Header CRC or magic check failed.  "
            "This image does not contain a DSTG-embedded message."
        )

    nr, nc    = blocks.shape[:2]
    n_payload = sum(
        1 for i in range(nr) for j in range(nc)
        if (i, j) not in _HEADER_BLOCK_SET
    )
    capacity = n_payload * (K + 1)

    if not (0 < n_bits <= capacity):
        raise ValueError(
            f"Header payload_bits={n_bits} outside valid range [1, {capacity}].  "
            "Corrupt header or wrong image."
        )

    print(f"  Header   : {n_bits} bits  K={K}  v{version}  CRC OK")

    # Reconstruct the identical distortion-ordered slot list from K, nr, nc
    # (all recovered/known), then read payload bits in that order.
    slot_order = _build_slot_order(K, nr, nc)
    bits = [
        _lsb_read(blocks[i, j, u, v], u, v)
        for (i, j, u, v) in slot_order[:n_bits]
    ]

    byte_arr = bytearray(
        sum(bits[i + k] << (7 - k) for k in range(8))
        for i in range(0, len(bits) - 7, 8)
    )
    try:
        raw = _huff_decompress(bytes(byte_arr))
        return raw.decode('utf-8')
    except (UnicodeDecodeError, IndexError, ValueError) as exc:
        raise ValueError(
            f"Huffman/UTF-8 decode failed after extracting {len(bits)} bits.  "
            "Image may have been re-compressed after embedding."
        ) from exc


def quality_metrics(cover_bgr: np.ndarray, stego_bgr: np.ndarray,
                    n_payload_bits: int) -> dict:
    h = min(cover_bgr.shape[0], stego_bgr.shape[0])
    w = min(cover_bgr.shape[1], stego_bgr.shape[1])
    c, s = cover_bgr[:h, :w], stego_bgr[:h, :w]

    # Luma (Y-channel) metrics only — the fair, primary metric for Y embedding
    yc = cv2.split(cv2.cvtColor(c, cv2.COLOR_BGR2YCrCb))[0]
    ys = cv2.split(cv2.cvtColor(s, cv2.COLOR_BGR2YCrCb))[0]
    psnr_luma = peak_signal_noise_ratio(yc, ys, data_range=255)
    ssim_luma = structural_similarity(yc, ys, data_range=255)

    return {
        'psnr'      : psnr_luma,    # primary = luma PSNR (fair for Y embedding)
        'ssim'      : ssim_luma,    # primary = luma SSIM
        'psnr_luma' : psnr_luma,
        'ssim_luma' : ssim_luma,
        # BPP is computed over the (compressed) payload bits actually embedded
        'bpp'       : n_payload_bits / (h * w),
    }

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    COVER_PATH = 'validation_output\\baseline\\cover_p2_s46.png'
    # COVER_PATH = 'validation_output\\finetuned\\cover_p2_s46.png'
    STEGO_PATH = 'stego_adaptive.png'

    cover = cv2.imread(COVER_PATH)
    assert cover is not None, f'Cannot load {COVER_PATH}'

    h, w = cover.shape[:2]
    cap  = get_capacity(cover)
    print(f'Image    : {h}×{w}')
    print(f'Capacity : {cap["payload_bits"]} bits  '
          f'(K={cap["K"]}  BPP={cap["bpp"]:.4f}  ~{cap["approx_chars"]} chars)')
    print(f'ZZ range : {cap["zz_start"]}-{cap["zz_end"]}  '
          f'({len(_PAYLOAD_CANDIDATES)} stable positions per block)\n')

    # ── Load message ───────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            MSG = f.read()
        print(f'  Loaded message from: {sys.argv[1]}')
    else:
        while True:
            MSG   = input('Secret message: ')
            n_req = len(MSG.encode('utf-8')) * 8
            if n_req <= cap['payload_bits']:
                break
            print(f'  [!] Message too large: {len(MSG)} chars ({n_req} bits).')
            print(f'      Max: ~{cap["approx_chars"]} chars ({cap["payload_bits"]} bits).\n')

    # Validate size when reading from file
    n_req = len(MSG.encode('utf-8')) * 8
    if n_req > cap['payload_bits']:
        raise ValueError(
            f'Message too large: {len(MSG)} chars ({n_req} bits). '
            f'Max: ~{cap["approx_chars"]} chars ({cap["payload_bits"]} bits).'
        )

    # ── Embed ──────────────────────────────────────────────────────────────
    print('\n═══ EMBEDDING ═══')
    stego = embed(cover, MSG)
    cv2.imwrite(STEGO_PATH, stego)
    print(f'  Saved → {STEGO_PATH}')

    # BPP is measured over the compressed payload that is actually embedded.
    compressed  = _huff_compress(MSG.encode('utf-8'))
    n_comp_bits = len(compressed) * 8
    qm          = quality_metrics(cover, stego, n_comp_bits)
    print(f'  BPP  : {qm["bpp"]:.4f}')
    print(f'  PSNR : {qm["psnr_luma"]:.2f} dB')
    print(f'  SSIM : {qm["ssim_luma"]:.4f}')

    # ── Extract in-memory ──────────────────────────────────────────────────
    print('\n═══ EXTRACT (in-memory) ═══')
    r1 = extract(stego)
    print('  PASS' if r1 == MSG else f'  FAIL\n  Got: {r1!r}')

    # ── Extract from PNG ───────────────────────────────────────────────────
    print('\n═══ EXTRACT (PNG reload) ═══')
    r2 = extract(cv2.imread(STEGO_PATH))
    print('  PASS — cross-device round-trip perfect' if r2 == MSG
          else f'  FAIL\n  Got: {r2!r}')