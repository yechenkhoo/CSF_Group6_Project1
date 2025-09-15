import argparse, hashlib, struct, sys, wave, os
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from PIL import Image

MAGIC = b"INF2"
VERSION = 1
COV_IMAGE = 0
COV_AUDIO = 1

@dataclass
class Header:
    magic: bytes
    version: int
    cover_type: int
    lsb_count: int
    payload_len: int
    payload_sha256: bytes

    def pack(self) -> bytes:
        return MAGIC + struct.pack(
            ">BBBBQ", VERSION, self.cover_type, self.lsb_count, 0, self.payload_len
        ) + self.payload_sha256
        # Note: the single zero byte is reserved for future options.

    @staticmethod
    def unpack(b: bytes) -> "Header":
        if len(b) < 4 + 1 + 1 + 1 + 1 + 8 + 32:
            raise ValueError("Header too short")
        if b[:4] != MAGIC:
            raise ValueError("Bad magic")
        version, cover, lsb, _opt, plen = struct.unpack(">BBBBQ", b[4:4+1+1+1+1+8])
        sha = b[4+1+1+1+1+8:4+1+1+1+1+8+32]
        return Header(MAGIC, version, cover, lsb, plen, sha)

HEADER_BYTES = 4 + (1+1+1+1+8) + 32
# Bits needed depend on lsb count chosen at encode-time.

# ---------- Key schedule ----------
def seed_from_key(key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h, "big")

def traversal_indices(n_slots: int, seed: int, start_at: Optional[int]=None) -> np.ndarray:
    """Return a deterministic permutation of [0..n_slots-1] with a key-based rotation."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_slots, dtype=np.uint64)
    rng.shuffle(idx)
    if start_at is None:
        start_at = seed % n_slots
    # rotate so that we start at (start_at)
    return np.concatenate([idx[start_at:], idx[:start_at]])

# ---------- Bit helpers ----------
def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    # big-endian bit order inside each byte (MSB..LSB)
    bits = np.unpackbits(arr)
    return bits

def bits_to_bytes(bits: np.ndarray) -> bytes:
    # pad to multiple of 8
    if bits.size % 8 != 0:
        pad = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

def pack_stream_for_lsb(bitstream: np.ndarray, lsb: int) -> Tuple[np.ndarray, int]:
    """Group bits into chunks of 'lsb' per cover slot. Returns (chunks, n_slots_used)."""
    if lsb < 1 or lsb > 8:
        raise ValueError("lsb must be 1..8")
    n = bitstream.size
    rem = n % lsb
    if rem != 0:
        pad = lsb - rem
        bitstream = np.concatenate([bitstream, np.zeros(pad, dtype=np.uint8)])
    grouped = bitstream.reshape(-1, lsb)
    # interpret each row as an integer (MSB..LSB inside the chunk)
    weights = (1 << np.arange(lsb-1, -1, -1)).astype(np.uint16)  # e.g., lsb=3 -> [4,2,1]
    vals = (grouped * weights).sum(axis=1).astype(np.uint16)
    return vals, vals.size

def unpack_stream_from_lsb(vals: np.ndarray, total_bits: int, lsb: int) -> np.ndarray:
    """Inverse of pack_stream_for_lsb."""
    # vals are in [0 .. (1<<lsb)-1]
    out = np.zeros((vals.size, lsb), dtype=np.uint8)
    for i in range(lsb):
        shift = (lsb - 1 - i)
        out[:, i] = (vals >> shift) & 1
    bits = out.reshape(-1)
    return bits[:total_bits]

# ---------- Capacity ----------
def capacity_bits_image(img: np.ndarray, lsb: int) -> int:
    return img.size * lsb  # bytes == channels * H * W; each byte gets 'lsb' bits

def capacity_bits_audio(samples: np.ndarray, lsb: int) -> int:
    return samples.size * lsb  # each int16 sample holds 'lsb' bits

# ---------- Image I/O ----------
def load_image_bytes(path: str) -> Tuple[np.ndarray, Tuple[int,int,int], str]:
    im = Image.open(path).convert("RGBA" if path.lower().endswith(".png") else "RGB")
    arr = np.array(im, dtype=np.uint8)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    return arr, arr.shape, mode

def save_image_bytes(path: str, arr: np.ndarray, mode_hint: str):
    im = Image.fromarray(arr.astype(np.uint8), mode="RGBA" if arr.shape[-1]==4 else "RGB")
    # Preserve extension's format
    im.save(path)

# ---------- Audio I/O (16-bit PCM) ----------
def load_wav_int16(path: str) -> Tuple[np.ndarray, int, int]:
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM WAV supported")
        raw = wf.readframes(n_frames)
    data = np.frombuffer(raw, dtype=np.int16)
    # keep interleaved layout; shape = (n_frames * n_ch,)
    return data.copy(), n_ch, fr  # copy to make it writable

def save_wav_int16(path: str, data: np.ndarray, n_channels: int, fr: int):
    if data.dtype != np.int16:
        data = data.astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(fr)
        wf.writeframes(data.tobytes())

# ---------- Core Embed / Extract ----------
def do_embed_image(cover_path: str, payload_path: str, out_path: str, key: str, lsb: int):
    img, shape, mode = load_image_bytes(cover_path)
    flat = img.reshape(-1)  # uint8
    seed = seed_from_key(key)
    idx = traversal_indices(flat.size, seed)

    payload = open(payload_path, "rb").read()
    h = Header(MAGIC, VERSION, COV_IMAGE, lsb, len(payload), hashlib.sha256(payload).digest())
    header_bytes = h.pack()

    # Build bitstream: header + payload
    bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
    chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

    cap_bits = capacity_bits_image(img, lsb)
    if needed_slots > flat.size:
        need = (needed_slots * lsb + 7)//8
        cap = cap_bits // 8
        raise ValueError(f"Payload requires ~{need} bytes but capacity is {cap} bytes.")

    # Write chunks into LSBs along permutation
    mask = np.uint8(0xFF ^ ((1 << lsb) - 1))
    target = flat.copy()
    sel = idx[:needed_slots].astype(np.int64)
    target[sel] = (target[sel] & mask) | chunks.astype(np.uint8)
    stego = target.reshape(shape)
    save_image_bytes(out_path, stego, mode)
    print(f"Embedded {len(payload)} bytes into image -> {out_path}")

def do_extract_image(stego_path: str, out_payload_path: str, key: str, lsb: int):
    img, shape, mode = load_image_bytes(stego_path)
    flat = img.reshape(-1)
    seed = seed_from_key(key)
    idx = traversal_indices(flat.size, seed)

    # First, read header bits
    hdr_bits_needed = HEADER_BYTES * 8
    # number of cover slots (values) to read to get header, each slot contributes 'lsb' bits
    slots_for_hdr = (hdr_bits_needed + lsb - 1) // lsb
    sel_hdr = idx[:slots_for_hdr].astype(np.int64)
    vals_hdr = (flat[sel_hdr] & ((1 << lsb) - 1)).astype(np.uint16)
    hdr_bits = unpack_stream_from_lsb(vals_hdr, hdr_bits_needed, lsb)
    hdr = Header.unpack(bits_to_bytes(hdr_bits))

    if hdr.cover_type != COV_IMAGE or hdr.lsb_count != lsb:
        raise ValueError("Wrong key/cover/lsb settings (header mismatch).")

    total_payload_bits = hdr.payload_len * 8
    slots_for_payload = (total_payload_bits + lsb - 1) // lsb
    sel_pl = idx[slots_for_hdr:slots_for_hdr + slots_for_payload].astype(np.int64)
    vals_pl = (flat[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
    pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
    payload = bits_to_bytes(pay_bits)

    if hashlib.sha256(payload).digest() != hdr.payload_sha256:
        raise ValueError("Integrity check failed (wrong key or corrupted stego).")

    open(out_payload_path, "wb").write(payload)
    print(f"Extracted {len(payload)} bytes from image -> {out_payload_path}")

def do_embed_audio(cover_path: str, payload_path: str, out_path: str, key: str, lsb: int):
    samples, n_ch, fr = load_wav_int16(cover_path)
    # Work with uint16 to avoid sign issues when masking
    buf = samples.view(np.uint16)
    seed = seed_from_key(key)
    idx = traversal_indices(buf.size, seed)

    payload = open(payload_path, "rb").read()
    h = Header(MAGIC, VERSION, COV_AUDIO, lsb, len(payload), hashlib.sha256(payload).digest())
    header_bytes = h.pack()

    bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
    chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

    cap_bits = capacity_bits_audio(buf, lsb)
    if needed_slots > buf.size:
        need = (needed_slots * lsb + 7)//8
        cap = cap_bits // 8
        raise ValueError(f"Payload requires ~{need} bytes but capacity is {cap} bytes.")

    mask = np.uint16(0xFFFF ^ ((1 << lsb) - 1))
    target = buf.copy()
    sel = idx[:needed_slots].astype(np.int64)
    target[sel] = (target[sel] & mask) | chunks.astype(np.uint16)
    # Save back as int16
    out_i16 = target.view(np.int16)
    save_wav_int16(out_path, out_i16, n_ch, fr)
    print(f"Embedded {len(payload)} bytes into audio -> {out_path}")

def do_extract_audio(stego_path: str, out_payload_path: str, key: str, lsb: int):
    samples, n_ch, fr = load_wav_int16(stego_path)
    buf = samples.view(np.uint16)
    seed = seed_from_key(key)
    idx = traversal_indices(buf.size, seed)

    hdr_bits_needed = HEADER_BYTES * 8
    slots_for_hdr = (hdr_bits_needed + lsb - 1) // lsb
    sel_hdr = idx[:slots_for_hdr].astype(np.int64)
    vals_hdr = (buf[sel_hdr] & ((1 << lsb) - 1)).astype(np.uint16)
    hdr_bits = unpack_stream_from_lsb(vals_hdr, hdr_bits_needed, lsb)
    hdr = Header.unpack(bits_to_bytes(hdr_bits))

    if hdr.cover_type != COV_AUDIO or hdr.lsb_count != lsb:
        raise ValueError("Wrong key/cover/lsb settings (header mismatch).")

    total_payload_bits = hdr.payload_len * 8
    slots_for_payload = (total_payload_bits + lsb - 1) // lsb
    sel_pl = idx[slots_for_hdr:slots_for_hdr + slots_for_payload].astype(np.int64)
    vals_pl = (buf[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
    pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
    payload = bits_to_bytes(pay_bits)

    if hashlib.sha256(payload).digest() != hdr.payload_sha256:
        raise ValueError("Integrity check failed (wrong key or corrupted stego).")

    open(out_payload_path, "wb").write(payload)
    print(f"Extracted {len(payload)} bytes from audio -> {out_payload_path}")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Minimal LSB stego CLI (image/audio)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="Embed payload into cover")
    e.add_argument("--cover", required=True, help="Cover file (.png/.bmp or .wav)")
    e.add_argument("--payload", required=True, help="Payload file (any bytes)")
    e.add_argument("--out", required=True, help="Output stego file")
    e.add_argument("--key", required=True, help="Key (string)")
    e.add_argument("--lsb", type=int, default=2, help="LSBs to use (1-8)")
    # TODO: --region options later: image rectangle or audio sample range

    d = sub.add_parser("decode", help="Extract payload from stego")
    d.add_argument("--stego", required=True, help="Stego file (.png/.bmp or .wav)")
    d.add_argument("--out", required=True, help="Where to write extracted payload")
    d.add_argument("--key", required=True, help="Key (string)")
    d.add_argument("--lsb", type=int, default=2, help="LSBs used (must match)")

    args = p.parse_args()

    if args.cmd == "encode":
        ext = os.path.splitext(args.cover)[1].lower()
        if ext in (".png", ".bmp"):
            do_embed_image(args.cover, args.payload, args.out, args.key, args.lsb)
        elif ext == ".wav":
            do_embed_audio(args.cover, args.payload, args.out, args.key, args.lsb)
        else:
            sys.exit("Unsupported cover type. Use PNG/BMP or 16-bit PCM WAV.")
    else:
        ext = os.path.splitext(args.stego)[1].lower()
        if ext in (".png", ".bmp"):
            do_extract_image(args.stego, args.out, args.key, args.lsb)
        elif ext == ".wav":
            do_extract_audio(args.stego, args.out, args.key, args.lsb)
        else:
            sys.exit("Unsupported stego type. Use PNG/BMP or 16-bit PCM WAV.")

if __name__ == "__main__":
    main()
