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
        return (
            MAGIC
            + struct.pack(
                ">BBBBQ", VERSION, self.cover_type, self.lsb_count, 0, self.payload_len
            )
            + self.payload_sha256
        )
        # Note: the single zero byte is reserved for future options.

    @staticmethod
    def unpack(b: bytes) -> "Header":
        if len(b) < 4 + 1 + 1 + 1 + 1 + 8 + 32:
            raise ValueError("Header too short")
        if b[:4] != MAGIC:
            raise ValueError("Bad magic")
        version, cover, lsb, _opt, plen = struct.unpack(
            ">BBBBQ", b[4 : 4 + 1 + 1 + 1 + 1 + 8]
        )
        sha = b[4 + 1 + 1 + 1 + 1 + 8 : 4 + 1 + 1 + 1 + 1 + 8 + 32]
        return Header(MAGIC, version, cover, lsb, plen, sha)


HEADER_BYTES = 4 + (1 + 1 + 1 + 1 + 8) + 32
# Bits needed depend on lsb count chosen at encode-time.


# ---------- Key schedule ----------
def seed_from_key(key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h, "big")


def traversal_indices(
    n_slots: int, seed: int, start_at: Optional[int] = None
) -> np.ndarray:
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
    weights = (1 << np.arange(lsb - 1, -1, -1)).astype(
        np.uint16
    )  # e.g., lsb=3 -> [4,2,1]
    vals = (grouped * weights).sum(axis=1).astype(np.uint16)
    return vals, vals.size


def unpack_stream_from_lsb(vals: np.ndarray, total_bits: int, lsb: int) -> np.ndarray:
    """Inverse of pack_stream_for_lsb."""
    # vals are in [0 .. (1<<lsb)-1]
    out = np.zeros((vals.size, lsb), dtype=np.uint8)
    for i in range(lsb):
        shift = lsb - 1 - i
        out[:, i] = (vals >> shift) & 1
    bits = out.reshape(-1)
    return bits[:total_bits]


# ---------- Capacity ----------
def capacity_bits_image(img: np.ndarray, lsb: int) -> int:
    return img.size * lsb  # bytes == channels * H * W; each byte gets 'lsb' bits


def capacity_bits_audio(samples: np.ndarray, lsb: int) -> int:
    return samples.size * lsb  # each int16 sample holds 'lsb' bits


# ---------- Image I/O ----------
def load_image_bytes(path: str) -> Tuple[np.ndarray, Tuple[int, int, int], str]:
    im = Image.open(path).convert("RGBA" if path.lower().endswith(".png") else "RGB")
    arr = np.array(im, dtype=np.uint8)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    return arr, arr.shape, mode


def save_image_bytes(path: str, arr: np.ndarray, mode_hint: str):
    im = Image.fromarray(
        arr.astype(np.uint8), mode="RGBA" if arr.shape[-1] == 4 else "RGB"
    )
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


# ---------- Region Selection ----------
def get_region_indices(img_shape, region):
    """Get flat indices for a rectangular region"""
    if region is None:
        return np.arange(np.prod(img_shape))

    h, w, c = img_shape
    x, y, rw, rh = region["x"], region["y"], region["width"], region["height"]

    # Ensure region bounds are within image
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = min(rw, w - x)
    rh = min(rh, h - y)

    indices = []
    for row in range(y, y + rh):
        for col in range(x, x + rw):
            for ch in range(c):
                flat_idx = (row * w + col) * c + ch
                indices.append(flat_idx)

    return np.array(indices)


def calculate_region_capacity(img_shape, region, lsb):
    """Calculate capacity for a specific region"""
    if region is None:
        return np.prod(img_shape) * lsb // 8

    h, w, c = img_shape
    x, y, rw, rh = region["x"], region["y"], region["width"], region["height"]

    # checks if region bounds are within image
    rw = min(rw, w - x)
    rh = min(rh, h - y)

    region_pixels = rw * rh * c
    return region_pixels * lsb // 8


# ---------- Audio Time Range Selection ----------
def time_to_sample_indices(time_range, sample_rate, n_channels, total_samples):
    """Convert time range to sample indices"""
    if time_range is None:
        return np.arange(total_samples)
    
    start_time = time_range["start_time"]
    end_time = time_range["end_time"]
    
    # Convert time to sample indices
    start_sample = int(start_time * sample_rate * n_channels)
    end_sample = int(end_time * sample_rate * n_channels)
    
    # Ensure bounds are within audio
    start_sample = max(0, min(start_sample, total_samples - 1))
    end_sample = max(start_sample + 1, min(end_sample, total_samples))
    
    return np.arange(start_sample, end_sample)


def calculate_audio_time_capacity(audio_path, time_range, lsb):
    """Calculate capacity for a specific time range in audio"""
    if time_range is None:
        with wave.open(audio_path, "rb") as wf:
            n_ch = wf.getnchannels()
            n_frames = wf.getnframes()
            total_samples = n_frames * n_ch
        return total_samples * lsb // 8
    
    with wave.open(audio_path, "rb") as wf:
        n_ch = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        total_samples = n_frames * n_ch
    
    # Get sample indices for time range
    indices = time_to_sample_indices(time_range, sample_rate, n_ch, total_samples)
    
    return len(indices) * lsb // 8


# ---------- Core Embed / Extract ----------
def do_embed_image(
    cover_path: str, payload_path: str, out_path: str, key: str, lsb: int
):
    img, shape, mode = load_image_bytes(cover_path)
    flat = img.reshape(-1)  # uint8
    seed = seed_from_key(key)
    idx = traversal_indices(flat.size, seed)

    payload = open(payload_path, "rb").read()
    h = Header(
        MAGIC, VERSION, COV_IMAGE, lsb, len(payload), hashlib.sha256(payload).digest()
    )
    header_bytes = h.pack()

    # Build bitstream: header + payload
    bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
    chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

    cap_bits = capacity_bits_image(img, lsb)
    if needed_slots > flat.size:
        need = (needed_slots * lsb + 7) // 8
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
    sel_pl = idx[slots_for_hdr : slots_for_hdr + slots_for_payload].astype(np.int64)
    vals_pl = (flat[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
    pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
    payload = bits_to_bytes(pay_bits)

    if hashlib.sha256(payload).digest() != hdr.payload_sha256:
        raise ValueError("Integrity check failed (wrong key or corrupted stego).")

    open(out_payload_path, "wb").write(payload)
    print(f"Extracted {len(payload)} bytes from image -> {out_payload_path}")


def do_embed_image_region(
    cover_path: str, payload_path: str, out_path: str, key: str, lsb: int, region=None
):
    """Embed payload into image with optional region selection"""
    img, shape, mode = load_image_bytes(cover_path)
    flat = img.reshape(-1)
    seed = seed_from_key(key)

    # gets indices for embedding region
    if region:
        available_indices = get_region_indices(shape, region)
        if len(available_indices) == 0:
            raise ValueError("Selected region is empty")
        idx = traversal_indices(len(available_indices), seed)
        # need to map back to original flat indices
        idx = available_indices[idx]
    else:
        idx = traversal_indices(flat.size, seed)

    payload = open(payload_path, "rb").read()
    h = Header(
        MAGIC, VERSION, COV_IMAGE, lsb, len(payload), hashlib.sha256(payload).digest()
    )
    header_bytes = h.pack()

    # builds bitstream: header + payload
    bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
    chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

    if needed_slots > len(idx):
        need = (needed_slots * lsb + 7) // 8
        cap = (len(idx) * lsb) // 8
        region_info = (
            f" (region {region['width']}×{region['height']})" if region else ""
        )
        raise ValueError(
            f"Payload requires ~{need} bytes but capacity is {cap} bytes{region_info}."
        )

    # Write chunks into LSBs along permutation
    mask = np.uint8(0xFF ^ ((1 << lsb) - 1))
    target = flat.copy()
    sel = idx[:needed_slots].astype(np.int64)
    target[sel] = (target[sel] & mask) | chunks.astype(np.uint8)
    stego = target.reshape(shape)
    save_image_bytes(out_path, stego, mode)
    region_info = f" (region {region['width']}×{region['height']})" if region else ""
    print(f"Embedded {len(payload)} bytes into image{region_info} -> {out_path}")


def do_extract_image_region(
    stego_path: str, out_payload_path: str, key: str, lsb: int, region=None
):
    """Extract payload from image with optional region selection"""
    img, shape, mode = load_image_bytes(stego_path)
    flat = img.reshape(-1)
    seed = seed_from_key(key)

    # Get indices for extraction region (must match embedding)
    if region:
        available_indices = get_region_indices(shape, region)
        if len(available_indices) == 0:
            raise ValueError("Selected region is empty")
        idx = traversal_indices(len(available_indices), seed)
        # Map back to original flat indices
        idx = available_indices[idx]
    else:
        idx = traversal_indices(flat.size, seed)

    # First, read header bits
    hdr_bits_needed = HEADER_BYTES * 8
    slots_for_hdr = (hdr_bits_needed + lsb - 1) // lsb

    if slots_for_hdr > len(idx):
        raise ValueError("Not enough capacity to read header from selected region")

    sel_hdr = idx[:slots_for_hdr].astype(np.int64)
    vals_hdr = (flat[sel_hdr] & ((1 << lsb) - 1)).astype(np.uint16)
    hdr_bits = unpack_stream_from_lsb(vals_hdr, hdr_bits_needed, lsb)
    hdr = Header.unpack(bits_to_bytes(hdr_bits))

    if hdr.cover_type != COV_IMAGE or hdr.lsb_count != lsb:
        raise ValueError("Wrong key/cover/lsb settings (header mismatch).")

    total_payload_bits = hdr.payload_len * 8
    slots_for_payload = (total_payload_bits + lsb - 1) // lsb

    if slots_for_hdr + slots_for_payload > len(idx):
        raise ValueError("Not enough capacity to read payload from selected region")

    sel_pl = idx[slots_for_hdr : slots_for_hdr + slots_for_payload].astype(np.int64)
    vals_pl = (flat[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
    pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
    payload = bits_to_bytes(pay_bits)

    if hashlib.sha256(payload).digest() != hdr.payload_sha256:
        raise ValueError("Integrity check failed (wrong key or corrupted stego).")

    open(out_payload_path, "wb").write(payload)
    region_info = f" (region {region['width']}×{region['height']})" if region else ""
    print(
        f"Extracted {len(payload)} bytes from image{region_info} -> {out_payload_path}"
    )


def do_embed_audio(
    cover_path: str, payload_path: str, out_path: str, key: str, lsb: int
):
    samples, n_ch, fr = load_wav_int16(cover_path)
    # Work with uint16 to avoid sign issues when masking
    buf = samples.view(np.uint16)
    seed = seed_from_key(key)
    idx = traversal_indices(buf.size, seed)

    payload = open(payload_path, "rb").read()
    h = Header(
        MAGIC, VERSION, COV_AUDIO, lsb, len(payload), hashlib.sha256(payload).digest()
    )
    header_bytes = h.pack()

    bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
    chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

    cap_bits = capacity_bits_audio(buf, lsb)
    if needed_slots > buf.size:
        need = (needed_slots * lsb + 7) // 8
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
    sel_pl = idx[slots_for_hdr : slots_for_hdr + slots_for_payload].astype(np.int64)
    vals_pl = (buf[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
    pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
    payload = bits_to_bytes(pay_bits)

    if hashlib.sha256(payload).digest() != hdr.payload_sha256:
        raise ValueError("Integrity check failed (wrong key or corrupted stego).")

    open(out_payload_path, "wb").write(payload)
    print(f"Extracted {len(payload)} bytes from audio -> {out_payload_path}")


def do_embed_audio_region(
    cover_path: str, payload_path: str, out_path: str, key: str, lsb: int, time_range=None
):
    """Embed payload into audio with optional time range selection"""
    samples, n_ch, fr = load_wav_int16(cover_path)
    buf = samples.view(np.uint16)
    seed = seed_from_key(key)

    # Get indices for embedding time range
    if time_range:
        available_indices = time_to_sample_indices(time_range, fr, n_ch, buf.size)
        if len(available_indices) == 0:
            raise ValueError("Selected time range is empty")
        idx = traversal_indices(len(available_indices), seed)
        # Map back to original sample indices
        idx = available_indices[idx]
    else:
        idx = traversal_indices(buf.size, seed)

    payload = open(payload_path, "rb").read()
    h = Header(
        MAGIC, VERSION, COV_AUDIO, lsb, len(payload), hashlib.sha256(payload).digest()
    )
    header_bytes = h.pack()

    # Build bitstream: header + payload
    bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
    chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

    if needed_slots > len(idx):
        need = (needed_slots * lsb + 7) // 8
        cap = (len(idx) * lsb) // 8
        time_info = (
            f" (time {time_range['start_time']:.1f}s-{time_range['end_time']:.1f}s)" 
            if time_range else ""
        )
        raise ValueError(
            f"Payload requires ~{need} bytes but capacity is {cap} bytes{time_info}."
        )

    # Write chunks into LSBs along permutation
    mask = np.uint16(0xFFFF ^ ((1 << lsb) - 1))
    target = buf.copy()
    sel = idx[:needed_slots].astype(np.int64)
    target[sel] = (target[sel] & mask) | chunks.astype(np.uint16)
    
    # Save back as int16
    out_i16 = target.view(np.int16)
    save_wav_int16(out_path, out_i16, n_ch, fr)
    
    time_info = (
        f" (time {time_range['start_time']:.1f}s-{time_range['end_time']:.1f}s)" 
        if time_range else ""
    )
    print(f"Embedded {len(payload)} bytes into audio{time_info} -> {out_path}")


def do_extract_audio_region(
    stego_path: str, out_payload_path: str, key: str, lsb: int, time_range=None
):
    """Extract payload from audio with optional time range selection"""
    samples, n_ch, fr = load_wav_int16(stego_path)
    buf = samples.view(np.uint16)
    seed = seed_from_key(key)

    # Get indices for extraction time range (must match embedding)
    if time_range:
        available_indices = time_to_sample_indices(time_range, fr, n_ch, buf.size)
        if len(available_indices) == 0:
            raise ValueError("Selected time range is empty")
        idx = traversal_indices(len(available_indices), seed)
        # Map back to original sample indices
        idx = available_indices[idx]
    else:
        idx = traversal_indices(buf.size, seed)

    # First, read header bits
    hdr_bits_needed = HEADER_BYTES * 8
    slots_for_hdr = (hdr_bits_needed + lsb - 1) // lsb

    if slots_for_hdr > len(idx):
        raise ValueError("Not enough capacity to read header from selected time range")

    sel_hdr = idx[:slots_for_hdr].astype(np.int64)
    vals_hdr = (buf[sel_hdr] & ((1 << lsb) - 1)).astype(np.uint16)
    hdr_bits = unpack_stream_from_lsb(vals_hdr, hdr_bits_needed, lsb)
    hdr = Header.unpack(bits_to_bytes(hdr_bits))

    if hdr.cover_type != COV_AUDIO or hdr.lsb_count != lsb:
        raise ValueError("Wrong key/cover/lsb settings (header mismatch).")

    total_payload_bits = hdr.payload_len * 8
    slots_for_payload = (total_payload_bits + lsb - 1) // lsb

    if slots_for_hdr + slots_for_payload > len(idx):
        raise ValueError("Not enough capacity to read payload from selected time range")

    sel_pl = idx[slots_for_hdr : slots_for_hdr + slots_for_payload].astype(np.int64)
    vals_pl = (buf[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
    pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
    payload = bits_to_bytes(pay_bits)

    if hashlib.sha256(payload).digest() != hdr.payload_sha256:
        raise ValueError("Integrity check failed (wrong key or corrupted stego).")

    open(out_payload_path, "wb").write(payload)
    
    time_info = (
        f" (time {time_range['start_time']:.1f}s-{time_range['end_time']:.1f}s)" 
        if time_range else ""
    )
    print(f"Extracted {len(payload)} bytes from audio{time_info} -> {out_payload_path}")


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

# ---------- Video support (experimental, lossless pipeline) ----------
try:
    import imageio.v2 as imageio  # imageio provides ffmpeg-backed readers/writers
except Exception as _e:
    imageio = None  # Will raise at runtime if used without dependency

# Define video cover type without touching existing constants
COV_VIDEO = 2


def _load_video_frames_rgb(video_path: str):
    """Yield frames as uint8 RGB arrays and return fps, size via side-channel.

    Returns a tuple (frames_iterable, meta) where meta = {"fps": float, "size": (w, h), "n_frames": int}
    """
    if imageio is None:
        raise RuntimeError("imageio is required for video support. Please install imageio[ffmpeg].")

    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30)
    size = None
    n_frames = meta.get("nframes", None)

    def _iter():
        nonlocal size
        for frame in reader:
            # frame is HxWx3 uint8 in RGB order
            if size is None:
                size = (frame.shape[1], frame.shape[0])
            yield frame.astype(np.uint8, copy=False)
        reader.close()

    frames_iter = _iter()
    return frames_iter, {"fps": fps, "size": size, "n_frames": n_frames}


def _save_video_frames_rgb(out_path: str, frames_iter, fps: float):
    """Write RGB uint8 frames using a mathematically lossless pipeline.

    We use libx264rgb with -crf 0 and rgb24 pixel format to preserve exact bytes.
    Requires ffmpeg with libx264 support.
    """
    if imageio is None:
        raise RuntimeError("imageio is required for video support. Please install imageio[ffmpeg].")

    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264rgb",
        format="FFMPEG",
        quality=None,
        ffmpeg_params=["-crf", "0", "-pix_fmt", "rgb24", "-preset", "veryslow"],
        macro_block_size=1,  # prevent implicit resizing that would destroy embedded bits
    )
    try:
        for f in frames_iter:
            writer.append_data(f)
    finally:
        writer.close()


def _video_traversal_indices_for_frames(frame_shapes: list, lsb: int, key_seed: int, frame_step: int):
    """Build a global index mapping over selected frames' flattened bytes.

    Returns (frame_indices, flat_indices_per_frame, total_slots) where:
    - frame_indices: list of frame numbers used (subset)
    - flat_indices_per_frame: list of numpy arrays of flat indices inside that frame (uint64)
    - total_slots: total number of available byte slots across selected frames
    """
    selected_frames = list(range(0, len(frame_shapes), max(1, int(frame_step))))
    flat_indices_per_frame = []
    total_slots = 0
    for _fi in selected_frames:
        h, w, c = frame_shapes[_fi]
        n = h * w * c
        idx = np.arange(n, dtype=np.uint64)
        flat_indices_per_frame.append(idx)
        total_slots += n

    # Build a global permutation across all slots for uniform scatter
    global_idx = traversal_indices(total_slots, key_seed)

    return selected_frames, flat_indices_per_frame, global_idx, total_slots


def _iter_video_frames(video_path: str):
    frames, meta = _load_video_frames_rgb(video_path)
    # Materialize frames into memory to allow random-like traversal; for large videos this could be heavy
    cached = []
    for f in frames:
        cached.append(f.copy())
    if meta.get("n_frames") is None:
        meta["n_frames"] = len(cached)
    return cached, meta


def do_embed_video(cover_path: str, payload_path: str, out_path: str, key: str, lsb: int, frame_step: int = 10):
    """Embed payload into a video by modifying LSBs of selected frames (every frame_step frames).

    The output is encoded losslessly (libx264rgb -crf 0) to preserve embedded bits.
    """
    if lsb < 1 or lsb > 8:
        raise ValueError("lsb must be 1..8 for video as well")

    frames, meta = _iter_video_frames(cover_path)
    if not frames:
        raise ValueError("No frames found in video")

    frame_shapes = [f.shape for f in frames]  # list of (H, W, 3)
    seed = seed_from_key(key)

    payload = open(payload_path, "rb").read()
    header = Header(MAGIC, VERSION, COV_VIDEO, lsb, len(payload), hashlib.sha256(payload).digest())
    header_bytes = header.pack()

    bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
    chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

    # Capacity
    selected_frames, per_frame_idx, global_perm, total_slots = _video_traversal_indices_for_frames(
        frame_shapes, lsb, seed, frame_step
    )
    if needed_slots > total_slots:
        need = (needed_slots * lsb + 7) // 8
        cap = (total_slots * lsb) // 8
        raise ValueError(f"Payload requires ~{need} bytes but capacity is {cap} bytes across selected frames.")

    # Map global permutation to (frame, local_index)
    mask = np.uint8(0xFF ^ ((1 << lsb) - 1))
    remaining = needed_slots
    cursor = 0
    # Precompute cumulative sizes
    per_frame_sizes = [idx.size for idx in per_frame_idx]
    cum_sizes = np.cumsum([0] + per_frame_sizes)

    # Apply writes
    for i, frame_number in enumerate(selected_frames):
        start_global = cum_sizes[i]
        end_global = cum_sizes[i + 1]
        # Among the global permutation, find positions that fall into this frame's slot range
        in_frame_mask = (global_perm < end_global) & (global_perm >= start_global)
        picks = global_perm[in_frame_mask] - start_global
        if picks.size == 0:
            continue
        apply_count = min(picks.size, remaining)
        if apply_count <= 0:
            break
        flat_indices = per_frame_idx[i][picks[:apply_count]].astype(np.int64)
        frame = frames[frame_number]
        flat = frame.reshape(-1)
        flat[flat_indices] = (flat[flat_indices] & mask) | chunks[cursor : cursor + apply_count].astype(np.uint8)
        frames[frame_number] = flat.reshape(frame.shape)
        cursor += apply_count
        remaining -= apply_count
        if remaining <= 0:
            break

    # Save stego video
    _save_video_frames_rgb(out_path, (frames[j] for j in range(len(frames))), fps=meta["fps"])


def do_extract_video(stego_path: str, out_payload_path: str, key: str, lsb: int, frame_step: int = 10):
    """Extract payload from a losslessly-encoded stego video produced by do_embed_video."""
    if lsb < 1 or lsb > 8:
        raise ValueError("lsb must be 1..8 for video as well")

    frames, meta = _iter_video_frames(stego_path)
    if not frames:
        raise ValueError("No frames found in video")

    frame_shapes = [f.shape for f in frames]
    seed = seed_from_key(key)

    selected_frames, per_frame_idx, global_perm, total_slots = _video_traversal_indices_for_frames(
        frame_shapes, lsb, seed, frame_step
    )

    # First read header
    hdr_bits_needed = HEADER_BYTES * 8
    slots_for_hdr = (hdr_bits_needed + lsb - 1) // lsb

    if slots_for_hdr > total_slots:
        raise ValueError("Not enough capacity to read header from selected frames")

    def _read_slots(n_slots: int) -> np.ndarray:
        vals = np.zeros(n_slots, dtype=np.uint16)
        taken = 0
        cursor = 0
        per_frame_sizes = [idx.size for idx in per_frame_idx]
        cum_sizes = np.cumsum([0] + per_frame_sizes)
        for i, frame_number in enumerate(selected_frames):
            if taken >= n_slots:
                break
            start_global = cum_sizes[i]
            end_global = cum_sizes[i + 1]
            in_frame_mask = (global_perm < end_global) & (global_perm >= start_global)
            picks = global_perm[in_frame_mask] - start_global
            if picks.size == 0:
                continue
            # Respect remaining
            apply_count = min(picks.size, n_slots - taken)
            flat_indices = per_frame_idx[i][picks[:apply_count]].astype(np.int64)
            frame = frames[frame_number]
            flat = frame.reshape(-1)
            vals[cursor : cursor + apply_count] = (flat[flat_indices] & ((1 << lsb) - 1)).astype(np.uint16)
            cursor += apply_count
            taken += apply_count
        return vals

    vals_hdr = _read_slots(slots_for_hdr)
    hdr_bits = unpack_stream_from_lsb(vals_hdr, hdr_bits_needed, lsb)
    hdr = Header.unpack(bits_to_bytes(hdr_bits))

    if hdr.cover_type != COV_VIDEO or hdr.lsb_count != lsb:
        raise ValueError("Wrong key/cover/lsb settings (header mismatch).")

    total_payload_bits = hdr.payload_len * 8
    slots_for_payload = (total_payload_bits + lsb - 1) // lsb

    vals_pl = _read_slots(slots_for_hdr + slots_for_payload)[slots_for_hdr:]
    pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
    payload = bits_to_bytes(pay_bits)

    if hashlib.sha256(payload).digest() != hdr.payload_sha256:
        raise ValueError("Integrity check failed (wrong key or corrupted stego).")

    open(out_payload_path, "wb").write(payload)
    print(f"Extracted {len(payload)} bytes from video -> {out_payload_path}")