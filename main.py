import argparse, hashlib, struct, sys, wave, os, json, subprocess, tempfile
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
    idx = traversal_indices(buf.size, seed)

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
COV_VIDEO_STREAM = 3  # For specific video/audio stream embedding


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


def _get_video_stream_info(video_path: str):
    """Get information about available video and audio streams in the video file."""
    if imageio is None:
        raise RuntimeError("imageio is required for video stream analysis. Please install imageio[ffmpeg].")
    
    try:
        # Use ffprobe to get detailed stream information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            # Fallback if ffprobe fails
            return {
                'video': [{'index': 0, 'codec': 'unknown', 'width': 0, 'height': 0, 'fps': 30, 'bitrate': 'unknown'}],
                'audio': []
            }
        
        data = json.loads(result.stdout)
        
        streams = {
            'video': [],
            'audio': []
        }
        
        video_count = 0
        audio_count = 0
        
        for i, stream in enumerate(data.get('streams', [])):
            codec_type = stream.get('codec_type', '')
            codec_name = stream.get('codec_name', 'unknown')
            
            if codec_type == 'video':
                # Calculate FPS more safely
                try:
                    fps_str = stream.get('r_frame_rate', '30/1')
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den) if float(den) != 0 else 30.0
                    else:
                        fps = float(fps_str)
                except:
                    fps = 30.0
                
                streams['video'].append({
                    'index': video_count,
                    'actual_index': i,
                    'codec': codec_name,
                    'width': stream.get('width', 0),
                    'height': stream.get('height', 0),
                    'fps': fps,
                    'bitrate': stream.get('bit_rate', 'unknown')
                })
                video_count += 1
                
            elif codec_type == 'audio':
                streams['audio'].append({
                    'index': audio_count,
                    'actual_index': i,
                    'codec': codec_name,
                    'sample_rate': stream.get('sample_rate', 0),
                    'channels': stream.get('channels', 0),
                    'bitrate': stream.get('bit_rate', 'unknown')
                })
                audio_count += 1
        
        # Ensure we have at least one video stream entry for the UI
        if not streams['video']:
            streams['video'].append({
                'index': 0, 'actual_index': 0, 'codec': 'unknown', 
                'width': 0, 'height': 0, 'fps': 30, 'bitrate': 'unknown'
            })
        
        return streams
        
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        # Fallback to basic info if anything fails
        return {
            'video': [{'index': 0, 'actual_index': 0, 'codec': 'unknown', 'width': 0, 'height': 0, 'fps': 30, 'bitrate': 'unknown'}],
            'audio': []
        }


def _load_video_stream_data(video_path: str, stream_type: str, stream_index: int):
    """Load raw data from a specific video or audio stream."""
    if imageio is None:
        raise RuntimeError("imageio is required for stream processing. Please install imageio[ffmpeg].")
    
    try:
        # First, let's check what streams are actually available
        probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_result.stdout)
            streams = probe_data.get('streams', [])
            
            # Find the actual stream index based on type
            target_streams = [i for i, s in enumerate(streams) if s.get('codec_type') == stream_type]
            if not target_streams:
                raise RuntimeError(f"No {stream_type} streams found in video")
            if stream_index >= len(target_streams):
                raise RuntimeError(f"{stream_type.capitalize()} stream {stream_index} not found. Available: 0-{len(target_streams)-1}")
            
            # Use the actual stream index from the file
            actual_stream_idx = target_streams[stream_index]
            
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            # Fallback to simple stream index
            actual_stream_idx = stream_index
        
        if stream_type == 'video':
            # Extract raw video frames - use simpler mapping
            cmd = [
                'ffmpeg', '-v', 'error', '-i', video_path, 
                '-map', f'0:{actual_stream_idx}', '-f', 'rawvideo', 
                '-pix_fmt', 'rgb24', '-'
            ]
        elif stream_type == 'audio':
            # Extract raw audio data - use simpler mapping
            cmd = [
                'ffmpeg', '-v', 'error', '-i', video_path,
                '-map', f'0:{actual_stream_idx}', '-f', 's16le',
                '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-'
            ]
        else:
            raise ValueError(f"Unsupported stream type: {stream_type}")
        
        result = subprocess.run(cmd, capture_output=True, check=False)
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed to extract {stream_type} stream {stream_index}: {error_msg}")
        
        if len(result.stdout) == 0:
            raise RuntimeError(f"No data extracted from {stream_type} stream {stream_index}")
            
        return result.stdout
        
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg using:\n" +
                         "  macOS: brew install ffmpeg\n" +
                         "  Windows: choco install ffmpeg\n" +
                         "  Linux: sudo apt install ffmpeg\n" +
                         "Or use frame-based embedding instead of stream-based.")
    except Exception as e:
        raise RuntimeError(f"Failed to extract {stream_type} stream {stream_index} from {video_path}: {str(e)}")


def _save_video_with_stream_data(input_path: str, output_path: str, stream_type: str, 
                                stream_index: int, modified_data: bytes):
    """Save video with modified stream data back to file."""
    if imageio is None:
        raise RuntimeError("imageio is required for stream processing. Please install imageio[ffmpeg].")
    
    try:
        # For simplicity, let's use the existing frame-based approach but with the modified data
        # This is more reliable than trying to replace individual streams
        
        if stream_type == 'video':
            # For video streams, we'll reconstruct the entire video
            # First, get video properties
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', input_path]
            try:
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                probe_data = json.loads(probe_result.stdout)
                streams = probe_data.get('streams', [])
                video_streams = [s for s in streams if s.get('codec_type') == 'video']
                
                if video_streams:
                    width = video_streams[0].get('width', 640)
                    height = video_streams[0].get('height', 480)
                    fps = eval(video_streams[0].get('r_frame_rate', '30/1'))
                else:
                    width, height, fps = 640, 480, 30
            except:
                width, height, fps = 640, 480, 30
            
            # Save modified video data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.rgb') as temp_file:
                temp_file.write(modified_data)
                temp_data_path = temp_file.name
            
            try:
                # Create new video from raw RGB data
                cmd = [
                    'ffmpeg', '-v', 'error', '-y',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', 
                    '-s', f'{width}x{height}', '-r', str(fps),
                    '-i', temp_data_path,
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, check=False)
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
                    raise RuntimeError(f"FFmpeg failed to create video: {error_msg}")
                    
            finally:
                os.unlink(temp_data_path)
                
        elif stream_type == 'audio':
            # For audio streams, create a new video with modified audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.raw') as temp_file:
                temp_file.write(modified_data)
                temp_data_path = temp_file.name
            
            try:
                # Replace audio stream while keeping video
                cmd = [
                    'ffmpeg', '-v', 'error', '-y',
                    '-i', input_path,
                    '-f', 's16le', '-ar', '44100', '-ac', '2', '-i', temp_data_path,
                    '-map', '0:v', '-map', '1:a', 
                    '-c:v', 'copy', '-c:a', 'aac',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, check=False)
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
                    raise RuntimeError(f"FFmpeg failed to replace audio: {error_msg}")
                    
            finally:
                os.unlink(temp_data_path)
            
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg to use stream-based embedding.")
    except Exception as e:
        raise RuntimeError(f"Failed to save modified {stream_type} stream to {output_path}: {str(e)}")


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
        # Ensure picks are integers and within bounds
        picks_subset = picks[:apply_count].astype(np.int64)
        flat_indices = picks_subset  # per_frame_idx[i] is just np.arange(n), so picks_subset are the actual indices
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
    
    # Get file sizes for comparison
    original_size = os.path.getsize(cover_path)
    stego_size = os.path.getsize(out_path)
    size_change = stego_size - original_size
    size_change_pct = (size_change / original_size * 100) if original_size > 0 else 0
    
    print(f"Embedded {len(payload)} bytes into video (frame-based) -> {out_path}")
    print(f"Original file: {original_size:,} bytes")
    print(f"Stego file: {stego_size:,} bytes") 
    print(f"Size change: {size_change:+,} bytes ({size_change_pct:+.2f}%)")
    print(f"Capacity used: {needed_slots}/{total_slots} slots ({needed_slots/total_slots*100:.2f}%)")


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
            # Ensure picks are integers and within bounds
            picks_subset = picks[:apply_count].astype(np.int64)
            flat_indices = picks_subset  # per_frame_idx[i] is just np.arange(n), so picks_subset are the actual indices
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


def do_embed_video_stream(cover_path: str, payload_path: str, out_path: str, key: str, 
                         lsb: int, stream_type: str, stream_index: int = 0):
    """Embed payload into a specific video or audio stream within a video file.
    
    Args:
        cover_path: Input video file path
        payload_path: Payload file to embed
        out_path: Output video file path
        key: Encryption key
        lsb: Number of LSB bits to use
        stream_type: 'video' or 'audio'
        stream_index: Index of the stream (0-based)
    """
    if lsb < 1 or lsb > 8:
        raise ValueError("lsb must be 1..8")
    
    if stream_type not in ['video', 'audio']:
        raise ValueError("stream_type must be 'video' or 'audio'")

    # For video streams, try the simplified approach first since FFmpeg stream manipulation can be complex
    if stream_type == 'video':
        try:
            payload_size = _embed_in_video_frames_simple(cover_path, payload_path, out_path, key, lsb)
            print(f"Embedded {payload_size} bytes into video frames -> {out_path}")
            return
        except Exception as e:
            print(f"Simple video embedding failed: {e}")
            # Fall through to try the original method
    
    # Try the original FFmpeg-based method
    try:
        # Load raw stream data
        raw_data = _load_video_stream_data(cover_path, stream_type, stream_index)
        if len(raw_data) == 0:
            raise ValueError(f"No data found in {stream_type} stream {stream_index}")

        # Convert to numpy array for processing
        if stream_type == 'video':
            # RGB video data (uint8)
            data_array = np.frombuffer(raw_data, dtype=np.uint8)
        else:
            # Audio data (int16) - convert to uint16 for LSB operations
            int16_data = np.frombuffer(raw_data, dtype=np.int16)
            data_array = int16_data.view(np.uint16)

        seed = seed_from_key(key)
        idx = traversal_indices(data_array.size, seed)

        payload = open(payload_path, "rb").read()
        header = Header(MAGIC, VERSION, COV_VIDEO_STREAM, lsb, len(payload), hashlib.sha256(payload).digest())
        header_bytes = header.pack()

        # Build bitstream: header + payload
        bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
        chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

        # Check capacity
        if needed_slots > data_array.size:
            need = (needed_slots * lsb + 7) // 8
            cap = (data_array.size * lsb) // 8
            raise ValueError(f"Payload requires ~{need} bytes but {stream_type} stream capacity is {cap} bytes.")

        # Embed data into LSBs
        if stream_type == 'video':
            mask = np.uint8(0xFF ^ ((1 << lsb) - 1))
            target = data_array.copy()
            sel = idx[:needed_slots].astype(np.int64)
            target[sel] = (target[sel] & mask) | chunks.astype(np.uint8)
            modified_data = target.tobytes()
        else:
            mask = np.uint16(0xFFFF ^ ((1 << lsb) - 1))
            target = data_array.copy()
            sel = idx[:needed_slots].astype(np.int64)
            target[sel] = (target[sel] & mask) | chunks.astype(np.uint16)
            # Convert back to int16 for audio
            modified_data = target.view(np.int16).tobytes()

        # Save video with modified stream
        _save_video_with_stream_data(cover_path, out_path, stream_type, stream_index, modified_data)
        
        # Get file sizes for comparison
        original_size = os.path.getsize(cover_path)
        stego_size = os.path.getsize(out_path)
        size_change = stego_size - original_size
        size_change_pct = (size_change / original_size * 100) if original_size > 0 else 0
        
        print(f"Embedded {len(payload)} bytes into {stream_type} stream {stream_index} -> {out_path}")
        print(f"Original file: {original_size:,} bytes")
        print(f"Stego file: {stego_size:,} bytes") 
        print(f"Size change: {size_change:+,} bytes ({size_change_pct:+.2f}%)")
        print(f"Capacity used: {needed_slots}/{data_array.size} slots ({needed_slots/data_array.size*100:.2f}%)")
        
    except Exception as e:
        if stream_type == 'video':
            # If both methods fail for video, give a clear error message
            raise RuntimeError(f"Failed to embed into video stream: {str(e)}. Consider using frame-based embedding instead.")
        else:
            # For audio, re-raise the original error
            raise


def do_extract_video_stream(stego_path: str, out_payload_path: str, key: str, 
                           lsb: int, stream_type: str, stream_index: int = 0):
    """Extract payload from a specific video or audio stream within a video file.
    
    Args:
        stego_path: Input stego video file path
        out_payload_path: Output file for extracted payload
        key: Decryption key
        lsb: Number of LSB bits used
        stream_type: 'video' or 'audio'
        stream_index: Index of the stream (0-based)
    """
    if lsb < 1 or lsb > 8:
        raise ValueError("lsb must be 1..8")
    
    if stream_type not in ['video', 'audio']:
        raise ValueError("stream_type must be 'video' or 'audio'")

    # For video streams, try the simplified approach first
    if stream_type == 'video':
        try:
            payload = _extract_from_video_frames_simple(stego_path, key, lsb)
            open(out_payload_path, "wb").write(payload)
            print(f"Extracted {len(payload)} bytes from video frames -> {out_payload_path}")
            return
        except Exception as e:
            print(f"Simple video extraction failed: {e}")
            # Fall through to try the original method

    # Try the original FFmpeg-based method
    try:
        # Load raw stream data
        raw_data = _load_video_stream_data(stego_path, stream_type, stream_index)
        if len(raw_data) == 0:
            raise ValueError(f"No data found in {stream_type} stream {stream_index}")

        # Convert to numpy array for processing
        if stream_type == 'video':
            data_array = np.frombuffer(raw_data, dtype=np.uint8)
        else:
            int16_data = np.frombuffer(raw_data, dtype=np.int16)
            data_array = int16_data.view(np.uint16)

        seed = seed_from_key(key)
        idx = traversal_indices(data_array.size, seed)

        # First, read header bits
        hdr_bits_needed = HEADER_BYTES * 8
        slots_for_hdr = (hdr_bits_needed + lsb - 1) // lsb
        
        if slots_for_hdr > data_array.size:
            raise ValueError("Not enough data to read header from stream")

        sel_hdr = idx[:slots_for_hdr].astype(np.int64)
        
        if stream_type == 'video':
            vals_hdr = (data_array[sel_hdr] & ((1 << lsb) - 1)).astype(np.uint16)
        else:
            vals_hdr = (data_array[sel_hdr] & ((1 << lsb) - 1)).astype(np.uint16)
        
        hdr_bits = unpack_stream_from_lsb(vals_hdr, hdr_bits_needed, lsb)
        hdr = Header.unpack(bits_to_bytes(hdr_bits))

        if hdr.cover_type != COV_VIDEO_STREAM or hdr.lsb_count != lsb:
            raise ValueError("Wrong key/cover/lsb settings (header mismatch for stream).")

        total_payload_bits = hdr.payload_len * 8
        slots_for_payload = (total_payload_bits + lsb - 1) // lsb
        
        if slots_for_hdr + slots_for_payload > data_array.size:
            raise ValueError("Not enough data to read payload from stream")

        sel_pl = idx[slots_for_hdr : slots_for_hdr + slots_for_payload].astype(np.int64)
        
        if stream_type == 'video':
            vals_pl = (data_array[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
        else:
            vals_pl = (data_array[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
        
        pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
        payload = bits_to_bytes(pay_bits)

        if hashlib.sha256(payload).digest() != hdr.payload_sha256:
            raise ValueError("Integrity check failed (wrong key or corrupted stream stego).")

        open(out_payload_path, "wb").write(payload)
        print(f"Extracted {len(payload)} bytes from {stream_type} stream {stream_index} -> {out_payload_path}")
        
    except Exception as e:
        if stream_type == 'video':
            # If both methods fail for video, give a clear error message
            raise RuntimeError(f"Failed to extract from video stream: {str(e)}. The file might not contain stream-based embedded data or was encoded with frame-based embedding.")
        else:
            # For audio, re-raise the original error
            raise


def _embed_in_video_frames_simple(cover_path: str, payload_path: str, out_path: str, key: str, lsb: int):
    """Simplified video embedding that works directly with video frames using imageio."""
    if imageio is None:
        raise RuntimeError("imageio is required for video support. Please install imageio[ffmpeg].")
    
    # Load video frames
    frames, meta = _iter_video_frames(cover_path)
    if not frames:
        raise ValueError("No frames found in video")
    
    # Flatten all frame data into a single array for embedding
    all_frame_data = []
    for frame in frames:
        all_frame_data.append(frame.reshape(-1))
    
    combined_data = np.concatenate(all_frame_data)
    
    seed = seed_from_key(key)
    idx = traversal_indices(combined_data.size, seed)

    payload = open(payload_path, "rb").read()
    header = Header(MAGIC, VERSION, COV_VIDEO_STREAM, lsb, len(payload), hashlib.sha256(payload).digest())
    header_bytes = header.pack()

    # Build bitstream: header + payload
    bits = np.concatenate([bytes_to_bits(header_bytes), bytes_to_bits(payload)])
    chunks, needed_slots = pack_stream_for_lsb(bits, lsb)

    # Check capacity
    if needed_slots > combined_data.size:
        need = (needed_slots * lsb + 7) // 8
        cap = (combined_data.size * lsb) // 8
        raise ValueError(f"Payload requires ~{need} bytes but video capacity is {cap} bytes.")

    # Embed data into LSBs
    mask = np.uint8(0xFF ^ ((1 << lsb) - 1))
    target = combined_data.copy()
    sel = idx[:needed_slots].astype(np.int64)
    target[sel] = (target[sel] & mask) | chunks.astype(np.uint8)

    # Reconstruct frames
    current_pos = 0
    for i, frame in enumerate(frames):
        frame_size = frame.size
        frame_data = target[current_pos:current_pos + frame_size]
        frames[i] = frame_data.reshape(frame.shape)
        current_pos += frame_size

    # Save video
    _save_video_frames_rgb(out_path, (frames[j] for j in range(len(frames))), fps=meta["fps"])
    
    # Get file sizes for comparison
    original_size = os.path.getsize(cover_path)
    stego_size = os.path.getsize(out_path)
    size_change = stego_size - original_size
    size_change_pct = (size_change / original_size * 100) if original_size > 0 else 0
    
    print(f"Original file: {original_size:,} bytes")
    print(f"Stego file: {stego_size:,} bytes") 
    print(f"Size change: {size_change:+,} bytes ({size_change_pct:+.2f}%)")
    print(f"Capacity used: {needed_slots}/{combined_data.size} slots ({needed_slots/combined_data.size*100:.2f}%)")
    
    return len(payload)


def _extract_from_video_frames_simple(stego_path: str, key: str, lsb: int):
    """Simplified video extraction that works directly with video frames using imageio."""
    if imageio is None:
        raise RuntimeError("imageio is required for video support. Please install imageio[ffmpeg].")
    
    # Load video frames
    frames, meta = _iter_video_frames(stego_path)
    if not frames:
        raise ValueError("No frames found in video")
    
    # Flatten all frame data into a single array for extraction
    all_frame_data = []
    for frame in frames:
        all_frame_data.append(frame.reshape(-1))
    
    combined_data = np.concatenate(all_frame_data)
    
    seed = seed_from_key(key)
    idx = traversal_indices(combined_data.size, seed)

    # First, read header bits
    hdr_bits_needed = HEADER_BYTES * 8
    slots_for_hdr = (hdr_bits_needed + lsb - 1) // lsb
    
    if slots_for_hdr > combined_data.size:
        raise ValueError("Not enough data to read header from video")

    sel_hdr = idx[:slots_for_hdr].astype(np.int64)
    vals_hdr = (combined_data[sel_hdr] & ((1 << lsb) - 1)).astype(np.uint16)
    hdr_bits = unpack_stream_from_lsb(vals_hdr, hdr_bits_needed, lsb)
    hdr = Header.unpack(bits_to_bytes(hdr_bits))

    if hdr.cover_type != COV_VIDEO_STREAM or hdr.lsb_count != lsb:
        raise ValueError("Wrong key/cover/lsb settings (header mismatch for video stream).")

    total_payload_bits = hdr.payload_len * 8
    slots_for_payload = (total_payload_bits + lsb - 1) // lsb
    
    if slots_for_hdr + slots_for_payload > combined_data.size:
        raise ValueError("Not enough data to read payload from video")

    sel_pl = idx[slots_for_hdr : slots_for_hdr + slots_for_payload].astype(np.int64)
    vals_pl = (combined_data[sel_pl] & ((1 << lsb) - 1)).astype(np.uint16)
    
    pay_bits = unpack_stream_from_lsb(vals_pl, total_payload_bits, lsb)
    payload = bits_to_bytes(pay_bits)

    if hashlib.sha256(payload).digest() != hdr.payload_sha256:
        raise ValueError("Integrity check failed (wrong key or corrupted video stream stego).")

    return payload