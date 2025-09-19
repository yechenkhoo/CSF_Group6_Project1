import io
import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from main import (
    do_embed_image,
    do_extract_image,
    do_embed_audio,
    do_extract_audio,
    do_embed_image_region,
    do_extract_image_region,
)

SUPPORTED_IMAGE_EXTS = {".png", ".bmp"}
SUPPORTED_AUDIO_EXTS = {".wav"}

st.set_page_config(page_title="LSB Stego", layout="wide")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("LSB Steganography")

st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Encode (Embed)", "Decode (Extract)"])
lsb = st.sidebar.slider("LSBs to use", 1, 8, 3)
key = st.sidebar.text_input("Key (required)", value="")

# st.sidebar.caption("Tip: PNG/BMP for images (lossless), 16-bit PCM for WAV.")


def _save_to_tmp(uploaded_file, suffix: str) -> str:
    """Save an UploadedFile to a temporary path, return path."""
    # resets file pointer to beginning
    uploaded_file.seek(0)
    data = uploaded_file.read()
    # resets again for any subsequent reads
    uploaded_file.seek(0)

    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def get_region_selection(image, key="region"):
    """Allow user to specify a rectangular region on the image"""
    if hasattr(image, "size"):
        width, height = image.size
    else:
        height, width = image.shape[:2]

    use_region = st.checkbox(
        "Use specific region instead of whole image", key=f"{key}_use"
    )
    if use_region:
        st.subheader("Select Embedding Region")
        col1, col2 = st.columns(2)

        with col1:
            x = st.slider("Start X", 0, width - 1, 0, key=f"{key}_x")
            y = st.slider("Start Y", 0, height - 1, 0, key=f"{key}_y")

        with col2:
            w = st.slider("Width", 1, width - x, min(200, width - x), key=f"{key}_w")
            h = st.slider("Height", 1, height - y, min(200, height - y), key=f"{key}_h")

        st.info(f"Region: ({x},{y}) to ({x+w},{y+h}) - {w}×{h} pixels")
        return {"x": x, "y": y, "width": w, "height": h}

    return None


def show_region_preview(image, region):
    """Show the selected region highlighted on the image"""
    if region is None:
        return image

    from PIL import ImageDraw

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]

    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    return preview


def encode_ui():
    st.subheader("Encode: Hide a payload inside a cover")
    c1, c2 = st.columns(2)

    with c1:
        cover_up = st.file_uploader(
            "Cover file (.png/.bmp or .wav)", type=["png", "bmp", "wav"], key="cover"
        )
        payload_up = st.file_uploader("Payload file (any)", type=None, key="payload")
        out_name = st.text_input("Output stego filename", value="stego")
        ext_choice = st.selectbox("Output type", [".png", ".bmp", ".wav"], index=0)
        go = st.button("Embed", type="primary")

        if cover_up is not None:
            # Capacity check
            try:
                cov_ext = os.path.splitext(cover_up.name)[1].lower()
                # Use a copy of the file data to avoid moving file pointer
                cover_up.seek(0)
                cov_bytes = cover_up.read()
                cover_up.seek(0)  # Reset for later use

                if cov_ext in SUPPORTED_IMAGE_EXTS:
                    cap = _capacity_image_bytes(cov_bytes, lsb)
                    st.info(f"Cover capacity: {cap:,} bytes (lsb={lsb})")
                elif cov_ext in SUPPORTED_AUDIO_EXTS:
                    cap = _capacity_wav_bytes(cov_bytes, lsb)
                    st.info(f"Cover capacity: {cap:,} bytes (lsb={lsb})")
                else:
                    st.warning("Unsupported cover type")
            except Exception as e:
                st.warning(f"Capacity check failed: {e}")

    with c2:
        st.markdown("**Preview**")
        region = None
        if cover_up is not None and cover_up.type.startswith("image/"):
            # Create image from bytes to avoid file pointer issues
            cover_up.seek(0)
            pil_image = Image.open(io.BytesIO(cover_up.read()))
            cover_up.seek(0)  # Reset for later use
            st.image(pil_image, caption=f"Cover: {cover_up.name}", width="stretch")

            # Add region selection for images
            region = get_region_selection(pil_image, key="encode")

            # Show region preview if selected
            if region:
                preview_img = show_region_preview(pil_image, region)
                st.image(
                    preview_img,
                    caption="Selected region (red outline)",
                    width="stretch",
                )

                # Update capacity for region
                try:
                    from main import calculate_region_capacity

                    img_array = np.array(
                        pil_image.convert("RGBA" if pil_image.mode == "RGBA" else "RGB")
                    )
                    region_cap = calculate_region_capacity(img_array.shape, region, lsb)
                    st.info(f"Region capacity: {region_cap:,} bytes (lsb={lsb})")
                except Exception as e:
                    st.warning(f"Region capacity calculation failed: {e}")

        elif cover_up is not None and cover_up.type.startswith("audio/"):
            st.audio(cover_up)

    if go:
        if not key:
            st.error("Key is required")
        elif cover_up is None or payload_up is None:
            st.error("Please provide both cover and payload files")
        else:
            try:
                # Persist uploads to tmp paths
                cov_ext = os.path.splitext(cover_up.name)[1].lower()
                cover_path = _save_to_tmp(cover_up, suffix=cov_ext)
                payload_path = _save_to_tmp(
                    payload_up, suffix=os.path.splitext(payload_up.name)[1] or ".txt"
                )
                out_path = os.path.join(
                    tempfile.gettempdir(), (out_name or "stego") + ext_choice
                )

                if (
                    cov_ext in SUPPORTED_IMAGE_EXTS
                    and ext_choice in SUPPORTED_IMAGE_EXTS
                ):
                    do_embed_image_region(
                        cover_path, payload_path, out_path, key, lsb, region
                    )
                    region_info = (
                        f" (region {region['width']}×{region['height']})"
                        if region
                        else ""
                    )
                    st.success(f"Embedded into image stego{region_info}")
                    # Show stego and diff map
                    with open(cover_path, "rb") as f:
                        cov_img = Image.open(io.BytesIO(f.read()))
                    with open(out_path, "rb") as f:
                        stego_img = Image.open(io.BytesIO(f.read()))
                    cov_rgba = np.array(
                        Image.open(cover_path).convert("RGBA"), dtype=np.uint8
                    )
                    stego_rgba = np.array(
                        Image.open(out_path).convert("RGBA"), dtype=np.uint8
                    )
                    st.image(stego_img, caption="Stego image", width="stretch")

                    # Diff map on LSBs used
                    diff = ((stego_rgba[:, :, :3]) ^ (cov_rgba[:, :, :3])) & (
                        (1 << lsb) - 1
                    )

                    # amplify to visible 0..255
                    scale = 255 // ((1 << lsb) - 1)
                    diff_vis = (diff * scale).astype(np.uint8)
                    st.image(
                        diff_vis,
                        caption=f"Difference map of used LSBs (x{scale})",
                        width="stretch",
                    )
                    # Download
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "Download stego image",
                            f,
                            file_name=os.path.basename(out_path),
                        )

                elif cov_ext in SUPPORTED_AUDIO_EXTS and ext_choice == ".wav":
                    do_embed_audio(cover_path, payload_path, out_path, key, lsb)
                    st.success("Embedded into audio stego")
                    # Audio preview + LSB waveform viz
                    with open(out_path, "rb") as f:
                        stego_bytes = f.read()
                        st.audio(stego_bytes)
                    # Plot LSB-only waveform (first 50k samples for speed)
                    import wave

                    with wave.open(out_path, "rb") as wf:
                        n_ch = wf.getnchannels()
                        n_frames = wf.getnframes()
                        raw = wf.readframes(min(n_frames, 50000))
                    arr = np.frombuffer(raw, dtype=np.int16)
                    lsb_mask = (1 << lsb) - 1
                    lsb_wave = (arr.view(np.uint16) & lsb_mask).astype(np.uint16)
                    fig = plt.figure()
                    plt.plot(lsb_wave)
                    plt.title("LSB waveform (subset)")
                    st.pyplot(fig)
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "Download stego audio",
                            f,
                            file_name=os.path.basename(out_path),
                        )
                else:
                    st.error(
                        "Output type must match cover family (image→.png/.bmp, audio→.wav)"
                    )
            except Exception as e:
                st.error(f"Embed failed: {e}")


def decode_ui():
    st.subheader("Decode: Extract a payload from a stego file")
    c1, c2 = st.columns(2)

    with c1:
        stego_up = st.file_uploader(
            "Stego file (.png/.bmp or .wav)", type=["png", "bmp", "wav"], key="stego"
        )
        out_label = st.text_input("Name for extracted file", value="payload.txt")
        go2 = st.button("Extract", type="primary")

    with c2:
        st.markdown("**Preview**")
        decode_region = None
        if stego_up is not None and stego_up.type.startswith("image/"):
            stego_up.seek(0)
            pil_image = Image.open(io.BytesIO(stego_up.read()))
            stego_up.seek(0)  # Reset for later use
            st.image(pil_image, caption=f"Stego: {stego_up.name}", width="stretch")

            # put region selection here for images (must match encoding region)
            decode_region = get_region_selection(pil_image, key="decode")

            if decode_region:
                preview_img = show_region_preview(pil_image, decode_region)
                st.image(
                    preview_img,
                    caption="Selected region (red outline)",
                    width="stretch",
                )

        elif stego_up is not None and stego_up.type.startswith("audio/"):
            st.audio(stego_up)

    if go2:
        if not key:
            st.error("Key is required")
        elif stego_up is None:
            st.error("Please upload a stego file")
        else:
            try:
                stego_ext = os.path.splitext(stego_up.name)[1].lower()
                stego_path = _save_to_tmp(stego_up, suffix=stego_ext)
                out_path = os.path.join(
                    tempfile.gettempdir(), out_label or "payload.txt"
                )

                if stego_ext in SUPPORTED_IMAGE_EXTS:
                    do_extract_image_region(
                        stego_path, out_path, key, lsb, decode_region
                    )
                    region_info = (
                        f" (region {decode_region['width']}×{decode_region['height']})"
                        if decode_region
                        else ""
                    )
                    st.success(f"Extracted payload from image{region_info}")
                elif stego_ext in SUPPORTED_AUDIO_EXTS:
                    do_extract_audio(stego_path, out_path, key, lsb)
                    st.success("Extracted payload from audio")
                else:
                    st.error("Unsupported stego type")
                    st.stop()

                # Offer download and quick preview if small text or image
                with open(out_path, "rb") as f:
                    data = f.read()
                    st.download_button(
                        "Download extracted payload",
                        data,
                        file_name=os.path.basename(out_path),
                    )
                # Try preview
                try:
                    if out_label.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
                        st.image(data, caption="Extracted image preview")
                    elif (
                        out_label.lower().endswith((".txt", ".md", ".json", ".py"))
                        and len(data) < 200_000
                    ):
                        st.code(data.decode(errors="replace"), language="text")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Extract failed: {e}")


@st.cache_data
def _capacity_image_bytes(img_bytes: bytes, l: int) -> int:
    im = Image.open(io.BytesIO(img_bytes))
    # Keep alpha for PNG; RGB for BMP
    if im.format and im.format.lower() == "png":
        im = im.convert("RGBA")
    else:
        im = im.convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    cap_bits = arr.size * l
    return cap_bits // 8


@st.cache_data
def _capacity_wav_bytes(wav_bytes: bytes, l: int) -> int:
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        if sw != 2:
            raise ValueError("Only 16-bit PCM WAV supported")
        total_samples = n_frames * n_ch
    return (total_samples * l) // 8


# ---------- UI: Encode ----------
if mode == "Encode (Embed)":
    encode_ui()

# ---------- UI: Decode ----------
else:
    decode_ui()
