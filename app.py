import io
import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import wave

from main import (
    do_embed_image,
    do_extract_image,
    do_embed_audio,
    do_extract_audio,
    do_embed_image_region,
    do_extract_image_region,
    do_embed_audio_region,
    do_extract_audio_region,
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


def get_audio_time_selection(audio_path, key="audio_time"):
    """Allow user to specify a time range in the audio file"""
    try:
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / sample_rate
    except:
        # Fallback if we can't read the file
        duration = 60.0
        sample_rate = 44100

    use_time_range = st.checkbox(
        "Use specific time range instead of whole audio", key=f"{key}_use"
    )
    
    if use_time_range:
        st.subheader("Select Audio Time Range")
        col1, col2 = st.columns(2)
        
        with col1:
            start_time = st.slider(
                "Start Time (seconds)", 
                0.0, 
                max(0.0, duration - 0.1), 
                0.0, 
                step=0.1, 
                key=f"{key}_start"
            )
            
        with col2:
            max_duration = duration - start_time
            time_duration = st.slider(
                "Duration (seconds)", 
                0.1, 
                max(0.1, max_duration), 
                min(10.0, max_duration), 
                step=0.1, 
                key=f"{key}_duration"
            )
        
        end_time = start_time + time_duration
        st.info(f"Time range: {start_time:.1f}s to {end_time:.1f}s ({time_duration:.1f}s duration)")
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration": time_duration,
            "sample_rate": sample_rate
        }
    
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


def show_audio_time_preview(audio_path, time_range):
    """Show waveform with selected time range highlighted"""
    if time_range is None:
        return None
        
    try:
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            
            # Read a subset of frames for visualization (max 50k samples)
            max_frames = min(50000, n_frames)
            raw = wf.readframes(max_frames)
            
        samples = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            samples = samples[::n_channels]  # Take only first channel for visualization
            
        # Create time axis
        time_axis = np.linspace(0, max_frames / sample_rate, len(samples))
        
        # Plot waveform
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(time_axis, samples, color='blue', alpha=0.7, linewidth=0.5)
        
        # Highlight selected region
        start_time = time_range["start_time"]
        end_time = time_range["end_time"]
        
        # Only highlight if the range is within our visualization window
        if start_time < time_axis[-1]:
            highlight_end = min(end_time, time_axis[-1])
            ax.axvspan(start_time, highlight_end, color='red', alpha=0.3, label='Selected Range')
            ax.axvline(x=start_time, color='red', linestyle='--', alpha=0.8)
            ax.axvline(x=highlight_end, color='red', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform with Selected Time Range')
        ax.grid(True, alpha=0.3)
        if start_time < time_axis[-1]:
            ax.legend()
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate audio preview: {e}")
        return None


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
        time_range = None
        
        if cover_up is not None and cover_up.type.startswith("image/"):
            # Create image from bytes to avoid file pointer issues
            cover_up.seek(0)
            pil_image = Image.open(io.BytesIO(cover_up.read()))
            cover_up.seek(0)  # Reset for later use
            st.image(pil_image, caption=f"Cover: {cover_up.name}", use_container_width=True)

            # Add region selection for images
            region = get_region_selection(pil_image, key="encode")

            # Show region preview if selected
            if region:
                preview_img = show_region_preview(pil_image, region)
                st.image(
                    preview_img,
                    caption="Selected region (red outline)",
                    use_container_width=True,
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
            
            # Save to temp file for time range selection
            cov_ext = os.path.splitext(cover_up.name)[1].lower()
            temp_audio_path = _save_to_tmp(cover_up, suffix=cov_ext)
            
            # Add time range selection for audio
            time_range = get_audio_time_selection(temp_audio_path, key="encode")
            
            # Show audio waveform preview with selected range
            if time_range:
                waveform_fig = show_audio_time_preview(temp_audio_path, time_range)
                if waveform_fig:
                    st.pyplot(waveform_fig)
                    plt.close(waveform_fig)
                
                # Calculate capacity for selected time range
                try:
                    from main import calculate_audio_time_capacity
                    time_cap = calculate_audio_time_capacity(temp_audio_path, time_range, lsb)
                    st.info(f"Time range capacity: {time_cap:,} bytes (lsb={lsb})")
                except Exception as e:
                    st.warning(f"Time range capacity calculation failed: {e}")

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
                    st.image(stego_img, caption="Stego image", use_container_width=True)

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
                        use_container_width=True,
                    )
                    # Download
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "Download stego image",
                            f,
                            file_name=os.path.basename(out_path),
                        )

                elif cov_ext in SUPPORTED_AUDIO_EXTS and ext_choice == ".wav":
                    if time_range:
                        do_embed_audio_region(cover_path, payload_path, out_path, key, lsb, time_range)
                        time_info = f" (time {time_range['start_time']:.1f}s-{time_range['end_time']:.1f}s)"
                        st.success(f"Embedded into audio stego{time_info}")
                    else:
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
                    plt.close(fig)
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
        decode_time_range = None
        
        if stego_up is not None and stego_up.type.startswith("image/"):
            stego_up.seek(0)
            pil_image = Image.open(io.BytesIO(stego_up.read()))
            stego_up.seek(0)  # Reset for later use
            st.image(pil_image, caption=f"Stego: {stego_up.name}", use_container_width=True)

            # put region selection here for images (must match encoding region)
            decode_region = get_region_selection(pil_image, key="decode")

            if decode_region:
                preview_img = show_region_preview(pil_image, decode_region)
                st.image(
                    preview_img,
                    caption="Selected region (red outline)",
                    use_container_width=True,
                )

        elif stego_up is not None and stego_up.type.startswith("audio/"):
            st.audio(stego_up)
            
            # Save to temp file for time range selection
            stego_ext = os.path.splitext(stego_up.name)[1].lower()
            temp_stego_path = _save_to_tmp(stego_up, suffix=stego_ext)
            
            # Add time range selection for audio (must match encoding time range)
            decode_time_range = get_audio_time_selection(temp_stego_path, key="decode")
            
            # Show audio waveform preview with selected range
            if decode_time_range:
                waveform_fig = show_audio_time_preview(temp_stego_path, decode_time_range)
                if waveform_fig:
                    st.pyplot(waveform_fig)
                    plt.close(waveform_fig)

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
                    if decode_time_range:
                        do_extract_audio_region(stego_path, out_path, key, lsb, decode_time_range)
                        time_info = f" (time {decode_time_range['start_time']:.1f}s-{decode_time_range['end_time']:.1f}s)"
                        st.success(f"Extracted payload from audio{time_info}")
                    else:
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