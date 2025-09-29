import io
import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import wave
from collections import Counter

from main import (
    do_embed_image,
    do_extract_image,
    do_embed_audio,
    do_extract_audio,
    do_embed_image_region,
    do_extract_image_region,
    do_embed_audio_region,
    do_extract_audio_region,
    do_embed_video_stream,
    do_extract_video_stream,
    _get_video_stream_info,
    _load_video_stream_data,
    do_embed_video,
    do_extract_video,
    do_embed_video_iframe,
    do_extract_video_iframe,
    _iter_video_frames,
)

SUPPORTED_IMAGE_EXTS = {".png", ".bmp"}
SUPPORTED_AUDIO_EXTS = {".wav"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv"}

st.set_page_config(page_title="LSB Stego", layout="wide")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {visibility: hidden;}
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


def show_video_frame_comparison(original_path, stego_path, frame_indices=None, lsb=2):
    """Show before/after comparison of video frames with difference highlighting"""
    try:
        # Load frames from both videos
        orig_frames, orig_meta = _iter_video_frames(original_path)
        stego_frames, stego_meta = _iter_video_frames(stego_path)
        
        if not orig_frames or not stego_frames:
            return None
            
        # Select frames to compare (default: first, middle, last few frames)
        n_frames = min(len(orig_frames), len(stego_frames))
        if frame_indices is None:
            if n_frames >= 10:
                frame_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
            else:
                frame_indices = [0, n_frames//2, n_frames-1] if n_frames >= 3 else [0]
        
        # Limit number of frames for display
        frame_indices = frame_indices[:4]  # Max 4 frames
        
        # Create comparison figure
        fig, axes = plt.subplots(3, len(frame_indices), figsize=(4*len(frame_indices), 10))
        if len(frame_indices) == 1:
            axes = axes.reshape(3, 1)
            
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx >= n_frames:
                continue
                
            orig_frame = orig_frames[frame_idx]
            stego_frame = stego_frames[frame_idx]
            
            # Original frame
            axes[0, i].imshow(orig_frame)
            axes[0, i].set_title(f'Original Frame {frame_idx}', fontsize=10)
            axes[0, i].axis('off')
            
            # Stego frame
            axes[1, i].imshow(stego_frame)
            axes[1, i].set_title(f'Stego Frame {frame_idx}', fontsize=10)
            axes[1, i].axis('off')
            
            # Difference (LSB changes amplified)
            diff = np.abs(stego_frame.astype(np.float32) - orig_frame.astype(np.float32))
            
            # Amplify LSB differences for visibility
            lsb_mask = (1 << lsb) - 1
            scale_factor = max(1, 255 // max(1, lsb_mask))  # Avoid division by zero
            diff_masked = diff * scale_factor
            diff_masked = np.clip(diff_masked, 0, 255).astype(np.uint8)
            
            axes[2, i].imshow(diff_masked, cmap='hot')
            axes[2, i].set_title(f'Differences (×{scale_factor}) Frame {frame_idx}', fontsize=10)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Could not generate video comparison: {e}")
        return None


def show_video_stream_analysis(video_path, stream_type, stream_index, lsb=2):
    """Show analysis of video stream data for embedding visualization"""
    try:
        # Get stream info
        streams = _get_video_stream_info(video_path)
        
        # Load raw stream data (first portion for visualization)
        raw_data = _load_video_stream_data(video_path, stream_type, stream_index)
        if len(raw_data) == 0:
            return None
            
        # Limit data size for visualization
        max_samples = 50000
        if len(raw_data) > max_samples:
            if stream_type == 'video':
                # For video, take samples from beginning
                viz_data = raw_data[:max_samples]
            else:
                # For audio, take evenly spaced samples
                step = len(raw_data) // max_samples
                viz_data = raw_data[::step][:max_samples]
        else:
            viz_data = raw_data
            
        if stream_type == 'video':
            # Convert to uint8 array for video data
            data_array = np.frombuffer(viz_data, dtype=np.uint8)
            title = f"Video Stream {stream_index} RGB Data"
            ylabel = "RGB Value (0-255)"
        else:
            # Convert to int16 array for audio data  
            data_array = np.frombuffer(viz_data, dtype=np.int16)
            title = f"Audio Stream {stream_index} Samples"
            ylabel = "Amplitude"
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        # Plot 1: Data distribution (first 5000 samples)
        sample_count = min(len(data_array), 5000)
        sample_indices = np.arange(sample_count)
        ax1.plot(sample_indices, data_array[:sample_count], alpha=0.7, linewidth=0.5)
        ax1.set_title(f"{title} (First {sample_count:,} samples)")
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: LSB distribution analysis
        if stream_type == 'video':
            lsb_values = data_array & ((1 << lsb) - 1)
        else:
            lsb_values = data_array.view(np.uint16) & ((1 << lsb) - 1)
            
        bins = min(32, (1 << lsb))
        ax2.hist(lsb_values, bins=bins, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_title(f"LSB Distribution Analysis (using {lsb} LSBs)")
        ax2.set_xlabel(f"LSB Value (0-{(1 << lsb) - 1})")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)
        
        # Add stream information as subtitle
        if stream_type == 'video' and streams.get('video'):
            stream_info = streams['video'][stream_index] if stream_index < len(streams['video']) else None
            if stream_info:
                info_text = f"Codec: {stream_info['codec']} | Resolution: {stream_info['width']}×{stream_info['height']} | FPS: {stream_info['fps']:.1f}"
                fig.suptitle(f"Video Stream Analysis\n{info_text}", fontsize=12)
        elif stream_type == 'audio' and streams.get('audio'):
            stream_info = streams['audio'][stream_index] if stream_index < len(streams['audio']) else None
            if stream_info:
                info_text = f"Codec: {stream_info['codec']} | Sample Rate: {stream_info['sample_rate']}Hz | Channels: {stream_info['channels']}"
                fig.suptitle(f"Audio Stream Analysis\n{info_text}", fontsize=12)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Could not generate stream analysis: {e}")
        return None
    
def plot_image_lsb_distribution(orig_path, stego_path, lsb=1):
    orig = np.array(Image.open(orig_path).convert("RGB"), dtype=np.uint8)
    stego = np.array(Image.open(stego_path).convert("RGB"), dtype=np.uint8)
    mask = (1 << lsb) - 1
    orig_lsb = orig & mask
    stego_lsb = stego & mask

    orig_flat = orig_lsb.ravel()
    stego_flat = stego_lsb.ravel()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axs[0].hist(orig_flat, bins=mask+1, range=(0, mask), color='blue', alpha=0.7)
    axs[0].set_title("Original Image LSB Histogram")
    axs[0].set_xlabel("LSB Value")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(stego_flat, bins=mask+1, range=(0, mask), color='orange', alpha=0.7)
    axs[1].set_title("Stego Image LSB Histogram")
    axs[1].set_xlabel("LSB Value")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_audio_lsb_distribution(orig_path, stego_path, lsb=1):
    def read_wave(path):
        with wave.open(path, "rb") as wf:
            return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    arr_orig = read_wave(orig_path)
    arr_stego = read_wave(stego_path)

    mask = (1 << lsb) - 1
    orig_lsb = arr_orig & mask
    stego_lsb = arr_stego & mask

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axs[0].hist(orig_lsb, bins=mask+1, range=(0, mask), color='blue', alpha=0.7)
    axs[0].set_title("Original Audio LSB Histogram")
    axs[0].set_xlabel("LSB Value")
    axs[0].set_ylabel("Frequency")

    axs[1].hist(stego_lsb, bins=mask+1, range=(0, mask), color='orange', alpha=0.7)
    axs[1].set_title("Stego Audio LSB Histogram")
    axs[1].set_xlabel("LSB Value")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def encode_ui():
    st.subheader("Encode: Hide a payload inside a cover")
    c1, c2 = st.columns(2)

    with c1:
        cover_up = st.file_uploader(
            "Cover file (.png/.bmp/.mp4/.mov/.mkv or .wav)", type=["png", "bmp", "wav", "mp4", "mov", "mkv"], key="cover"
        )
        # payload_up = st.file_uploader("Payload file (any)", type=None, key="payload")

        payload_mode = st.radio("Payload type", ["Text", "File"], horizontal=True)

        if payload_mode == "Text":
            payload_text = st.text_area("Enter text to hide", height=150, key="payload_text")
            payload_up = None
        else:
            payload_text = None
            payload_up = st.file_uploader("Payload file (any)", type=None, key="payload")


        out_name = st.text_input("Output stego filename", value="stego")
        ext_choice = st.selectbox("Output type", [".png", ".bmp", ".wav", ".mp4"], index=0)
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
                elif cov_ext in SUPPORTED_VIDEO_EXTS:
                    st.info("Video capacity not estimated here. Use default settings or separate video UI.")
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
        elif cover_up is not None and (cover_up.type.startswith("video/") or os.path.splitext(cover_up.name)[1].lower() in SUPPORTED_VIDEO_EXTS):
            st.video(cover_up)
            
            # Video embedding method selection
            video_method = st.radio(
                "Video embedding method:",
                ["Frame-based", "Stream-based"],
                key="video_method"
            )
            
            if video_method == "Frame-based":
                frame_step = st.number_input(
                    "Every Nth frame for video (must match on decode)",
                    min_value=1,
                    max_value=1000,
                    value=10,
                    step=1,
                    key="vid_step_preview",
                )
                st.caption("Choose .mp4 as output. Keep frame step consistent for decoding.")
            else:
                # Stream-based embedding options
                st.subheader("Stream Selection")
                
                # Get video stream info
                try:
                    cov_ext = os.path.splitext(cover_up.name)[1].lower()
                    temp_video_path = _save_to_tmp(cover_up, suffix=cov_ext)
                    
                    from main import _get_video_stream_info
                    stream_info = _get_video_stream_info(temp_video_path)
                    
                    # Display available streams
                    if stream_info['video']:
                        st.write("**Available Video Streams:**")
                        for i, stream in enumerate(stream_info['video']):
                            st.write(f"Stream {i}: {stream['codec']} ({stream['width']}x{stream['height']}, {stream['fps']:.1f} fps)")
                    
                    if stream_info['audio']:
                        st.write("**Available Audio Streams:**")
                        for i, stream in enumerate(stream_info['audio']):
                            st.write(f"Stream {i}: {stream['codec']} ({stream['channels']} channels, {stream['sample_rate']} Hz)")
                    
                    # Stream selection
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        stream_type = st.selectbox(
                            "Stream type:",
                            ["video", "audio"],
                            key="stream_type_encode"
                        )
                    
                    with col2:
                        if stream_type == "video" and stream_info['video']:
                            max_video_idx = len(stream_info['video']) - 1
                            stream_index = st.number_input(
                                "Video stream index:",
                                min_value=0,
                                max_value=max_video_idx,
                                value=0,
                                key="video_stream_idx_encode"
                            )
                        elif stream_type == "audio" and stream_info['audio']:
                            max_audio_idx = len(stream_info['audio']) - 1
                            stream_index = st.number_input(
                                "Audio stream index:",
                                min_value=0,
                                max_value=max_audio_idx,
                                value=0,
                                key="audio_stream_idx_encode"
                            )
                        else:
                            stream_index = 0
                            st.warning(f"No {stream_type} streams found")
                    
                    st.caption("Stream-based embedding hides payload within raw stream data.")
                    
                except Exception as e:
                    st.warning(f"Could not analyze video streams: {e}")
                    # Fallback defaults
                    stream_type = "video"
                    stream_index = 0

    if go:
        if not key:
            st.error("Key is required")
        elif cover_up is None:
            st.error("Please provide a cover file")
        elif payload_mode == "File" and payload_up is None:
            st.error("Please provide a payload file")
        elif payload_mode == "Text" and not payload_text.strip():
            st.error("Please enter some text to hide")
        else:
            try:
                # Persist uploads to tmp paths
                cov_ext = os.path.splitext(cover_up.name)[1].lower()
                cover_path = _save_to_tmp(cover_up, suffix=cov_ext)
                
                # payload_path = _save_to_tmp(
                #     payload_up, suffix=os.path.splitext(payload_up.name)[1] or ".txt"
                # )

                if payload_mode == "Text":
                    payload_fd, payload_path = tempfile.mkstemp(suffix=".txt")
                    with os.fdopen(payload_fd, "w", encoding="utf-8") as f:
                        f.write(payload_text or "")
                else:
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

                    # Histogram of pixel differences
                    try:
                        plot_image_lsb_distribution(cover_path, out_path, lsb=lsb)
                    except Exception as e:
                        st.warning(f"Could not generate difference histogram: {e}")

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

                    # Histogram of audio differences
                    try:
                        plot_audio_lsb_distribution(cover_path, out_path, lsb=lsb)
                    except Exception as e:
                        st.warning(f"Could not generate audio difference histogram: {e}")

                elif cov_ext in SUPPORTED_VIDEO_EXTS and ext_choice == ".mp4":
                    # Check which video method was selected
                    selected_method = st.session_state.get("video_method", "Frame-based")
                    
                    if selected_method == "Frame-based":
                        # NEW IFRAME-ONLY embedding - completely separate from stream-based

                        frame_step_val = int(st.session_state.get("vid_step_preview", 10))
                        
                        # Load original frames for comparison
                        try:
                            original_frames, orig_meta = _iter_video_frames(cover_path)
                        except Exception as e:
                            st.error(f"Failed to load original video frames: {e}")
                            st.stop()
                        
                        # Call the NEW iframe-only function
                        do_embed_video_iframe(cover_path, payload_path, out_path, key, lsb, frame_step_val)
                        
                        # Get file sizes for display
                        original_size = os.path.getsize(cover_path)
                        stego_size = os.path.getsize(out_path)
                        payload_size = os.path.getsize(payload_path)
                        size_change = stego_size - original_size
                        size_change_pct = (size_change / original_size * 100) if original_size > 0 else 0
                        
                        st.success("Embedded into video stego (IFRAME-ONLY)")
                        
                        # Show file size comparison
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original File", f"{original_size:,} bytes")
                        with col2:
                            st.metric("Stego File", f"{stego_size:,} bytes", f"{size_change:+,} bytes")
                        with col3:
                            st.metric("Payload Size", f"{payload_size:,} bytes")
                        
                        if size_change_pct != 0:
                            if abs(size_change_pct) < 0.01:
                                st.success(f"IFRAME Size change: {size_change:+,} bytes (<0.01%)")
                            else:
                                st.success(f"IFRAME Size change: {size_change:+,} bytes ({size_change_pct:+.2f}%)")
                        else:
                            st.success("IFRAME No size change detected")
                                                
                        # Load stego frames for comparison
                        try:
                            stego_frames, stego_meta = _iter_video_frames(out_path)
                        except Exception as e:
                            st.warning(f"Could not load stego frames for visualization: {e}")
                            stego_frames = None
                        
                        # Show video diff visualization
                        if stego_frames is not None:
                            st.subheader("Video Steganography Analysis")
                            
                            # Calculate which I-frames were modified
                            total_frames = len(original_frames)
                            gop_size = max(frame_step_val, 10)
                            iframe_candidates = list(range(0, total_frames, gop_size))
                            scatter_frames = [i for i in range(1, total_frames, max(1, total_frames // 20)) 
                                            if i not in iframe_candidates][:len(iframe_candidates)//2]
                            selected_frames = sorted(iframe_candidates + scatter_frames)
                            
                            st.info(f"I-frame pattern: GOP size {gop_size}, specialized keyframe embedding")
                            
                            # Show sample I-frames comparison
                            st.subheader("I-Frame Comparison (IFRAME-ONLY Method)")
                            num_samples = min(3, len(selected_frames))
                            sample_indices = selected_frames[:num_samples] if len(selected_frames) >= num_samples else selected_frames
                            
                            for i, frame_idx in enumerate(sample_indices):
                                iframe_type = i % 3
                                iframe_names = ["DCT Block", "Frequency Domain", "Keyframe Optimization"]
                                st.write(f"**I-Frame {frame_idx} ({iframe_names[iframe_type]} Pattern):**")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.image(original_frames[frame_idx], caption=f"Original I-Frame {frame_idx}", use_container_width=True)
                                
                                with col2:
                                    st.image(stego_frames[frame_idx], caption=f"Stego I-Frame {frame_idx}", use_container_width=True)
                                
                                with col3:
                                    # Calculate I-frame specific LSB difference
                                    orig_frame = original_frames[frame_idx]
                                    stego_frame = stego_frames[frame_idx]
                                    
                                    # Compute difference in LSBs (with enhanced LSB for some patterns)
                                    if iframe_type in [0, 2]:  # DCT and keyframe use enhanced LSB
                                        enhanced_lsb = min(lsb + 1, 6)
                                        diff = (stego_frame ^ orig_frame) & ((1 << enhanced_lsb) - 1)
                                        scale = 255 // ((1 << enhanced_lsb) - 1) if enhanced_lsb < 8 else 1
                                    else:
                                        diff = (stego_frame ^ orig_frame) & ((1 << lsb) - 1)
                                        scale = 255 // ((1 << lsb) - 1) if lsb < 8 else 1
                                    
                                    diff_vis = (diff * scale).astype(np.uint8)
                                    
                                    st.image(diff_vis, caption=f"I-Frame LSB Diff (x{scale})", use_container_width=True)
                            
                            # Show modified I-frames grid
                            if len(selected_frames) > 3:
                                st.subheader("All Modified I-Frames (Thumbnails)")
                                cols_per_row = 6
                                rows = (len(selected_frames) + cols_per_row - 1) // cols_per_row
                                
                                for row in range(min(3, rows)):  # Show max 3 rows
                                    cols = st.columns(cols_per_row)
                                    for col_idx in range(cols_per_row):
                                        frame_idx_in_list = row * cols_per_row + col_idx
                                        if frame_idx_in_list < len(selected_frames):
                                            frame_num = selected_frames[frame_idx_in_list]
                                            iframe_type = frame_idx_in_list % 3
                                            with cols[col_idx]:
                                                # Show I-frame specific difference thumbnail
                                                orig_frame = original_frames[frame_num]
                                                stego_frame = stego_frames[frame_num]
                                                
                                                # Use appropriate LSB for I-frame pattern
                                                if iframe_type in [0, 2]:
                                                    enhanced_lsb = min(lsb + 1, 6)
                                                    diff = (stego_frame ^ orig_frame) & ((1 << enhanced_lsb) - 1)
                                                    scale = 255 // ((1 << enhanced_lsb) - 1) if enhanced_lsb < 8 else 1
                                                else:
                                                    diff = (stego_frame ^ orig_frame) & ((1 << lsb) - 1)
                                                    scale = 255 // ((1 << lsb) - 1) if lsb < 8 else 1
                                                
                                                diff_vis = (diff * scale).astype(np.uint8)
                                                pattern_emoji = ["🟦", "🟨", "🟩"][iframe_type]
                                                st.image(diff_vis, caption=f"{pattern_emoji}I{frame_num}", use_container_width=True)
                                
                                if len(selected_frames) > rows * cols_per_row:
                                    st.caption(f"... and {len(selected_frames) - rows * cols_per_row} more I-frames with specialized patterns")
                            
                        else:
                            st.info("IFRAME-ONLY video embedded successfully, but frame comparison visualization is not available.")
                    
                    else:
                        # New stream-based embedding
                        from main import do_embed_video_stream
                        
                        # Get stream parameters
                        embed_stream_type = st.session_state.get("stream_type_encode", "video")
                        if embed_stream_type == "video":
                            embed_stream_index = int(st.session_state.get("video_stream_idx_encode", 0))
                        else:
                            embed_stream_index = int(st.session_state.get("audio_stream_idx_encode", 0))
                        
                        try:
                            do_embed_video_stream(cover_path, payload_path, out_path, key, lsb)
                        except RuntimeError as e:
                            if "FFmpeg not found" in str(e):
                                st.warning("FFmpeg not found. Falling back to frame-based embedding...")
                                st.info("To use stream-based embedding, please install FFmpeg:\n" +
                                       "- macOS: `brew install ffmpeg`\n" + 
                                       "- Windows: `choco install ffmpeg`\n" +
                                       "- Linux: `sudo apt install ffmpeg`")
                                # Fall back to frame-based embedding
                                do_embed_video(cover_path, payload_path, out_path, key, lsb, 10)
                                embed_stream_type = "frames"
                                embed_stream_index = "N/A"
                            else:
                                raise e
                        
                        # Get file sizes for display
                        original_size = os.path.getsize(cover_path)
                        stego_size = os.path.getsize(out_path)
                        payload_size = os.path.getsize(payload_path)
                        size_change = stego_size - original_size
                        size_change_pct = (size_change / original_size * 100) if original_size > 0 else 0
                        
                        st.success(f"Embedded into video stego ({embed_stream_type} stream {embed_stream_index})")
                        
                        st.subheader("Stream-based Steganography Analysis")
                        st.info(f"Payload hidden in {embed_stream_type} stream {embed_stream_index} using {lsb} LSB(s)")
                        
                        # Show file size comparison
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original File", f"{original_size:,} bytes")
                        with col2:
                            st.metric("Stego File", f"{stego_size:,} bytes", f"{size_change:+,} bytes")
                        with col3:
                            st.metric("Payload Size", f"{payload_size:,} bytes")
                        
                        if size_change_pct != 0:
                            if abs(size_change_pct) < 0.01:
                                st.success(f"Size change: {size_change:+,} bytes (<0.01%)")
                            else:
                                st.success(f"Size change: {size_change:+,} bytes ({size_change_pct:+.2f}%)")
                        else:
                            st.success("No size change detected")
                        
                        st.write("Stream-based embedding modifies the raw stream data directly, making it more robust against certain types of analysis.")
                        
                        # Show stream information if available
                        try:
                            streams = _get_video_stream_info(cover_path)
                            if embed_stream_type == "video" and streams.get("video"):
                                stream_info = streams["video"][embed_stream_index] if embed_stream_index < len(streams["video"]) else None
                                if stream_info:
                                    st.write(f"**Stream Details:** {stream_info['codec']} codec, {stream_info['width']}×{stream_info['height']} @ {stream_info['fps']:.1f}fps")
                            elif embed_stream_type == "audio" and streams.get("audio"):
                                stream_info = streams["audio"][embed_stream_index] if embed_stream_index < len(streams["audio"]) else None
                                if stream_info:
                                    st.write(f"**Stream Details:** {stream_info['codec']} codec, {stream_info['sample_rate']}Hz, {stream_info['channels']} channels")
                        except Exception:
                            pass  # If stream info fails, continue without it
                        
                        # Visual Analysis Section
                        st.subheader("Visual Analysis")
                        
                        # Stream analysis visualization
                        with st.expander("Stream Data Analysis", expanded=True):
                            try:
                                stream_fig = show_video_stream_analysis(cover_path, embed_stream_type, embed_stream_index, lsb)
                                if stream_fig:
                                    st.pyplot(stream_fig)
                                    plt.close(stream_fig)
                            except Exception as e:
                                st.warning(f"Could not generate stream analysis: {e}")
                        
                     
                        # Audio waveform comparison for audio streams
                        if embed_stream_type == "audio":
                            with st.expander("Audio Waveform Analysis", expanded=False):
                                try:
                                    # Show original audio preview
                                    st.write("**Original Audio:**")
                                    with open(cover_path, "rb") as f:
                                        st.audio(f.read())
                                    
                                    # Show stego audio preview
                                    st.write("**Stego Audio:**")
                                    with open(out_path, "rb") as f:
                                        st.audio(f.read())
                                    
                                    st.caption("Listen for any audible differences (there should be none with proper LSB embedding)")
                                except Exception as e:
                                    st.warning(f"Could not generate audio comparison: {e}")
                    
                    # Common download section for both methods
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "Download stego video",
                            f,
                            file_name=os.path.basename(out_path),
                        )
                else:
                    st.error(
                        "Output type must match cover family (image→.png/.bmp, audio→.wav, video→.mp4)"
                    )
            except Exception as e:
                st.error(f"Embed failed: {e}")


def decode_ui():
    st.subheader("Decode: Extract a payload from a stego file")
    c1, c2 = st.columns(2)

    with c1:
        stego_up = st.file_uploader(
            "Stego file (.png/.bmp/.mp4/.mov/.mkv or .wav)", type=["png", "bmp", "wav", "mp4", "mov", "mkv"], key="stego"
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
        elif stego_up is not None and (stego_up.type.startswith("video/") or os.path.splitext(stego_up.name)[1].lower() in SUPPORTED_VIDEO_EXTS):
            st.video(stego_up)
            
            # Video decoding method selection  
            decode_method = st.radio(
                "Video decoding method (must match encoding):",
                ["Frame-based", "Stream-based"],
                key="video_decode_method"
            )
            
            if decode_method == "Frame-based":
                frame_step = st.number_input(
                    "Every Nth frame for video (must match encode)",
                    min_value=1,
                    max_value=1000,
                    value=10,
                    step=1,
                    key="vid_step_preview_decode",
                )
            else:
                # Stream-based decoding options
                st.subheader("Stream Selection (must match encoding)")
                
                # Get video stream info
                try:
                    stego_ext = os.path.splitext(stego_up.name)[1].lower()
                    temp_stego_path = _save_to_tmp(stego_up, suffix=stego_ext)
                    
                    from main import _get_video_stream_info
                    stream_info = _get_video_stream_info(temp_stego_path)
                    
                    # Display available streams
                    if stream_info['video']:
                        st.write("**Available Video Streams:**")
                        for i, stream in enumerate(stream_info['video']):
                            st.write(f"Stream {i}: {stream['codec']} ({stream['width']}x{stream['height']}, {stream['fps']:.1f} fps)")
                    
                    if stream_info['audio']:
                        st.write("**Available Audio Streams:**")
                        for i, stream in enumerate(stream_info['audio']):
                            st.write(f"Stream {i}: {stream['codec']} ({stream['channels']} channels, {stream['sample_rate']} Hz)")
                    
                    # Stream selection
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        decode_stream_type = st.selectbox(
                            "Stream type:",
                            ["video", "audio"],
                            key="stream_type_decode"
                        )
                    
                    with col2:
                        if decode_stream_type == "video" and stream_info['video']:
                            max_video_idx = len(stream_info['video']) - 1
                            decode_stream_index = st.number_input(
                                "Video stream index:",
                                min_value=0,
                                max_value=max_video_idx,
                                value=0,
                                key="video_stream_idx_decode"
                            )
                        elif decode_stream_type == "audio" and stream_info['audio']:
                            max_audio_idx = len(stream_info['audio']) - 1
                            decode_stream_index = st.number_input(
                                "Audio stream index:",
                                min_value=0,
                                max_value=max_audio_idx,
                                value=0,
                                key="audio_stream_idx_decode"
                            )
                        else:
                            decode_stream_index = 0
                            st.warning(f"No {decode_stream_type} streams found")
                    
                    st.caption("Stream selection must match what was used during encoding.")
                    
                except Exception as e:
                    st.warning(f"Could not analyze video streams: {e}")
                    # Fallback defaults
                    decode_stream_type = "video"
                    decode_stream_index = 0

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
                elif stego_ext in SUPPORTED_VIDEO_EXTS:
                    # Check which video method was selected for decoding
                    selected_decode_method = st.session_state.get("video_decode_method", "Frame-based")
                    
                    with st.spinner("Decoding video payload..."):
                        if selected_decode_method == "Frame-based":
                            # NEW IFRAME-ONLY extraction - completely separate from stream-based
                            step_val = int(st.session_state.get("vid_step_preview_decode", 10))
                            
                            try:
                                # Call the NEW iframe-only extraction function
                                do_extract_video_iframe(stego_path, out_path, key, lsb, step_val)
                                st.success("✅ Extracted payload from video (IFRAME-ONLY)")
                            except Exception as e:
                                if "Bad magic" in str(e) or "Bad magic" in getattr(e, "args", [""])[0] or "iframe-only" in str(e).lower():
                                    step_candidates = [step_val] + [1, 2, 3, 5, 10, 15, 20, 24, 25, 30]
                                    lsb_candidates = [lsb] + [i for i in range(1, 7) if i != lsb]  # Max 6 for iframe
                                    tried = set()
                                    found = None
                                    for lsb_try in lsb_candidates:
                                        for step_try in step_candidates:
                                            key_t = (lsb_try, step_try)
                                            if key_t in tried:
                                                continue
                                            tried.add(key_t)
                                            try:
                                                # Try IFRAME-ONLY extraction with different parameters
                                                do_extract_video_iframe(stego_path, out_path, key, int(lsb_try), int(step_try))
                                                found = (lsb_try, step_try)
                                                break
                                            except Exception as e2:
                                                if ("Bad magic" in str(e2) or "Bad magic" in getattr(e2, "args", [""])[0] or 
                                                    "iframe-only" in str(e2).lower()):
                                                    continue
                                                else:
                                                    # some other error; surface it
                                                    raise
                                        if found:
                                            break
                                    if found:
                                        st.info(f"🔍 Auto-detected IFRAME settings: LSB={found[0]}, frame step={found[1]}")
                                        st.success("✅ Extracted payload from video (IFRAME-ONLY)")
                                    else:
                                        st.error("❌ Failed to extract from IFRAME-ONLY video. Possible causes: wrong key/LSB, wrong frame step, or this video uses stream-based encoding instead of iframe-only. Try stream-based decoding or check encoding method.")
                                        st.stop()
                                else:
                                    raise
                        
                        else:
                            # New stream-based extraction
                            from main import do_extract_video_stream
                            
                            # Get stream parameters
                            decode_stream_type = st.session_state.get("stream_type_decode", "video")
                            if decode_stream_type == "video":
                                decode_stream_index = int(st.session_state.get("video_stream_idx_decode", 0))
                            else:
                                decode_stream_index = int(st.session_state.get("audio_stream_idx_decode", 0))
                            
                            try:
                                do_extract_video_stream(stego_path, out_path, key, lsb)
                                st.success(f"Extracted payload from video stream")
                            except Exception as e:
                                st.error(f"Failed to extract from {decode_stream_type} stream {decode_stream_index}: {e}")
                                st.info("Ensure the stream type, index, and LSB settings match those used during encoding.")
                                st.stop()
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

# ---------- Video (MP4/MOV/MKV) Experimental UI ----------
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv"}


