<<<<<<< HEAD
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
import os
import io
import base64
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from steganography.lsb_image import ImageLSBSteganography

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure upload folders
UPLOAD_FOLDER = 'uploads'
STEGO_FOLDER = 'stego_output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STEGO_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STEGO_FOLDER'] = STEGO_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'image': {'png', 'bmp', 'gif', 'jpg', 'jpeg'},
    'audio': {'wav', 'pcm'},
    'payload': {'txt', 'pdf', 'exe', 'bin', 'py', 'java', 'c', 'cpp'}
}

def allowed_file(filename, file_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['GET', 'POST'])
def encode():
    if request.method == 'POST':
        try:
            # Get form data
            cover_file = request.files.get('cover_file')
            payload_type = request.form.get('payload_type')
            payload_text = request.form.get('payload_text')
            payload_file = request.files.get('payload_file')
            key = int(request.form.get('key'))
            lsb_count = int(request.form.get('lsb_count', 1))
            cover_type = request.form.get('cover_type')
            embed_x = int(request.form.get('embed_x', 0))
            embed_y = int(request.form.get('embed_y', 0))
            
            # Validate inputs
            if not cover_file or not key:
                flash('Please provide cover file and key', 'error')
                return redirect(url_for('encode'))
            
            # Prepare payload
            if payload_type == 'text' and payload_text:
                payload_data = payload_text.encode('utf-8')
            elif payload_type == 'file' and payload_file:
                payload_data = payload_file.read()
            else:
                flash('Please provide payload data', 'error')
                return redirect(url_for('encode'))
            
            # Save cover file
            cover_filename = secure_filename(cover_file.filename)
            cover_path = os.path.join(app.config['UPLOAD_FOLDER'], cover_filename)
            cover_file.save(cover_path)
            
            # Process based on cover type
            if cover_type == 'image':
                # Use image LSB steganography
                lsb_stego = ImageLSBSteganography()
                
                # Force BMP format for stego output
                name, ext = os.path.splitext(cover_filename)
                stego_filename = f"{name}_stego.bmp"  # Always use BMP
                stego_path = os.path.join(app.config['STEGO_FOLDER'], stego_filename)
                
                # Perform encoding
                success = lsb_stego.encode(
                    cover_image_path=cover_path,
                    payload=payload_data,
                    output_path=stego_path,
                    key=key,
                    lsb_bits=lsb_count,
                    start_pos=(embed_x, embed_y)
                )
                
                if success:
                    flash(f'Encoding successful! Stego file saved as {stego_filename}', 'success')
                    # Store result info in session for display
                    session['last_encoding'] = {
                        'cover_file': cover_filename,
                        'stego_file': stego_filename,
                        'payload_size': len(payload_data),
                        'lsb_bits': lsb_count,
                        'key': key
                    }
                else:
                    flash('Encoding failed. Please check your inputs.', 'error')
                    
            elif cover_type == 'audio':
                flash('Audio encoding will be implemented next', 'info')
            
            return redirect(url_for('encode'))
            
        except Exception as e:
            flash(f'Error during encoding: {str(e)}', 'error')
            return redirect(url_for('encode'))
    
    return render_template('encode.html')

@app.route('/decode', methods=['GET', 'POST'])
def decode():
    if request.method == 'POST':
        try:
            print("🔍 DEBUG: Starting decode")
            
            stego_file = request.files.get('stego_file')
            key = int(request.form.get('key'))
            lsb_count = int(request.form.get('lsb_count', 1))
            cover_type = request.form.get('cover_type')
            start_x = int(request.form.get('start_x', 0))
            start_y = int(request.form.get('start_y', 0))
            
            print(f"🔍 DEBUG: Key: {key}, LSB count: {lsb_count}")
            
            if not stego_file or not key:
                flash('Please provide stego file and key', 'error')
                return redirect(url_for('decode'))
            
            # Save stego file
            stego_filename = secure_filename(stego_file.filename)
            stego_path = os.path.join(app.config['UPLOAD_FOLDER'], stego_filename)
            stego_file.save(stego_path)
            
            if cover_type == 'image':
                # Use image LSB steganography
                lsb_stego = ImageLSBSteganography()
                
                # Perform decoding
                payload_data = lsb_stego.decode(
                    stego_image_path=stego_path,
                    key=key,
                    lsb_bits=lsb_count,
                    start_pos=(start_x, start_y)
                )
                
                if payload_data:
                    # Try to decode as text first
                    try:
                        payload_text = payload_data.decode('utf-8')
                        flash(f'Decoding successful! Message: "{payload_text}"', 'success')
                        session['last_decoding'] = {
                            'payload_text': payload_text,
                            'payload_size': len(payload_data),
                            'is_text': True,
                            'key_used': key,
                            'lsb_bits_used': lsb_count
                        }
                    except UnicodeDecodeError:
                        # Binary data
                        flash(f'Decoding successful! Extracted {len(payload_data)} bytes of binary data', 'success')
                        session['last_decoding'] = {
                            'payload_data': payload_data.hex(),
                            'payload_size': len(payload_data),
                            'is_text': False,
                            'key_used': key,
                            'lsb_bits_used': lsb_count
                        }
                else:
                    flash('Decoding failed. Please check your key and settings.', 'error')
                    
            elif cover_type == 'audio':
                flash('Audio decoding will be implemented next', 'info')
            
            return redirect(url_for('decode'))
            
        except Exception as e:
            flash(f'Error during decoding: {str(e)}', 'error')
            return redirect(url_for('decode'))
    
    return render_template('decode.html')

@app.route('/clear_session')
def clear_session():
    session.pop('last_decoding', None)
    session.pop('last_encoding', None)
    flash('Results cleared', 'info')
    return redirect(url_for('decode'))

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/api/calculate_capacity', methods=['POST'])
def calculate_capacity():
    try:
        # This would need the actual file, but for now return estimated capacity
        data = request.get_json()
        cover_type = data.get('cover_type')
        lsb_count = int(data.get('lsb_count', 1))
        file_size = int(data.get('file_size', 0))
        
        if cover_type == 'image':
            # Rough estimate: assume average image has width*height*3 bytes
            # This is simplified - real calculation needs actual image dimensions
            estimated_pixels = file_size // 3  # Very rough estimate
            capacity_bits = estimated_pixels * lsb_count
            capacity_bytes = capacity_bits // 8 - 8  # Subtract header
        elif cover_type == 'audio':
            # For 16-bit audio: file_size/2 samples
            estimated_samples = file_size // 2
            capacity_bits = estimated_samples * lsb_count
            capacity_bytes = capacity_bits // 8 - 8  # Subtract header
        else:
            capacity_bytes = 0
        
        return jsonify({
            'capacity_bits': capacity_bits,
            'capacity_bytes': max(0, capacity_bytes),
            'capacity_kb': round(max(0, capacity_bytes) / 1024, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
=======
import io
import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from main import (
    do_embed_image, do_extract_image, do_embed_audio, do_extract_audio,
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
    data = uploaded_file.read()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

def encode_ui():
    st.subheader("Encode: Hide a payload inside a cover")
    c1, c2 = st.columns(2)

    with c1:
        cover_up = st.file_uploader("Cover file (.png/.bmp or .wav)", type=["png", "bmp", "wav"], key="cover")
        payload_up = st.file_uploader("Payload file (any)", type=None, key="payload")
        out_name = st.text_input("Output stego filename", value="stego")
        ext_choice = st.selectbox("Output type", [".png", ".bmp", ".wav"], index=0)
        go = st.button("Embed", type="primary")

        if cover_up is not None:
            # Capacity check
            try:
                cov_ext = os.path.splitext(cover_up.name)[1].lower()
                cov_bytes = cover_up.getvalue()
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
        if cover_up is not None and cover_up.type.startswith("image/"):
            st.image(cover_up, caption=f"Cover: {cover_up.name}", width='stretch')
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
                payload_path = _save_to_tmp(payload_up, suffix=os.path.splitext(payload_up.name)[1] or ".txt")
                out_path = os.path.join(tempfile.gettempdir(), (out_name or "stego") + ext_choice)

                if cov_ext in SUPPORTED_IMAGE_EXTS and ext_choice in SUPPORTED_IMAGE_EXTS:
                    do_embed_image(cover_path, payload_path, out_path, key, lsb)
                    st.success("Embedded into image stego")
                    # Show stego and diff map
                    with open(cover_path, "rb") as f: cov_img = Image.open(io.BytesIO(f.read()))
                    with open(out_path, "rb") as f: stego_img = Image.open(io.BytesIO(f.read()))
                    cov_rgba = np.array(Image.open(cover_path).convert("RGBA"), dtype=np.uint8)
                    stego_rgba = np.array(Image.open(out_path).convert("RGBA"), dtype=np.uint8)
                    st.image(stego_img, caption="Stego image", width='stretch')

                    # Diff map on LSBs used
                    diff = ((stego_rgba[:, :, :3]) ^ (cov_rgba[:, :, :3])) & ((1 << lsb) - 1)

                    # amplify to visible 0..255
                    scale = 255 // ((1 << lsb) - 1)
                    diff_vis = (diff * scale).astype(np.uint8)
                    st.image(diff_vis, caption=f"Difference map of used LSBs (x{scale})", width='stretch')
                    # Download
                    with open(out_path, "rb") as f:
                        st.download_button("Download stego image", f, file_name=os.path.basename(out_path))

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
                        n_ch = wf.getnchannels(); n_frames = wf.getnframes()
                        raw = wf.readframes(min(n_frames, 50000))
                    arr = np.frombuffer(raw, dtype=np.int16)
                    lsb_mask = (1 << lsb) - 1
                    lsb_wave = (arr.view(np.uint16) & lsb_mask).astype(np.uint16)
                    fig = plt.figure()
                    plt.plot(lsb_wave)
                    plt.title("LSB waveform (subset)")
                    st.pyplot(fig)
                    with open(out_path, "rb") as f:
                        st.download_button("Download stego audio", f, file_name=os.path.basename(out_path))
                else:
                    st.error("Output type must match cover family (image→.png/.bmp, audio→.wav)")
            except Exception as e:
                st.error(f"Embed failed: {e}")

def decode_ui():
    st.subheader("Decode: Extract a payload from a stego file")
    stego_up = st.file_uploader("Stego file (.png/.bmp or .wav)", type=["png", "bmp", "wav"], key="stego")
    out_label = st.text_input("Name for extracted file", value="payload.txt")
    go2 = st.button("Extract", type="primary")

    if go2:
        if not key:
            st.error("Key is required")
        elif stego_up is None:
            st.error("Please upload a stego file")
        else:
            try:
                stego_ext = os.path.splitext(stego_up.name)[1].lower()
                stego_path = _save_to_tmp(stego_up, suffix=stego_ext)
                out_path = os.path.join(tempfile.gettempdir(), out_label or "payload.txt")

                if stego_ext in SUPPORTED_IMAGE_EXTS:
                    do_extract_image(stego_path, out_path, key, lsb)
                    st.success("Extracted payload from image")
                elif stego_ext in SUPPORTED_AUDIO_EXTS:
                    do_extract_audio(stego_path, out_path, key, lsb)
                    st.success("Extracted payload from audio")
                else:
                    st.error("Unsupported stego type")
                    st.stop()

                # Offer download and quick preview if small text or image
                with open(out_path, "rb") as f:
                    data = f.read()
                    st.download_button("Download extracted payload", data, file_name=os.path.basename(out_path))
                # Try preview
                try:
                    if out_label.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
                        st.image(data, caption="Extracted image preview")
                    elif out_label.lower().endswith((".txt", ".md", ".json", ".py")) and len(data) < 200_000:
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
        n_ch = wf.getnchannels(); sw = wf.getsampwidth(); n_frames = wf.getnframes()
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
>>>>>>> 2b4e116893d4cf19d91047e505707b5d695021cc
