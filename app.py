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
