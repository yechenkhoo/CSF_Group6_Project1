"""
LSB Image Steganography Module
Handles LSB replacement for image files (BMP, PNG, GIF)
"""

import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, List

class ImageLSBSteganography:
    """LSB steganography for image files"""
    
    def __init__(self):
        self.supported_formats = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']
    
    def encode(self, cover_image_path: str, payload: bytes, output_path: str, 
            key: int, lsb_bits: int = 1, start_pos: Tuple[int, int] = (0, 0)) -> bool:
        try:
            print(f"🔍 ENCODE: Starting encode with key={key}, lsb_bits={lsb_bits}")
            print(f"🔍 ENCODE: Payload size: {len(payload)} bytes")
            
            # Load cover image
            cover_img = Image.open(cover_image_path)
            
            # Convert to RGB if necessary
            if cover_img.mode != 'RGB':
                cover_img = cover_img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(cover_img)
            height, width, channels = img_array.shape
            print(f"🔍 ENCODE: Image dimensions: {width}x{height}x{channels}")
            
            # Calculate capacity
            total_pixels = (height - start_pos[1]) * (width - start_pos[0])
            capacity_bits = total_pixels * channels * lsb_bits
            capacity_bytes = capacity_bits // 8
            
            # Check if payload fits
            payload_with_header = self._add_payload_header(payload, key)
            print(f"🔍 ENCODE: Payload with header size: {len(payload_with_header)} bytes")
            print(f"🔍 ENCODE: Header (hex): {payload_with_header[:8].hex()}")
            
            if len(payload_with_header) > capacity_bytes:
                raise ValueError(f"Payload too large. Max capacity: {capacity_bytes} bytes")
            
            # Convert payload to binary string
            payload_bits = self._bytes_to_bits(payload_with_header)
            print(f"🔍 ENCODE: Total bits to embed: {len(payload_bits)}")
            
            # Generate LSB positions based on key
            lsb_positions = self._generate_lsb_positions(key, len(payload_bits), 
                                                    height, width, channels, 
                                                    lsb_bits, start_pos)
            
            print(f"🔍 ENCODE: Generated {len(lsb_positions)} positions")
            print(f"🔍 ENCODE: First few positions: {lsb_positions[:5]}")
            
            # Embed payload
            stego_array = self._embed_bits(img_array, payload_bits, lsb_positions, lsb_bits)
            
            # Save stego image
            stego_img = Image.fromarray(stego_array)
        
            # Force BMP format to preserve LSB modifications
            stego_img.save(output_path, 'BMP')
            print(f"✅ ENCODE: Successfully saved as BMP: {output_path}")
            
            # Verify the save worked correctly
            verify_img = Image.open(output_path)
            verify_array = np.array(verify_img)
            first_pos = lsb_positions[0] if lsb_positions else None
            if first_pos:
                y, x, channel, bit_pos = first_pos
                original_pixel = stego_array[y, x, channel]
                reloaded_pixel = verify_array[y, x, channel]
                print(f"🔍 VERIFY: Pixel ({y},{x},{channel}) - embedded: {original_pixel}, reloaded: {reloaded_pixel}")
                if original_pixel != reloaded_pixel:
                    print("❌ WARNING: Pixel values changed after save/reload!")
                else:
                    print("✅ VERIFY: Pixel values preserved correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ ENCODE ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def decode(self, stego_image_path: str, key: int, lsb_bits: int = 1, 
            start_pos: Tuple[int, int] = (0, 0)) -> Optional[bytes]:
        """
        Decode payload from stego image
        """
        try:
            print(f"🔍 DECODE: Starting decode with key={key}, lsb_bits={lsb_bits}")
            
            # Load stego image
            stego_img = Image.open(stego_image_path)
            
            # Convert to RGB if necessary
            if stego_img.mode != 'RGB':
                stego_img = stego_img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(stego_img)
            height, width, channels = img_array.shape
            print(f"🔍 DECODE: Image dimensions: {width}x{height}x{channels}")
            
            # First, extract header to get payload size
            header_bits = 64  # 8 bytes for header (4 bytes size + 4 bytes key)
            header_positions = self._generate_lsb_positions(key, header_bits, 
                                                        height, width, channels,
                                                        lsb_bits, start_pos)
            
            print(f"🔍 DECODE: Generated {len(header_positions)} header positions")
            print(f"🔍 DECODE: First few positions: {header_positions[:5]}")
            
            # Extract header
            header_bits_str = self._extract_bits(img_array, header_positions, lsb_bits)
            print(f"🔍 DECODE: Header bits: {header_bits_str[:32]}...")  # First 32 bits
            
            header_bytes = self._bits_to_bytes(header_bits_str)
            print(f"🔍 DECODE: Header bytes (hex): {header_bytes.hex()}")
            
            # Parse header
            payload_size = int.from_bytes(header_bytes[:4], 'big')
            stored_key = int.from_bytes(header_bytes[4:8], 'big')
            
            print(f"🔍 DECODE: Parsed payload size: {payload_size}")
            print(f"🔍 DECODE: Parsed stored key: {stored_key}")
            print(f"🔍 DECODE: Provided key: {key}")
            print(f"🔍 DECODE: Keys match: {stored_key == key}")
            
            # Check if the extracted values are reasonable
            if payload_size > 10000 or payload_size < 1:
                print(f"❌ DECODE: Suspicious payload size: {payload_size}")
            
            if stored_key < 1 or stored_key > 2**31:
                print(f"❌ DECODE: Suspicious stored key: {stored_key}")
            
            # Verify key
            if stored_key != key:
                print(f"❌ DECODE: Key mismatch! Expected {key}, got {stored_key}")
                raise ValueError("Invalid key")
            
            print("✅ DECODE: Key validation passed")
            
            # Extract full payload
            total_bits_needed = header_bits + (payload_size * 8)
            all_positions = self._generate_lsb_positions(key, total_bits_needed,
                                                    height, width, channels,
                                                    lsb_bits, start_pos)
            
            # Extract payload bits (skip header)
            payload_positions = all_positions[header_bits:]
            payload_bits_str = self._extract_bits(img_array, payload_positions, lsb_bits)
            
            # Convert to bytes
            payload = self._bits_to_bytes(payload_bits_str)
            
            print(f"✅ DECODE: Successfully extracted {len(payload)} bytes")
            return payload
            
        except Exception as e:
            print(f"❌ DECODE ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_capacity(self, image_path: str, lsb_bits: int = 1, 
                          start_pos: Tuple[int, int] = (0, 0)) -> int:
        """Calculate maximum payload capacity in bytes"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            channels = 3  # RGB
            
            available_pixels = (height - start_pos[1]) * (width - start_pos[0])
            capacity_bits = available_pixels * channels * lsb_bits
            capacity_bytes = capacity_bits // 8
            
            # Subtract header size
            return max(0, capacity_bytes - 8)
            
        except Exception as e:
            print(f"Capacity calculation error: {e}")
            return 0
    
    def _add_payload_header(self, payload: bytes, key: int) -> bytes:
        """Add header with payload size and key"""
        size_bytes = len(payload).to_bytes(4, 'big')
        key_bytes = key.to_bytes(4, 'big')
        return size_bytes + key_bytes + payload
    
    def _bytes_to_bits(self, data: bytes) -> str:
        """Convert bytes to binary string"""
        return ''.join(format(byte, '08b') for byte in data)
    
    def _bits_to_bytes(self, bits: str) -> bytes:
        """Convert binary string to bytes"""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits += '0'
        
        return bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
    
    def _generate_lsb_positions(self, key: int, num_bits: int, height: int, 
                            width: int, channels: int, lsb_bits: int,
                            start_pos: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Generate LSB positions based on key - FIXED VERSION
        Returns list of (y, x, channel, bit_position) tuples
        """
        # Use a simple, deterministic approach that doesn't depend on payload size
        positions = []
        start_x, start_y = start_pos
        
        # Calculate available positions
        available_height = height - start_y
        available_width = width - start_x
        total_positions_per_pixel = channels * lsb_bits
        
        for bit_idx in range(num_bits):
            # Use key to offset the starting position, then go sequentially
            offset_position = (bit_idx + key) % (available_height * available_width * total_positions_per_pixel)
            
            # Convert to coordinates
            pixel_index = offset_position // total_positions_per_pixel
            bit_in_pixel = offset_position % total_positions_per_pixel
            
            # Convert pixel index to x, y
            row = pixel_index // available_width
            col = pixel_index % available_width
            
            y = start_y + row
            x = start_x + col
            
            # Convert bit position to channel and bit
            channel = bit_in_pixel // lsb_bits
            bit_pos = bit_in_pixel % lsb_bits
            
            # Ensure we don't go out of bounds
            if y < height and x < width and channel < channels:
                positions.append((y, x, channel, bit_pos))
            else:
                # If we run out of space, break
                break
        
        return positions
    
    def _embed_bits(self, img_array: np.ndarray, bits: str, 
                positions: List[Tuple[int, int, int, int]], lsb_bits: int) -> np.ndarray:
        """Embed bits into image array at specified positions"""
        stego_array = img_array.copy()
        
        for i, bit in enumerate(bits):
            if i >= len(positions):
                break
            
            y, x, channel, bit_pos = positions[i]
            
            if y < img_array.shape[0] and x < img_array.shape[1]:
                # Get original pixel value
                original_value = stego_array[y, x, channel]
                
                # Clear the target bit and set new bit
                pixel_value = stego_array[y, x, channel]
                
                # Clear bit at position
                mask = ~(1 << bit_pos)
                pixel_value = pixel_value & mask
                
                # Set new bit
                if bit == '1':
                    pixel_value = pixel_value | (1 << bit_pos)
                
                stego_array[y, x, channel] = pixel_value
                
                # Debug first 16 bits in detail
                if i < 16:
                    byte_pos = i // 8
                    bit_in_byte = i % 8
                    print(f"Embed {i:2d} (byte {byte_pos}, bit {bit_in_byte}): pos({y},{x},{channel},{bit_pos}) {original_value}->{pixel_value} bit={bit}")
        
        return stego_array
    
    def _extract_bits(self, img_array: np.ndarray, 
                    positions: List[Tuple[int, int, int, int]], lsb_bits: int) -> str:
        """Extract bits from image array at specified positions"""
        bits = ""
        
        for i, (y, x, channel, bit_pos) in enumerate(positions):
            if y < img_array.shape[0] and x < img_array.shape[1]:
                pixel_value = img_array[y, x, channel]
                bit = (pixel_value >> bit_pos) & 1
                bits += str(bit)
                
                # Debug first 16 bits (2 bytes) in detail
                if i < 16:
                    byte_pos = i // 8
                    bit_in_byte = i % 8
                    print(f"Bit {i:2d} (byte {byte_pos}, bit {bit_in_byte}): pos({y},{x},{channel},{bit_pos}) pixel={pixel_value:3d} extracted_bit={bit}")
        
        return bits