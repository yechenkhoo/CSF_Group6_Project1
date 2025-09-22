import cv2
import numpy as np
import os

# Settings
video_path = os.path.join('demo', 'media', 'cover_test.mp4')
payload_path = os.path.join('demo', 'payloads', 'message_test.txt')

# Generate a simple video (solid color frames with text)
fps = 24
frame_size = (640, 360)
num_frames = 60

os.makedirs(os.path.dirname(video_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

for i in range(num_frames):
    frame = np.full((frame_size[1], frame_size[0], 3), 180, dtype=np.uint8)
    cv2.putText(frame, f'Test Frame {i+1}', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    out.write(frame)
out.release()

# Generate a payload message
os.makedirs(os.path.dirname(payload_path), exist_ok=True)
with open(payload_path, 'w', encoding='utf-8') as f:
    f.write('This is a test payload message for video steganography.')

print(f'Generated video: {video_path}')
print(f'Generated payload: {payload_path}')
