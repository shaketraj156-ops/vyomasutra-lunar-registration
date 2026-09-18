import base64
import os

mp4_path = r"C:\Users\Dell\Desktop\vyomasutra-lunar-registration\data\gemini_generated_video_ef3f1bf2.mp4"
if not os.path.exists(mp4_path):
    raise FileNotFoundError("Local MP4 video not found!")

with open(mp4_path, "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode("utf-8")

out_py = r"C:\Users\Dell\Desktop\vyomasutra-lunar-registration\assets\video_texture_data.py"
with open(out_py, "w") as f:
    f.write(f'# Base64 encoded local video: gemini_generated_video_ef3f1bf2.mp4\n')
    f.write(f'LOCAL_VIDEO_B64 = "{video_b64}"\n')

print(f"Successfully generated {out_py} with size {os.path.getsize(out_py)} bytes")
