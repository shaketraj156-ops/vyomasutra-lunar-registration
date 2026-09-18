import base64
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

assets_dir = r"C:\Users\Dell\Desktop\vyomasutra-lunar-registration\assets"
input_2k = os.path.join(assets_dir, "moon_map_2k.jpg")

if not os.path.exists(input_2k):
    raise FileNotFoundError("2K NASA map not found")

img = Image.open(input_2k)

# Crisp photorealistic color map (1024x512)
img_map = img.resize((1024, 512), Image.Resampling.LANCZOS)
enhancer = ImageEnhance.Contrast(img_map)
img_map = enhancer.enhance(1.18)
enhancer_sharp = ImageEnhance.Sharpness(img_map)
img_map = enhancer_sharp.enhance(1.25)

map_out = os.path.join(assets_dir, "moon_map_final.jpg")
img_map.save(map_out, 'JPEG', quality=88)

# Bump map for crisp crater depth & rim shadows
gray = ImageOps.grayscale(img_map)
bump = ImageOps.autocontrast(gray, cutoff=2)
bump = bump.filter(ImageFilter.EDGE_ENHANCE_MORE)
bump_out = os.path.join(assets_dir, "moon_bump_final.jpg")
bump.save(bump_out, 'JPEG', quality=85)

with open(map_out, 'rb') as f:
    b64_map = base64.b64encode(f.read()).decode('utf-8')

with open(bump_out, 'rb') as f:
    b64_bump = base64.b64encode(f.read()).decode('utf-8')

py_out = os.path.join(assets_dir, "moon_texture_data.py")
with open(py_out, 'w') as f:
    f.write(f'# Official NASA LROC Photorealistic Moon Textures\n')
    f.write(f'NASA_MOON_MAP_B64 = "{b64_map}"\n')
    f.write(f'NASA_MOON_BUMP_B64 = "{b64_bump}"\n')

print(f"Successfully created {py_out} with size {os.path.getsize(py_out)} bytes")
