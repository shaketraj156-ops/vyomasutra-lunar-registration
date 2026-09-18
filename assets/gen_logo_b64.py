import os
import base64

root_dir = r"C:\Users\Dell\Desktop\vyomasutra-lunar-registration"
p1 = os.path.join(root_dir, "team_logo.png")
p2 = os.path.join(root_dir, "app", "team_logo.png")

logo_path = p1 if os.path.exists(p1) else p2
if not os.path.exists(logo_path):
    raise FileNotFoundError("team_logo.png not found")

with open(logo_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

out_file = os.path.join(root_dir, "assets", "team_logo_b64.py")
with open(out_file, "w") as f:
    f.write(f'# Team LUNAR FLUX Logo Base64 Data\n')
    f.write(f'TEAM_LOGO_B64 = "{b64}"\n')

print(f"Generated {out_file} successfully ({os.path.getsize(out_file)} bytes)")
