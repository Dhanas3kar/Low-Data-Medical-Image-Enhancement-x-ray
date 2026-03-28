import torch
import cv2
import numpy as np
from src.model import UNet

# ====== CONFIG ======
MODEL_PATH = "models/model_final.pt"
INPUT_IMAGE = "test_image.png"
OUTPUT_IMAGE = "output.png"
IMG_SIZE = 256

# ====== LOAD MODEL ======
model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print("✅ Model loaded")

# ====== LOAD IMAGE ======
img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ Image not found! Check file name")
    exit()

# Resize
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# Normalize
img = img / 255.0

# Convert to tensor
img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ====== PREDICT ======
with torch.no_grad():
    output = model(img_tensor)

# ====== POST PROCESS ======
output = output.squeeze().numpy()
output = (output * 255).astype(np.uint8)

output = cv2.GaussianBlur(output, (0,0), 1)
output = cv2.addWeighted(output, 1.5, output, -0.5, 0)

# Save output
cv2.imwrite(OUTPUT_IMAGE, output)

print("🎉 Output saved as", OUTPUT_IMAGE)