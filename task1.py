import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR"
from PIL import Image
import pdfplumber
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Load models
text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load files
logo_img = Image.open("sample_inputs/logo.png")
with open("sample_inputs/persona.txt", "r") as f:
    persona_text = f.read()

with pdfplumber.open("doc.pdf") as pdf:
    pdf_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# OCR on logo image
logo_text = pytesseract.image_to_string(logo_img)

# Embed text using CLIP
clip_inputs = clip_processor(text=[pdf_text, logo_text], images=logo_img, return_tensors="pt", padding=True)
clip_outputs = clip_model.get_text_features(**clip_inputs)
clip_vector = clip_outputs.detach().numpy().mean(axis=0)

# Embed persona using SentenceTransformer
persona_vector = text_embedder.encode(persona_text)

# Combine vectors
final_vector = np.mean([clip_vector, persona_vector], axis=0)

# Save as JSON
os.makedirs("output", exist_ok=True)
with open("output/identity_vector.json", "w") as f:
    json.dump(final_vector.tolist(), f)

print("✅ Identity vector saved to output/identity_vector.json")
