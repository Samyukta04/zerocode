import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import pdfplumber
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import easyocr

text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

logo_img = Image.open("sample_inputs/logo.png")
reader = easyocr.Reader(['en'])
logo_text = reader.readtext('sample_inputs/logo.png', detail=0)
logo_text = " ".join(logo_text)
with open("sample_inputs/persona.txt", "r") as f:
    persona_text = f.read()

with pdfplumber.open("doc.pdf") as pdf:
    pdf_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def truncate_text(text, processor, max_length=77):
    tokens = processor.tokenizer(
        text, 
        max_length=max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    return processor.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)

pdf_text_trunc = truncate_text(pdf_text, clip_processor)
logo_text_trunc = truncate_text(logo_text, clip_processor)

clip_inputs = clip_processor(
    text=[pdf_text_trunc, logo_text_trunc], 
    images=logo_img, 
    return_tensors="pt", 
    padding=True
)

text_inputs = {k: v for k, v in clip_inputs.items() if k != 'pixel_values'}
clip_outputs = clip_model.get_text_features(**text_inputs)
clip_vector = clip_outputs.detach().numpy().mean(axis=0)

persona_vector = text_embedder.encode(persona_text)


def match_vector_dims(vec1, vec2):
    max_len = max(len(vec1), len(vec2))
    def pad(v): return np.pad(v, (0, max_len - len(v)), mode='constant')
    return pad(vec1), pad(vec2)

clip_vector_fixed, persona_vector_fixed = match_vector_dims(clip_vector, persona_vector)
final_vector = np.mean([clip_vector_fixed, persona_vector_fixed], axis=0)

os.makedirs("output", exist_ok=True)
with open("output/identity_vector.json", "w") as f:
    json.dump(final_vector.tolist(), f)

print("✅ Identity vector saved to output/identity_vector.json")