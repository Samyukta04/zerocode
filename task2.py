import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
from transformers import pipeline

image = cv2.imread("sample_inputs/theme.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image.reshape(-1, 3)

kmeans = KMeans(n_clusters=5)
kmeans.fit(pixels)
dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
hex_colors = ['#{:02x}{:02x}{:02x}'.format(*color) for color in dominant_colors]

prompt = f"Describe a UI theme using these colors {hex_colors} and this description: Futuristic neon vibe for interactive UI. Output JSON with theme config."

gen = pipeline("text2text-generation", model="google/flan-t5-base")
response = gen(prompt, max_new_tokens=200)[0]["generated_text"]

os.makedirs("output", exist_ok=True)
with open("output/theme_config.json", "w") as f:
    json.dump({"theme_description": response}, f)

print("✅ Theme config saved to output/theme_config.json")
