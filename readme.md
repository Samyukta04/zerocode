# ZeroCode AI/ML Internship Assignment- SAMYUKTA GADE

This repository contains solutions for the AI/ML Intern assignment from ZeroCode. The assignment consists of two tasks involving multimodal processing using text, image, and PDF inputs to generate meaningful vector and UI theme outputs.

---

## 🧩 Task 1: Multimodal Identity Extractor

**Objective:**  
Generate an identity vector by analyzing a company’s branding assets — logo, persona description, and brand style guide PDF.

### 🔍 Inputs:
- `sample_inputs/logo.png`: Image of the company logo
- `sample_inputs/persona.txt`: A 1-2 line persona description
- `doc.pdf`: A PDF file describing the company branding and design

### ⚙️ Output:
- `output/identity_vector.json`: A JSON file containing the final identity vector generated from combined embeddings

---

## 🎨 Task 2: Visual Theme Interpreter & Config Generator

**Objective:**  
Generate a structured JSON UI theme configuration based on the image and prompt description.

### 🔍 Inputs:
- `sample_inputs/theme.png`: Theme reference image
- A hardcoded prompt in code: *"Futuristic neon vibe for interactive UI"*

### ⚙️ Output:
- `output/theme_config.json`: A JSON file with theme description/configuration generated via a language model

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have Tesseract OCR installed and added to PATH. If not, install from:
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Run Task 1

```bash
python task1.py
```

### 3. Run Task 2

```bash
python task2.py
```

---

## 📦 Project Structure

```
Zerocode/
└── output/
    ├── identity_vector.json
    └── theme_config.json
├── sample_inputs/
│   ├── logo.png
│   ├── theme.png
│   └── persona.txt
├── .gitignore
├── doc.pdf
├── readme.md
├── requirements.txt
├── task1.py
├── task2.py
├──zerocode-env

```

---

## ✅ Sample Output

### `identity_vector.json`

```json
[0.13452, 0.39821, 0.28901, ...]
```

### `theme_config.json`

```json
{
  "theme_description": "This futuristic UI theme uses neon blues and dark backgrounds to deliver an immersive, tech-forward interface..."
}
```

---

## 🙌 Author

**Samyukta Gade**  
AI/ML Intern Applicant | ZeroCode  
[GitHub Profile](https://github.com/Samyukta04)  
