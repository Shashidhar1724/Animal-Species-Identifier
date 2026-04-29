---
title: WildLens Animal Species Identifier
emoji: 🐾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Identify animal species with detailed info, age clues & more
---

# 🐾 WildLens — Animal Species Identifier

Upload a photo of an animal and WildLens will identify the species using a YOLOv8 model and provide:

- **Species classification** (type, conservation status, social structure)
- **Habitat & diet**
- **Average size and lifespan**
- **Age estimation clues** — visual cues to guess how old the animal is
- **Fun fact** about the species
- **Top-5 prediction confidence** breakdown

## 🚀 Setup

1. Upload your trained `best.pt` (YOLOv8 weights) to the Space files.
2. Upload your `class_labels.json` (format: `{"0": "antelope", "1": "bear", ...}`).
3. The app auto-loads them on startup.

## 🗂️ File Structure

```
your-space/
├── app.py               ← Main Gradio app (this file)
├── requirements.txt     ← Python dependencies
├── best.pt              ← YOLOv8 model weights  ← YOU MUST ADD THIS
├── class_labels.json    ← Class index → name map ← YOU MUST ADD THIS
└── README.md
```

## 📦 Local Run

```bash
pip install -r requirements.txt
# Place best.pt and class_labels.json in the same folder
python app.py
# Open http://localhost:7860
```

## 🤗 HuggingFace Deployment

1. Create a new Space at https://huggingface.co/spaces
2. Choose **Gradio** as the SDK
3. Upload `app.py`, `requirements.txt`, `best.pt`, and `class_labels.json`
4. The Space will build and launch automatically

---
*Built with [Gradio](https://gradio.app) and [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)*
