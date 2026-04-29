<div align="center">

# 🐾 Animal Species Identifier

### AI-Powered Wildlife Recognition with Voice Guide

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white)](https://ultralytics.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-F97316?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Upload any animal photo — get instant species identification, ecological facts, and an audio voice guide.**

<br/>



</div>

---

## 📸 App Preview

```
┌─────────────────────────────────────────────────────────┐
│            🐾 ANIMAL SPECIES IDENTIFIER                 │
│─────────────────────────────────────────────────────────│
│  ┌──────────────┐   ┌──────────────────────────────┐   │
│  │              │   │  🔍 Top Identification         │   │
│  │  [ Upload /  │   │  ──────────────────────       │   │
│  │   Camera /   │   │  🦁 Lion  —  97.3%            │   │
│  │   Paste ]    │   │  🐆 Leopard — 1.8%            │   │
│  │              │   │  🐯 Tiger  — 0.9%             │   │
│  └──────────────┘   └──────────────────────────────┘   │
│  ┌──────────────────────────┐  ┌─────────────────────┐  │
│  │  📋 Species Info          │  │  🔊 Voice Guide      │  │
│  │  Habitat: African Savanna │  │  ▶ Play Audio       │  │
│  │  Status: Vulnerable       │  │                     │  │
│  │  Diet: Carnivore          │  └─────────────────────┘  │
│  └──────────────────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **AI Classification** | YOLOv8-powered model trained on 90 animal species |
| 📊 **Confidence Scores** | Top-3 predictions with percentage confidence bars |
| 📋 **Species Knowledge Base** | Habitat, diet, lifespan, weight, conservation status & fun facts |
| 🔊 **Voice Guide** | Auto-generated audio narration via gTTS for every identification |
| 📷 **Multiple Input Modes** | Upload file, use webcam, or paste from clipboard |
| 🌿 **Age Estimation Clues** | Visual cues to estimate the animal's age from the image |
| 🎨 **Dark Sci-fi UI** | Custom animated scanning UI with emerald green theme |
| 🔄 **One-click Reset** | Scan Another Image button to start fresh instantly |

---

## 🐾 Supported Species (90 Classes)

<details>
<summary><b>Click to expand full species list</b></summary>

| # | Species | # | Species | # | Species |
|---|---------|---|---------|---|---------|
| 0 | 🦌 Antelope | 30 | 🪿 Goose | 60 | 🦜 Parrot |
| 1 | 🦡 Badger | 31 | 🦍 Gorilla | 61 | 🐦 Pelecaniformes |
| 2 | 🦇 Bat | 32 | 🦗 Grasshopper | 62 | 🐧 Penguin |
| 3 | 🐻 Bear | 33 | 🐹 Hamster | 63 | 🐷 Pig |
| 4 | 🐝 Bee | 34 | 🐇 Hare | 64 | 🕊️ Pigeon |
| 5 | 🪲 Beetle | 35 | 🦔 Hedgehog | 65 | 🦔 Porcupine |
| 6 | 🦬 Bison | 36 | 🦛 Hippopotamus | 66 | 🐾 Possum |
| 7 | 🐗 Boar | 37 | 🐦 Hornbill | 67 | 🦝 Raccoon |
| 8 | 🦋 Butterfly | 38 | 🐴 Horse | 68 | 🐀 Rat |
| 9 | 🐈 Cat | 39 | 🐦 Hummingbird | 69 | 🦌 Reindeer |
| 10 | 🐛 Caterpillar | 40 | 🐾 Hyena | 70 | 🦏 Rhinoceros |
| 11 | 🐒 Chimpanzee | 41 | 🪼 Jellyfish | 71 | 🐦 Sandpiper |
| 12 | 🪳 Cockroach | 42 | 🦘 Kangaroo | 72 | 🐴 Seahorse |
| 13 | 🐄 Cow | 43 | 🐨 Koala | 73 | 🦭 Seal |
| 14 | 🐺 Coyote | 44 | 🐞 Ladybugs | 74 | 🦈 Shark |
| 15 | 🦀 Crab | 45 | 🐆 Leopard | 75 | 🐑 Sheep |
| 16 | 🐦 Crow | 46 | 🦁 Lion | 76 | 🐍 Snake |
| 17 | 🦌 Deer | 47 | 🦎 Lizard | 77 | 🐦 Sparrow |
| 18 | 🐕 Dog | 48 | 🦞 Lobster | 78 | 🦑 Squid |
| 19 | 🐬 Dolphin | 49 | 🦟 Mosquito | 79 | 🐿️ Squirrel |
| 20 | 🫏 Donkey | 50 | 🦋 Moth | 80 | ⭐ Starfish |
| 21 | 🪷 Dragonfly | 51 | 🐭 Mouse | 81 | 🦢 Swan |
| 22 | 🦆 Duck | 52 | 🐙 Octopus | 82 | 🐯 Tiger |
| 23 | 🦅 Eagle | 53 | 🦒 Okapi | 83 | 🦃 Turkey |
| 24 | 🐘 Elephant | 54 | 🦧 Orangutan | 84 | 🐢 Turtle |
| 25 | 🦩 Flamingo | 55 | 🦦 Otter | 85 | 🐋 Whale |
| 26 | 🪰 Fly | 56 | 🦉 Owl | 86 | 🐺 Wolf |
| 27 | 🦊 Fox | 57 | 🐂 Ox | 87 | 🐾 Wombat |
| 28 | 🐐 Goat | 58 | 🦪 Oyster | 88 | 🐦 Woodpecker |
| 29 | 🐟 Goldfish | 59 | 🐼 Panda | 89 | 🦓 Zebra |

</details>

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11.9** (recommended)
- `pip` package manager
- `best.pt` model weights file

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/animal-species-identifier.git
cd animal-species-identifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Model Weights

Place your trained weights files in the project root:

```
animal-species-identifier/
├── app.py
├── best.pt           ← your trained YOLOv8 weights
├── last.pt           ← last checkpoint (optional)
├── class_labels.json
└── requirements.txt
```

### 4. Launch the App

```bash
python app.py
```

Then open your browser at **`http://localhost:7860`** 🎉

---

## 📦 Requirements

```txt
gradio>=4.0.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
gTTS
```

> 💡 **Tip:** Using a virtual environment is recommended.
>
> ```bash
> python -m venv venv
> source venv/bin/activate     # macOS/Linux
> venv\Scripts\activate        # Windows
> pip install -r requirements.txt
> ```

---

## 🏗️ Project Structure

```
animal-species-identifier/
│
├── app.py                  # Main Gradio application
├── best.pt                 # Best YOLOv8 model weights
├── last.pt                 # Last training checkpoint
├── class_labels.json       # 90-class species label map
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ How It Works

```
User uploads image
        │
        ▼
┌───────────────────┐
│   YOLOv8 Model    │  ← best.pt weights
│  (Classification) │
└───────────────────┘
        │
        ▼
  Top-3 Predictions
  + Confidence Scores
        │
        ▼
┌───────────────────┐
│  Species Knowledge │  ← SPECIES_INFO dictionary
│      Base          │     (90 animals × 10+ fields)
└───────────────────┘
        │
        ▼
  Habitat / Diet /
  Lifespan / Status /
  Age Clues / Fun Facts
        │
        ▼
┌───────────────────┐
│   gTTS Voice       │  ← Auto-generated audio
│   Narration        │     played in browser
└───────────────────┘
```

1. **Image ingestion** — accepts upload, webcam capture, or clipboard paste via Gradio.
2. **Inference** — the YOLO classifier runs `best.pt` and returns the top-3 predicted classes with confidence scores.
3. **Knowledge lookup** — the app queries the built-in `SPECIES_INFO` dictionary for rich ecological data on the top prediction.
4. **Voice generation** — a gTTS audio narration is synthesised on-the-fly and served as a playable audio widget.

---

## 🌍 Species Info Fields

Every identified animal displays the following data:

| Field | Example (Lion) |
|---|---|
| 🏠 Habitat | African Savanna |
| 🧬 Type | Mammal |
| 🛡️ Conservation Status | Vulnerable |
| 🍖 Diet | Carnivore |
| ⏳ Lifespan | 10–14 years (wild) |
| ⚖️ Average Weight | 120–250 kg |
| 📏 Average Length | 1.7–2.5 m body |
| 🔍 Age Estimation Clues | Cubs have spotted coats... |
| 👶 Young Called | Cub |
| 👥 Social Structure | Pride |
| 💡 Fun Fact | Lions are the only cats that live in social groups |

---

## 🔧 Configuration

You can override the default file paths using environment variables:

```bash
# Use a custom weights file
export WEIGHTS_PATH=/path/to/my_model.pt

# Use a custom labels file
export LABELS_PATH=/path/to/my_labels.json

python app.py
```

---

## 🖥️ UI Walkthrough

### Step 1 — Scan Page
- Upload an image, use your webcam, or paste from clipboard.
- The **Identify Animal** button activates once an image is loaded.
- An animated green scan line plays over the image while processing.

### Step 2 — Results Page
- **Left panel:** your uploaded image.
- **Right panel:** top-3 predictions with confidence percentage bars.
- **Bottom left:** full species info card with all ecological fields.
- **Bottom right:** auto-playing voice guide audio.
- Hit **🔄 Scan Another Image** to reset and start over.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

### Ideas for contributions

- [ ] Add more species (currently 90)
- [ ] Multi-language voice guide support
- [ ] Batch image processing
- [ ] Export results as PDF report
- [ ] Map view showing animal's native habitat range

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — state-of-the-art object detection & classification
- [Gradio](https://www.gradio.app/) — rapid ML web UI framework
- [gTTS](https://pypi.org/project/gTTS/) — Google Text-to-Speech Python library
- [PyTorch](https://pytorch.org/) — deep learning framework

---

<div align="center">

Made with ❤️ and 🐾 for wildlife education

**⭐ Star this repo if you found it useful!**

</div>
