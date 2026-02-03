# 🏛️ Architectural Style Classifier

A deep learning model that classifies buildings into 25 architectural styles. Built with PyTorch and EfficientNet, deployed as both an interactive Gradio demo and a REST API.


## 🎯 Live Demo

**Try it now:** [Hugging Face Space](https://huggingface.co/spaces/ZoeCD/architectural-styles-classifier)

Upload any building image and get instant predictions across 25 architectural styles.

## 📸 Example

| Input | Prediction |
|-------|------------|
| ![Gothic Cathedral](example_images/gothic.jpg) | **Gothic** (100%) |
| ![Tudor Revival House](example_images/tudor-revival.JPG) | **Tudor Revival** (95%) |
| ![Art Deco Hotel](example_images/art-deco.JPG) | **Art Deco** (98%) |

## 🏗️ Architectural Styles

The model recognizes 25 styles:

| | | | | |
|---|---|---|---|---|
| Achaemenid | American Craftsman | American Foursquare | Ancient Egyptian | Art Deco |
| Art Nouveau | Baroque | Bauhaus | Beaux-Arts | Byzantine |
| Chicago School | Colonial | Deconstructivism | Edwardian | Georgian |
| Gothic | Greek Revival | International | Novelty | Palladian |
| Postmodern | Queen Anne | Romanesque | Russian Revival | Tudor |

## 🚀 Quick Start

### Option 1: Use the API

```python
import requests

response = requests.post(
    "https://architectural-style-classifier.onrender.com/predict",
    files={"file": (os.path.basename(image_path), open("building.jpg", "rb"), "image/jpg")}
)

result = response.json()
print(result["top_style"])  # "Gothic"
print(result["predictions"])  # {"Gothic": 0.892, "Romanesque": 0.056, ...}
```

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/ZoeCD/architectural-style-classifier.git
cd architectural-style-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app/gradio_app.py
```

Then open http://localhost:7860 in your browser.

### Option 3: Docker

```bash
docker build -t arch-classifier .
docker run -p 8000:8000 arch-classifier
```

API available at http://localhost:8000. Interactive docs at http://localhost:8000/docs.

## 📁 Project Structure

```
architectural-style-classifier/
├── app/
│   ├── gradio_app.py      # Hugging Face Spaces demo
│   └── main.py            # FastAPI REST API
├── src/
│   ├── train.py           # Training script
│   └── predict.py         # Inference module
├── models/
│   ├── arch_classifier_best.pth
│   └── class_labels.json
├── notebooks/
│   └── data-exploration.ipynb
|   └── example-api.ipynb  # API request example
├── example_images/              # Sample images for demo
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🔧 API Reference

### `GET /`

Health check endpoint.

**Response:**
```json
{"message": "Architectural Style Classifier API", "status": "running"}
```

### `GET /health`

Service health status.

**Response:**
```json
{"status": "healthy"}
```

### `POST /predict`

Classify an architectural image.

**Request:** Multipart form with image file

**Response:**
```json
{
  "predictions": {
    "Gothic": 0.892,
    "Romanesque": 0.056,
    "Byzantine": 0.023,
    "Baroque": 0.012,
    "Art Nouveau": 0.008
  },
  "top_style": "Gothic"
}
```

## 🧠 Model Details

| Attribute | Value |
|-----------|-------|
| Architecture | EfficientNet-B0 |
| Pretrained on | ImageNet |
| Fine-tuned on | 15,432 architectural images |
| Input size | 224 × 224 |
| Output | 25 classes |
| Test accuracy | ~70% |

### Training Approach

1. **Transfer learning**: Started with ImageNet-pretrained EfficientNet-B0
2. **Two-phase training**: 
   - Phase 1: Trained classifier head only
   - Phase 2: Fine-tuned entire network with lower learning rate
3. **Class imbalance handling**: Weighted cross-entropy loss
4. **Data augmentation**: Random flips, rotation, color jitter

## 📊 Dataset

Based on the [Architectural Styles Dataset](https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset) from Kaggle, which combines:

- Original dataset from "Architectural Style Classification using Multinomial Latent Logistic Regression" (ECCV 2014) by Zhe Xu et al.
- Additional images scraped from Google Images

## 🙏 Acknowledgments

- Dataset by [dumitrux](https://github.com/dumitrux/architectural-style-recognition) and Zhe Xu et al.
- EfficientNet implementation from [torchvision](https://pytorch.org/vision/stable/models.html)
- Deployed with [Hugging Face Spaces](https://huggingface.co/spaces) and [Render](https://render.com)

