from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

import os
from src.predict import ArchStyleClassifier

app = FastAPI(
    title="Architectural Style Classifier API",
    description="API for classifying architectural styles using a pre-trained EfficientNet model.",
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "..", "models", "best_model.pth")
labels_path = os.path.join(base_dir, "..", "models", "class_labels.json")
classifier = ArchStyleClassifier(
    model_path=model_path,
    labels_path=labels_path
)

@app.get("/")
def root():
    return {"message": "Architectural Style Classifier API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    predictions = classifier.predict(image)

    return {"predictions": predictions, 
            "top_style": max(predictions, key=predictions.get)}