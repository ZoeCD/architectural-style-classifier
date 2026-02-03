from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from src.predict import ArchStyleClassifier

app = FastAPI(
    title="Architectural Style Classifier API",
    description="API for classifying architectural styles using a pre-trained EfficientNet model.",
)

classifier = ArchStyleClassifier(
    model_path="../models/best_model.pth",
    labels_path="../models/class_labels.json"
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