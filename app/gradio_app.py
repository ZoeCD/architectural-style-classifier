import gradio as gr
from PIL import Image
from src.predict import ArchStyleClassifier


classifier = ArchStyleClassifier(
    model_path="models/arch_style_classifier.pth",
    labels_path="data/class_names.json"
)

def classify_architecture(image):
    if image is None:
        return {}
    return classifier.predict(Image.fromarray(image))

# Example images

examples = [
    ["../example_images/art-deco.JPG"],
    ["../example_images/gothic.jpg"],
    ["../example_images/tudor-revival.JPG"],
]

demo = gr.Interface(
    fn = classify_architecture,
    inputs = gr.Image(label="Upload an Image of a Building"),
    outputs = gr.Label(num_top_classes=5, label="Predicted Architectural Styles"),
    title = "Architectural Style Classifier",
    description = "Upload an image of a building to classify its architectural style using a pre-trained EfficientNet model. Trained on 25 styles.",
    examples = examples
)

if __name__ == "__main__":
    demo.launch()