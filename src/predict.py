import torch
from torchvision import transforms, models
from PIL import Image
import json

class ArchStyleClassifier:
    def __init__(self, model_path, labels_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load labels
        with open(labels_path, 'r') as f:
            self.class_names = json.load(f)
        
        # Load model
        self.model = models.efficientnet_b0(pretrained=False)
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, len(self.class_names)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        # Return top 5 predictions
        top5_prob, top5_catid = torch.topk(probs, 5)
        results = {
            self.class_names[i]: float(prob) for i, prob in zip(top5_catid[0], top5_prob[0])
        }
        return results