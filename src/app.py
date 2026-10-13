from pathlib import Path

import gradio as gr
import requests
import torch
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "assets" / "examples"
LABELS_URL = "https://git.io/JJkYN"

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Download human-readable ImageNet labels
response = requests.get(LABELS_URL, timeout=30)
response.raise_for_status()
labels = [l.strip() for l in response.text.split("\n") if l.strip()]

# Image preprocessing required by ResNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(inp):
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    return {labels[i]: float(prediction[i]) for i in range(len(labels))}


def build_demo():
    examples = [str(p) for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        examples=examples or None,
    )


if __name__ == "__main__":
    build_demo().launch()
