# Image Classification with ResNet50 + Gradio

Upload an image and get the top 3 predicted ImageNet labels, using a pretrained ResNet50 (torchvision) and a Gradio web UI.

## Setup
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run
    python src/app.py

Put your own example images in `assets/examples/` and they will show up in the UI.
