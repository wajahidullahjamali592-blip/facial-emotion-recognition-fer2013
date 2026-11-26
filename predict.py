"""
Predict the emotion of one or more face images with the trained model.

Usage:
    python predict.py path/to/face.jpg
    python predict.py img1.jpg img2.png --model outputs/emotion_cnn.pth
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from train import MEAN, STD, EmotionCNN, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="image file(s) containing a face")
    p.add_argument("--model", default="outputs/emotion_cnn.pth")
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.model, map_location=device)
    classes = ckpt["classes"]
    model = EmotionCNN(len(classes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    for path in args.images:
        x = tf(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        top = probs.topk(3)
        print(f"\n{path}")
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {classes[idx]:<9} {score * 100:5.1f}%")


if __name__ == "__main__":
    main()
