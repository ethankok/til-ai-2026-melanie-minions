import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import matplotlib.pyplot as plt
import requests
import base64
import io
import sys
import os

sys.path.append(os.path.abspath("src"))
from noise_manager import NoiseManager

def simulate_attack():
    print("1. Downloading sample image & class labels...")
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    clean_img = Image.open(urllib.request.urlopen(req)).convert("RGB")
    
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(LABELS_URL).text.split('\n')

    print("2. Loading the Victim Model (VGG16)...")
    victim_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval()
    
    # THE FIX: We must resize and crop the image to 224x224!
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("3. Victim is analyzing the CLEAN image...")
    clean_tensor = preprocess(clean_img).unsqueeze(0)
    with torch.no_grad():
        clean_out = victim_model(clean_tensor)
    clean_probs = torch.nn.functional.softmax(clean_out[0], dim=0)
    clean_top_prob, clean_top_catid = torch.topk(clean_probs, 1)

    print("4. Sabotaging the image via your local NoiseManager...")
    buffered = io.BytesIO()
    clean_img.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    
    attacker = NoiseManager()
    noised_b64 = attacker.noise(image_bytes)
    noised_img = Image.open(io.BytesIO(base64.b64decode(noised_b64)))

    print("5. Victim is analyzing the SABOTAGED image...")
    noised_tensor = preprocess(noised_img).unsqueeze(0)
    with torch.no_grad():
        noised_out = victim_model(noised_tensor)
    noised_probs = torch.nn.functional.softmax(noised_out[0], dim=0)
    noised_top_prob, noised_top_catid = torch.topk(noised_probs, 1)

    print("6. Generating damage report...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(transforms.CenterCrop(224)(transforms.Resize(256)(clean_img)))
    clean_label = labels[clean_top_catid[0].item()]
    clean_conf = clean_top_prob[0].item() * 100
    axes[0].set_title(f"Opponent sees: {clean_label}\nConfidence: {clean_conf:.1f}%", color='green')
    axes[0].axis('off')
    
    axes[1].imshow(transforms.CenterCrop(224)(transforms.Resize(256)(noised_img)))
    noised_label = labels[noised_top_catid[0].item()]
    noised_conf = noised_top_prob[0].item() * 100
    axes[1].set_title(f"Opponent sees: {noised_label}\nConfidence: {noised_conf:.1f}%", color='red')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("damage_report.png")
    print("Done! Open damage_report.png to see the results.")

if __name__ == "__main__":
    simulate_attack()