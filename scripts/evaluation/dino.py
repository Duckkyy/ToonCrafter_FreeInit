import torch
import torchvision.transforms as T
import cv2
import numpy as np
from timm.models import create_model
from safetensors.torch import load_file
from pathlib import Path

# === CONFIGURATION ===
video_path = "/home/dai/research/ToonCrafter_FreeInit/results/bleach_frame10_sample0_w:o_FreeInit.mp4"
weights_path = "/home/dai/research/ToonCrafter_FreeInit/scripts/evaluation/model.safetensors"  # <-- Replace with actual path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Load DINO ViT-S/16 from .safetensors ===
model = create_model("vit_small_patch16_224", pretrained=False)
state_dict = load_file(weights_path)
model.load_state_dict(state_dict)
model.eval().to(device)

# === Image preprocessing ===
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def extract_features(frame, model):
    """Extract DINO CLS token feature from a single frame."""
    with torch.no_grad():
        input_tensor = transform(frame).unsqueeze(0).to(device)
        features = model.forward_features(input_tensor)
        # Some timm models return a dict, some a tensor
        if isinstance(features, dict) and 'x_norm_clstoken' in features:
            return features['x_norm_clstoken'].squeeze(0).cpu().numpy()
        else:
            return features.squeeze(0).cpu().numpy()


def compute_dino_score(video_path):
    """Compute DINO temporal consistency score for a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) < 2:
        raise ValueError("Need at least 2 frames to compute temporal consistency.")

    # Extract DINO features for all frames
    features = [extract_features(f, model) for f in frames]

    # Compute cosine similarity between frame 0 and each subsequent frame
    first_feat = features[0]
    sims = []
    for i in range(1, len(features)):
        cos_sim = np.dot(first_feat, features[i]) / (
            np.linalg.norm(first_feat) * np.linalg.norm(features[i])
        )
        sims.append(cos_sim)

    return np.mean(sims)


# === Run it ===
score = compute_dino_score(video_path)
print(f"[DINO Temporal Consistency Score] {Path(video_path).name}: {score:.4f}")
