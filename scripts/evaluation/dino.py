import torch
import torchvision.transforms as T
import cv2
import numpy as np
from timm.models import create_model

# Load DINO ViT-S/16
model = create_model('vit_small_patch16_224_dino', pretrained=True)
model.eval()
model.cuda()

# Preprocessing for DINO ViT
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def extract_features(frame, model):
    """Extract feature from a single frame using DINO ViT."""
    with torch.no_grad():
        input_tensor = transform(frame).unsqueeze(0).cuda()
        features = model.forward_features(input_tensor)
        return features['x_norm_clstoken'].squeeze(0).cpu().numpy()

def compute_dino_score(video_path):
    """Compute DINO-based temporal consistency score for a video."""
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
        raise ValueError("Video must have at least 2 frames.")

    # Extract features for all frames
    features = [extract_features(f, model) for f in frames]

    # Compute cosine similarity with first frame
    first_feat = features[0]
    similarities = []
    for i in range(1, len(features)):
        cos_sim = np.dot(first_feat, features[i]) / (
            np.linalg.norm(first_feat) * np.linalg.norm(features[i])
        )
        similarities.append(cos_sim)

    return np.mean(similarities)

# Example usage
video_path = "/home/dai/research/ToonCrafter_FreeInit/results/bleach_frame10_sample0_w:o_FreeInit.mp4"
dino_score = compute_dino_score(video_path)
print(f"DINO Temporal Consistency Score: {dino_score:.4f}")
