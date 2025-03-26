import cv2
import torch
import lpips
import numpy as np

def prepare_image(image_path, target_shape=None):
    """
    Reads an image, converts it from BGR to RGB, normalizes pixel values to [-1, 1],
    and converts it into a PyTorch tensor. Optionally resizes the image to target_shape.
    
    Parameters:
        image_path (str): Path to the image file.
        target_shape (tuple): Desired output shape as (width, height). If provided, the image is resized.
        
    Returns:
        tensor (torch.Tensor): A tensor of shape (1, C, H, W).
    """
    # Read the image (BGR format)
    img = cv2.imread(image_path)
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image if target_shape is provided
    if target_shape is not None:
        img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [-1, 1]
    img = img.astype(np.float32) / 127.5 - 1.0
    # Convert from HWC to CHW format and add a batch dimension
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return tensor

# Paths to your ground truth and generated images.
gt_image_path = "/home/dai/research/ToonCrafter_FreeInit/scripts/evaluation/LPIPS_evaluation/gt.jpg"     # Replace with your ground truth image path.
gen_image_path = "/home/dai/research/ToonCrafter_FreeInit/scripts/evaluation/LPIPS_evaluation/without_FI.jpg"        # Replace with your generated image path.

# First, read the ground truth image to get its dimensions.
gt_img = cv2.imread(gt_image_path)
gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
height, width, _ = gt_img_rgb.shape
target_shape = (width, height)  # Note: cv2.resize expects (width, height)

# Prepare the images: ensure the generated image is resized to match the ground truth.
gt_tensor = prepare_image(gt_image_path)
gen_tensor = prepare_image(gen_image_path, target_shape=target_shape)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gt_tensor = gt_tensor.to(device)
gen_tensor = gen_tensor.to(device)

# Initialize LPIPS model (choose 'alex' or 'vgg' as the backbone)
loss_fn = lpips.LPIPS(net='alex').to(device)

# Compute LPIPS score (lower score means higher perceptual similarity)
with torch.no_grad():
    lpips_score = loss_fn(gt_tensor, gen_tensor)
    
print("LPIPS score between the images: {:.4f}".format(lpips_score.item()))
