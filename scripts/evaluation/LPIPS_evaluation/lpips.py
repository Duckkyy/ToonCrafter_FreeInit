import cv2
import torch
import lpips
import numpy as np

def prepare_image(image_path):
    """
    Reads an image from the given path, converts it from BGR to RGB,
    normalizes it to the range [-1, 1], and converts it into a PyTorch tensor.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        tensor (torch.Tensor): A tensor of shape (1, C, H, W) ready for LPIPS.
    """
    # Read image with OpenCV (BGR format)
    img = cv2.imread(image_path)
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to [-1, 1]
    img = img.astype(np.float32) / 127.5 - 1.0
    # Convert from HWC to CHW format and add a batch dimension
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return tensor

# Paths to your ground truth and generated images.
gt_image_path = "gt.jpg"     # Replace with your ground truth image path.
gen_image_path = "with_FI.jpg"         # Replace with your generated image path.

# Prepare the images.
gt_tensor = prepare_image(gt_image_path)
gen_tensor = prepare_image(gen_image_path)

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
