import cv2
import numpy as np

def read_image(image_path):
    """
    Reads an image from the specified path and returns it in BGR format.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img

def calculate_mse(img1, img2):
    """
    Calculates the Mean Squared Error between two images.
    
    Both images must have the same dimensions.
    """
    # Convert images to float for precision
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    return mse

def calculate_psnr(gt_img, gen_img, max_pixel=255.0):
    """
    Calculates the PSNR (Peak Signal-to-Noise Ratio) between a ground truth image and a generated image.
    
    Both images must have the same dimensions.
    """
    mse = calculate_mse(gt_img, gen_img)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

# Paths to your ground truth and generated images
gt_image_path = "/home/dai/research/ToonCrafter_FreeInit/scripts/evaluation/LPIPS_evaluation/gt.jpg"    # Replace with your ground truth image path
gen_image_path = "/home/dai/research/ToonCrafter_FreeInit/scripts/evaluation/LPIPS_evaluation/with_FI.jpg"         # Replace with your generated image path

# Read the ground truth image
gt_img = read_image(gt_image_path)

# Read the generated image
gen_img = read_image(gen_image_path)

# Ensure both images have the same dimensions by resizing the generated image if needed
if gt_img.shape != gen_img.shape:
    height, width = gt_img.shape[:2]
    gen_img = cv2.resize(gen_img, (width, height), interpolation=cv2.INTER_AREA)

# Compute PSNR
psnr_value = calculate_psnr(gt_img, gen_img)
print("PSNR between the ground truth and generated image: {:.2f} dB".format(psnr_value))
