import cv2
import numpy as np

def read_video_frames(video_path):
    """
    Reads all frames from a video file.
    
    Parameters:
        video_path (str): Path to the video file.
        
    Returns:
        frames (list): A list of frames (NumPy arrays) read from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def compute_ssim(img1, img2, window_size=11, sigma=1.5, k1=0.01, k2=0.03, L=255):
    """
    Compute the Structural Similarity Index (SSIM) between two grayscale images.
    
    Parameters:
        img1 (ndarray): First grayscale image.
        img2 (ndarray): Second grayscale image.
        window_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for Gaussian kernel.
        k1 (float): Constant for SSIM (default 0.01).
        k2 (float): Constant for SSIM (default 0.03).
        L (int): Dynamic range of the pixel values (default 255 for 8-bit images).
        
    Returns:
        float: The SSIM value.
    """
    # Convert images to float64 for precision
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants for stability
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    
    # Compute means using a Gaussian filter
    mu1 = cv2.GaussianBlur(img1, (window_size, window_size), sigma)
    mu2 = cv2.GaussianBlur(img2, (window_size, window_size), sigma)
    
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (window_size, window_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (window_size, window_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (window_size, window_size), sigma) - mu1_mu2
    
    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def compute_average_ssim(video_frames1, video_frames2):
    """
    Computes the average SSIM between corresponding frames of two videos.
    
    Parameters:
        video_frames1 (list): List of frames for the first video.
        video_frames2 (list): List of frames for the second video.
        
    Returns:
        float: The average SSIM value over all frame pairs.
    """
    ssim_values = []
    for frame1, frame2 in zip(video_frames1, video_frames2):
        # Convert frames to grayscale
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        ssim_val = compute_ssim(frame1_gray, frame2_gray)
        ssim_values.append(ssim_val)
    return np.mean(ssim_values)

# Paths to your video files
video_path1 = "/home/dai/research/ToonCrafter_FreeInit/results/bleach_frame10_sample0_w\:o_FreeInit.mp4"  # Replace with the path to your first video.
video_path2 = "/home/dai/research/ToonCrafter_FreeInit/results/bleach_frame10_sample0.mp4" # Replace with the path to your second video.

# Read frames from both videos
video_frames1 = read_video_frames(video_path1)
video_frames2 = read_video_frames(video_path2)

print("Number of frames in first video:", len(video_frames1))
print("Number of frames in second video:", len(video_frames2))

if len(video_frames1) != len(video_frames2):
    print("Warning: The two videos have a different number of frames!")
else:
    avg_ssim = compute_average_ssim(video_frames1, video_frames2)
    print("Average SSIM between the two videos: {:.4f}".format(avg_ssim))


# Set the paths for the two videos you want to compare.
# video_path1 = "/home/dai/research/ToonCrafter_FreeInit/results/bleach_frame10_sample0_w\:o_FreeInit.mp4"   # Replace with the path to your first generated video.
# video_path2 = "/home/dai/research/ToonCrafter_FreeInit/results/bleach_frame10_sample0.mp4"  # Replace with the path to your second generated video.

