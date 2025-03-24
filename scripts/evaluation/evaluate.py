import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def read_video_frames(video_path):
    """
    Reads all frames from a video file.
    
    Parameters:
        video_path (str): Path to the video file.
        
    Returns:
        frames (list): A list of frames read from the video.
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

def compute_average_ssim(video_frames1, video_frames2):
    """
    Computes the average SSIM between corresponding frames of two videos.
    
    Assumes both videos have the same number of frames.
    
    Parameters:
        video_frames1 (list): List of frames (as NumPy arrays) for the first video.
        video_frames2 (list): List of frames (as NumPy arrays) for the second video.
        
    Returns:
        average_ssim (float): The average SSIM value over all frame pairs.
    """
    ssim_values = []
    for frame1, frame2 in zip(video_frames1, video_frames2):
        # Convert frames to grayscale, which is common for SSIM evaluation.
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        ssim_value, _ = ssim(frame1_gray, frame2_gray, full=True)
        ssim_values.append(ssim_value)
    
    return np.mean(ssim_values)

# Set the paths for the two videos you want to compare.
video_path1 = "/home/dai/research/ToonCrafter_FreeInit/results/bleach_frame10_sample0_w\:o_FreeInit.mp4"   # Replace with the path to your first generated video.
video_path2 = "/home/dai/research/ToonCrafter_FreeInit/results/bleach_frame10_sample0.mp4"  # Replace with the path to your second generated video.

# Read video frames.
video_frames1 = read_video_frames(video_path1)
video_frames2 = read_video_frames(video_path2)

print("Number of frames in first video:", len(video_frames1))
print("Number of frames in second video:", len(video_frames2))

# Ensure both videos have the same number of frames.
if len(video_frames1) != len(video_frames2):
    print("Warning: The two videos have a different number of frames!")
else:
    # Compute and print the average SSIM.
    avg_ssim = compute_average_ssim(video_frames1, video_frames2)
    print("Average SSIM between the two videos: {:.4f}".format(avg_ssim))
