import numpy as np
import cv2
from PIL import Image

PVD_BOUNDARIES = [7, 15, 31, 63, 127]

def test_ppdh(image_path):
    # Try PIL RGB
    img_pil = np.array(Image.open(image_path).convert("RGB"))
    print("--- PIL RGB ---")
    for i, name in enumerate(["R", "G", "B"]):
        channel = img_pil[:, :, i].flatten()
        score = calculate_score(channel)
        print(f"Channel {name} score: {score}")

    # Try CV2 BGR
    img_cv2 = cv2.imread(image_path)
    print("\n--- CV2 BGR ---")
    for i, name in enumerate(["B", "G", "R"]):
        channel = img_cv2[:, :, i].flatten()
        score = calculate_score(channel)
        print(f"Channel {name} score: {score}")

def calculate_score(channel):
    p1 = channel[0::2].astype(int)
    p2 = channel[1::2].astype(int)
    min_len = min(len(p1), len(p2))
    diffs = np.abs(p2[:min_len] - p1[:min_len])
    counts = np.bincount(diffs, minlength=256).astype(float)
    threshold = np.std(counts) * 2
    
    score = 0
    for k in PVD_BOUNDARIES:
        if k < 1 or k >= 254: continue
        second_derivative = float(counts[k + 1] - 2 * counts[k] + counts[k - 1])
        if abs(second_derivative) > threshold:
            score += 1
    return score

if __name__ == "__main__":
    path = r"c:\Users\dhruv\Desktop\stego2\data\combined\test\pvd\04001x.png"
    test_ppdh(path)
