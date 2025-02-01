import os
import cv2
import numpy as np
import glob

print("Starting conversion...")

# Get the directory of the currently running script (absolute path)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative paths based on the script's location
mask_folder = os.path.join(base_dir, "tif_input", "residue_background_tif")
output_folder = os.path.join(base_dir, "txt_output", "residue_background_txt")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all .tif mask files
mask_files = glob.glob(os.path.join(mask_folder, "*.tif"))

print(f"Found {len(mask_files)} .tif files.")
if len(mask_files) == 0:
    print(f"Error: No .tif files found in {mask_folder}")
    exit()

for mask_path in mask_files:
    print(f"Loading mask: {mask_path}")
    
    # Load mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Error: Could not read image {mask_path}")
        continue
    
    # Check unique values in mask
    unique_values = np.unique(mask)
    print(f"Processing {mask_path} - Unique pixel values: {unique_values}")

    # Optional: Thresholding to ensure binary image
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours (use original mask or binary version)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"Warning: No contours found in {mask_path}")
        continue

    # Get image dimensions
    img_h, img_w = mask.shape
    label_filename = os.path.join(output_folder, os.path.basename(mask_path).replace(".tif", ".txt"))

    with open(label_filename, "w") as f:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ensure center coordinates are integers
            center_x = x + w // 2
            center_y = y + h // 2

            # Normalize to YOLO format (x_center, y_center, width, height)
            x_center = center_x / img_w
            y_center = center_y / img_h
            w = w / img_w
            h = h / img_h
            
            # Determine class based on pixel value in the mask
            # Use the center pixel for class determination
            class_id = 0 if mask[center_y, center_x] == 0 else 1
            
            # YOLO format: class_id x_center y_center width height
            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")
    
    print(f"Wrote bounding boxes to: {label_filename}")

print("Conversion to YOLO bounding box format completed!")
