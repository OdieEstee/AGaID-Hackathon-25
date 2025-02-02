import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from sklearn.model_selection import train_test_split
from shutil import copy
import time

# For training with ultralytics YOLO (make sure to install ultralytics)
from ultralytics import YOLO

# Set your base directory (adjust if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
DEVICE = 'cuda' if USE_CUDA else 'cpu'
# Define class names (for visualization and dataset YAML)
CLASS_NAMES = {
    0: "Residue Sunlit",
    1: "Residue Shaded",
    2: "Background Sunlit",
    3: "Background Shaded"
}
CLASS_LIST = [CLASS_NAMES[i] for i in range(4)]  # for dataset YAML

def load_image_cuda(image_path):
    if USE_CUDA:
        return cv2.cuda.GpuMat(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def threshold_cuda(image_gpu):
    if USE_CUDA:
        return cv2.cuda.threshold(image_gpu, 127, 255, cv2.THRESH_BINARY)[1]
    return cv2.threshold(image_gpu, 127, 255, cv2.THRESH_BINARY)[1]

def find_contours_cuda(image_gpu):
    if USE_CUDA:
        image_cpu = image_gpu.download()
    else:
        image_cpu = image_gpu
    contours, _ = cv2.findContours(image_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --- Helper Functions for File Renaming ---
def add_res_suffix(directory):
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if name.endswith('_res'):
            continue
        new_filename = name + '_res' + ext
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")

def add_sunshade_suffix(directory):
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if name.endswith('_sunshade'):
            continue
        new_filename = name + '_sunshade' + ext
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")

# --- Visualization Function ---
def show_image_feedback(image_path, annotations):
    """
    Display the image with bounding box annotations.
    Each annotation is a tuple: (class_id, x, y, w, h) in pixel coordinates.
    Only residue labels (class 0 and 1) are annotated with text.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image for feedback:", image_path)
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_rgb)

    for ann in annotations:
        cls, x, y, w, h = ann
        # Draw the bounding box.
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        # Only display text for residue classes (0 and 1).
        if cls in (0, 1):
            ax.text(x, y, CLASS_NAMES.get(cls, str(cls)), color='white', fontsize=9,
                    bbox=dict(facecolor='red', alpha=0.5))

    ax.axis("off")
    plt.show(block=False)
    plt.pause(1)
    plt.close(fig)

# --- Conversion Functions ---
def convert_to_yolo_format_combined():
    """
    Convert paired residue and sunlit masks into a single YOLO-format label file for each image.
    This function merges the annotations from residue and sunlit masks into one file per image.
    """
    residue_dir = os.path.join(BASE_DIR, "label/residue_background")
    sunlit_dir = os.path.join(BASE_DIR, "label/sunlit_shaded")
    image_dir = os.path.join(BASE_DIR, "images/train")
    yolo_labels_dir = os.path.join(BASE_DIR, "labels")  # temporary labels folder

    # Rename files for consistency
    add_res_suffix(residue_dir)
    add_sunshade_suffix(sunlit_dir)

    if not os.path.exists(yolo_labels_dir):
        os.makedirs(yolo_labels_dir)

    all_samples = []
    processed = 0
    missing_images = 0
    missing_sunlit = 0
    invalid_masks = 0

    processed_basenames = set()

    # Process residue mask files
    residue_files = glob.glob(os.path.join(residue_dir, '**/*_res.tif'), recursive=True)
    print(f"Found {len(residue_files)} residue files")

    for tif_file in residue_files:
        print(f"\nProcessing residue mask: {tif_file}")
        rel_path = os.path.relpath(tif_file, residue_dir)
        base_name = os.path.basename(tif_file).replace('_res.tif', '')
        processed_basenames.add(base_name)

        # Construct corresponding sunlit mask and image paths.
        extra = rel_path.split('_res.tif')[1] if '_res.tif' in rel_path else ''
        sunlit_path = os.path.join(sunlit_dir, base_name.replace('_res', '_sunshad') + extra + '_sunshade.tif')
        image_path = os.path.join(image_dir, rel_path.replace('_res.tif', '.jpg').replace('_res', ''))

        print(f"Sunlit path: {sunlit_path}")
        print(f"Image path: {image_path}")

        if not os.path.exists(image_path):
            print(f"🚨 Missing image: {image_path}")
            missing_images += 1
            continue
        if not os.path.exists(sunlit_path):
            print(f"🚨 Missing sunlit mask: {sunlit_path}")
            missing_sunlit += 1
            continue

        residue_mask = load_image_cuda(tif_file)
        sunlit_mask = load_image_cuda(sunlit_path)
        if residue_mask is None:
            print(f"💀 Failed to read residue mask: {tif_file}")
            invalid_masks += 1
            continue
        if sunlit_mask is None:
            print(f"💀 Failed to read sunlit mask: {sunlit_path}")
            invalid_masks += 1
            continue

        # Threshold masks (assumes >127 as positive)
        residue_bin = threshold_cuda(residue_mask)
        sunlit_bin = threshold_cuda(sunlit_mask)

        # Generate annotations for residue areas:
        annotations = []
        residue_contours = find_contours_cuda(residue_bin)
        for cnt in residue_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            region = sunlit_bin[y:y+h, x:x+w]
            lit_ratio = np.count_nonzero(region) / (w * h)
            class_id = 0 if lit_ratio > 0.5 else 1  # 0: Residue Sunlit, 1: Residue Shaded
            annotations.append((class_id, x, y, w, h))

        # Generate annotations for background areas using the inverse of residue mask:
        background_bin = cv2.cuda.bitwise_not(residue_bin) if USE_CUDA else cv2.bitwise_not(residue_bin)
        background_contours = find_contours_cuda(background_bin)
        for cnt in background_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            region = sunlit_bin[y:y+h, x:x+w]
            lit_ratio = np.count_nonzero(region) / (w * h)
            class_id = 2 if lit_ratio > 0.5 else 3  # 2: Background Sunlit, 3: Background Shaded
            annotations.append((class_id, x, y, w, h))

        # Convert annotations to normalized YOLO format using image dimensions.
        img = cv2.imread(image_path)
        if img is None:
            print("💀 Could not load image:", image_path)
            continue
        img_height, img_width = img.shape[:2]
        yolo_lines = []
        for ann in annotations:
            cls, x, y, w, h = ann
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if yolo_lines:
            # Save all annotations into a single file in the temporary labels folder.
            yolo_txt_path = os.path.join(yolo_labels_dir, base_name.replace('_res', '') + '.txt')
            with open(yolo_txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            print("Saved YOLO annotation:", yolo_txt_path)
        else:
            print("No annotations generated for:", tif_file)

        # Show feedback
        #show_image_feedback(image_path, annotations)
        processed += 1
        all_samples.append((image_path, yolo_txt_path))

    print("\nResidue-based conversion complete.")
    print(f"Processed residue files: {processed}")
    print(f"Missing images: {missing_images}, Missing sunlit masks: {missing_sunlit}, Invalid masks: {invalid_masks}")

    return all_samples, processed_basenames

def convert_sunlit_shaded_only(processed_basenames):
    sunlit_dir = os.path.join(BASE_DIR, "label/sunlit_shaded")
    image_dir = os.path.join(BASE_DIR, "images/train")  # Or images/val as needed
    yolo_labels_dir = os.path.join(BASE_DIR, "labels", "train")  # Correct path!

    extra_samples = []
    sunlit_files = glob.glob(os.path.join(sunlit_dir, '**/*_sunshade.tif'), recursive=True)
    print(f"Found {len(sunlit_files)} sunlit_shaded files")

    for tif_file in sunlit_files:
        base_name = os.path.basename(tif_file).replace('_sunshade.tif', '')
        if base_name in processed_basenames:
            continue

        print(f"\nProcessing sunlit-only mask: {tif_file}")
        image_path = os.path.join(image_dir, base_name.replace('_sunshad', '') + '.jpg')
        if not os.path.exists(image_path):
            print(f"🚨 Missing image: {image_path}")
            continue

        sunlit_mask = cv2.imread(tif_file, cv2.IMREAD_GRAYSCALE)
        if sunlit_mask is None:
            print(f"💀 Failed to read sunlit mask: {tif_file}")
            continue

        _, sunlit_bin = cv2.threshold(sunlit_mask, 127, 255, cv2.THRESH_BINARY)
        annotations = []
        sunlit_contours, _ = cv2.findContours(sunlit_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sunlit_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            annotations.append((2, x, y, w, h))  # Background Sunlit (class 2)
        shaded_bin = cv2.bitwise_not(sunlit_bin)
        shaded_contours, _ = cv2.findContours(shaded_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in shaded_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            annotations.append((3, x, y, w, h))  # Background Shaded (class 3)

        img = cv2.imread(image_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        yolo_lines = []
        for ann in annotations:
            cls, x, y, w, h = ann
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if yolo_lines:
            os.makedirs(yolo_labels_dir, exist_ok=True)  # Create the directory!
            yolo_txt_path = os.path.join(yolo_labels_dir, base_name.replace('_sunshad', '') + '.txt')
            with open(yolo_txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            print("Saved YOLO annotation for sunlit-only file:", yolo_txt_path)
            #show_image_feedback(image_path, annotations)
            extra_samples.append((image_path, yolo_txt_path))

    return extra_samples

# --- Function to Split Dataset and Organize into Train/Val Folders ---
def split_dataset(all_samples, test_size=0.2):
    from sklearn.model_selection import train_test_split
    # Split sample tuples (image_path, label_path)
    train_samples, val_samples = train_test_split(all_samples, test_size=test_size, random_state=42)

    # Create new directories for train and val splits for both images and labels.
    train_images_dir = os.path.join(BASE_DIR, "images", "train")
    val_images_dir = os.path.join(BASE_DIR, "images", "val")
    train_labels_dir = os.path.join(BASE_DIR, "labels", "train")  # Corrected path
    val_labels_dir = os.path.join(BASE_DIR, "labels", "val")  # Corrected path
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Move files for training samples
    for img_path, label_path in train_samples:
        img_filename = os.path.basename(img_path)
        label_filename = os.path.basename(label_path)
        new_img_path = os.path.join(train_images_dir, img_filename)
        new_label_path = os.path.join(train_labels_dir, label_filename)
        if img_path != new_img_path:
            copy(img_path, new_img_path)
        if label_path != new_label_path:
            copy(label_path, new_label_path)

    # Move files for validation samples
    for img_path, label_path in val_samples:
        img_filename = os.path.basename(img_path)
        label_filename = os.path.basename(label_path)
        new_img_path = os.path.join(val_images_dir, img_filename)
        new_label_path = os.path.join(val_labels_dir, label_filename)
        if img_path != new_img_path:
            copy(img_path, new_img_path)
        if label_path != new_label_path:
            copy(label_path, new_label_path)

    # Return the new split directories for use in YAML preparation.
    return train_images_dir, val_images_dir, train_labels_dir, val_labels_dir # Return label dirs too

# --- Dataset YAML Preparation ---
def prepare_dataset_yaml(samples, yaml_path=os.path.join(BASE_DIR, "dataset.yaml")):
    dataset_dict = {
        'train': os.path.join(BASE_DIR, "images", "train"),
        'val': os.path.join(BASE_DIR, "images", "val"),
        'names': CLASS_LIST
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_dict, f)
    print("Dataset YAML saved to", yaml_path)
    return yaml_path

# --- Training Function ---
def train_yolov9():
    """
    Train YOLOv9 model using the prepared dataset YAML.
    Adjust epochs, image size, and batch size as needed.
    """
    # Path to your pretrained weights or model configuration.
    pretrained_weights = os.path.join(BASE_DIR, "pretrained", "yolov9c.pt")
    dataset_yaml = os.path.join(BASE_DIR, "dataset.yaml")

    model = YOLO(pretrained_weights)
    results = model.train(
        data=dataset_yaml,
        epochs=5,
        imgsz=512,
        batch=16,
        device=DEVICE,
        project=os.path.join(BASE_DIR, "results"),
        name="yolov9_4class"
    )

    # Print out some training metrics if available.
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    print("\nTraining complete. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return results

# --- Main Function ---
def main():
    start_time = time.time()
    # Create labels directories *before* conversion
    os.makedirs(os.path.join(BASE_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "labels", "val"), exist_ok=True)

    samples1, processed_basenames = convert_to_yolo_format_combined()
    samples2 = convert_sunlit_shaded_only(processed_basenames)

    all_samples = samples1 + samples2
    print(f"\nTotal samples converted: {len(all_samples)}")
    print(f"\nTotal time elapsed for sample conversion: {(time.time() - start_time) / 60:.2f} minutes")

    # Split the dataset and organize images and labels into train/val directories.
    train_images_dir, val_images_dir, train_labels_dir, val_labels_dir = split_dataset(all_samples, test_size=0.2)

    # Prepare the dataset YAML file for training.
    prepare_dataset_yaml(all_samples)

    # Finally, start training.
    print("Starting YOLOv9 training for 4-class detection...")
    train_yolov9()
    print("Training complete.")
    print(f"\nTotal time elapsed for sample conversion and training: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()