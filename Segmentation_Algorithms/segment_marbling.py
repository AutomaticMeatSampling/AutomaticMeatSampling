import cv2
import numpy as np
import matplotlib.pyplot as plt
import segment_ld
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve()
def rel_path(path):
    filepath = SCRIPT_DIR.parent / path
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    return filepath

def normalize_brightness(img, mask, target_mean=128, tissue_type="Tissue", show=True):
    """
    Description: Normalizes mean brightness of img within the mask to target_mean. Minimizes the affect of glare in images.

    Args:
        img (np.array): Input image
        mask (np.array): Binary mask for LD muscle
        target_mean (int): Target mean 
        
    Returns:
        img_norm (np.array): Brightness normalized image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray[mask > 0])
    adjustment = target_mean - mean_val
    img_norm = cv2.convertScaleAbs(img, alpha=1, beta=adjustment)
    img_norm[mask == 0] = 0  # Keep background as black

    if show:
        # Show before and after normalization
        # Add tissue type in title
        new_mean_val = np.mean(cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)[mask > 0])
        titles = [f"Original {tissue_type} Image\nMean Brightness in Mask: {mean_val:.2f}", f"Normalized {tissue_type} Image\nMean Brightness in Mask: {new_mean_val:.2f}"]
        segment_ld.show_images([img, img_norm], titles, filename=rel_path(f"temp/normalize_brightness-Normalized-Image-in-{tissue_type}.png"))

    return img_norm


def otsu_threshold_and_clean(enhanced_img, enhanced_muscle_img, mask, min_marbling_cc_size=300):
    """
    Description: Uses global otsu thresholding to find mask for marbling. Removes any small connected components with size < min_cc_size.

    Args:
        enhanced (np.array): Enhanced greyscale image
        mask (np.array): Binary mask for LD muscle
        min_cc_size (int): Minimum size for connected components to keep

    Returns:
        new_thresholded_marbling_mask (np.array): Binary mask for marbling
    """
    # Threshold marbling
    _, thresholded_marble_mask = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small flecks with connected Components
    # TODO: Fine tune min_marbling_cc_size for final images with specific robot camera resolution
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded_marble_mask, connectivity=8)
    sizes = stats[1:, -1]
    new_thresholded_marbling_mask = np.zeros_like(thresholded_marble_mask)
    for i, size in enumerate(sizes):
        if size > min_marbling_cc_size:
            new_thresholded_marbling_mask[labels == i + 1] = 255


    # Invert threshold marbling mask to get muscle mask
    _, thresholded_marble_mask2 = cv2.threshold(enhanced_muscle_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    new_thresholded_muscle_mask = cv2.bitwise_and(mask, cv2.bitwise_not(thresholded_marble_mask2))

    cv2.imwrite(rel_path("temp/output_new_thresholded_muscle_mask.png"), new_thresholded_muscle_mask)

    # Expand the black parts (to avoid touching possible marbling points) or shrink white parts with erosion
    # TODO: Fine tune for final images with specific robot camera resolution
    cleaned_muscle_mask = cv2.erode(new_thresholded_muscle_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Erase small pieces of muscle mask:

    return new_thresholded_marbling_mask, cleaned_muscle_mask

def enhanced_contrast_grayscale(img, mask=None, tissue_type="Tissue", show=True):
    """
    Description: Custom grayscale converter that attempts to make marbling stand out (Makes marbling whiter and muscle darker)
    Uses both DoubleGreen and TotalMix methods to enhance contrast.

    Args:
        img (np.array): Input image (colored)
        mask (np.array): Binary mask for LD muscle

    Returns:
        enhanced (np.array): Enhanced greyscale image
    """

    img_float = img.astype(np.float32) / 255.0
    B, G, R = cv2.split(img_float)

    if mask is not None:
        mask_bool = mask > 0
    else:
        mask_bool = np.ones(img.shape[:2], dtype=bool)

    # ------------------ DoubleGreen ------------------
    double_green = 2 * G - (R + B)
    double_green = np.clip(double_green, 0, 1)
    mg_min = double_green[mask_bool].min()
    mg_max = double_green[mask_bool].max()
    double_green_norm = (double_green - mg_min) / max(mg_max - mg_min, 1e-6)
    enhanced_doublegreen = np.clip(3 * double_green_norm, 0, 1)
    enhanced_doublegreen = (enhanced_doublegreen * 255).astype(np.uint8)

    # ------------------ TotalMix ------------------
    total_mix = R * G * B
    total_mix *= 10  # amplify
    total_mix = np.clip(total_mix, 0, 1)
    tm_min = total_mix[mask_bool].min()
    tm_max = total_mix[mask_bool].max()
    total_mix_norm = (total_mix - tm_min) / max(tm_max - tm_min, 1e-6)
    enhanced_totalmix = (total_mix_norm * 255).astype(np.uint8)

    # ------------------ Combine ------------------
    enhanced = ((enhanced_doublegreen.astype(np.float32) + enhanced_totalmix.astype(np.float32)) / 2.0).astype(np.uint8)

    if show:
        # Show enhanced image
        titles = [f"Original Normalized {tissue_type} Image", f"Enhanced Grayscale {tissue_type} Image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segment_ld.show_images([img, enhanced], titles, filename=rel_path("temp/enhanced_contrast_grayscale.png"))

    return enhanced

def remove_background(image, ld_mask):
    """
    Description: Uses the ld_mask to remove background from image
    """

    result = cv2.bitwise_and(image, image, mask=ld_mask)

    return result

def segment_tissue(image_path, ld_mask, show=True):
    """
    Description: Finds marbling in the image in the ld muscle region. Uses image processing algorithms (brightness normalization, contrast enhancement, otsu thresholding, connected components)
    to create marbling mask.

    Args:
        image_path (str): Path to the input image (colored)
        ld_mask (np.array): Binary mask for LD muscle

    Returns:
        dict: Dictionary containing intermediate results and final marbling mask
    """
    target_mean_brightness = 25 # TODO: FINE TUNE with real lighting situation
    target_mean_brightness_muscle = 40 # TODO: FINE TUNE with real lighting situation
    # Step 1: Read image
    image = cv2.imread(image_path)

    # Step 2: Pre-process LD mask
    print("Step 7: Cleaning LD mask and shrinking to avoid intramuscular fat...")
    ld_mask = segment_ld.post_processing_ld_mask(ld_mask, show=show)

    # Step 3: Remove background using LD mask
    no_bkg = remove_background(image, ld_mask)

    # Step 4: Normalize brightness
    print("Step 8: Normalizing brightness to reduce glare effects...")
    print("     Step 8a: Normalizing for marbling detection...")
    img_norm = normalize_brightness(no_bkg, ld_mask, target_mean_brightness, tissue_type="Marbling", show=show)
    print("     Step 8b: Normalizing for muscle detection...")
    img_norm_muscle = normalize_brightness(no_bkg, ld_mask, target_mean_brightness_muscle, tissue_type="Muscle", show=show)

    # Step 5: Convert to greyscale and enhance contrast
    print("Step 9: Converting to grayscale with enhanced contrast to highlight marbling and muscle...")
    print("     Step 9a: Enhancing for marbling detection...")
    enhanced_marbling = enhanced_contrast_grayscale(img_norm, ld_mask, tissue_type="Marbling", show=show)
    print("     Step 9b: Enhancing for muscle detection...")
    enhanced_muscle = enhanced_contrast_grayscale(img_norm_muscle, ld_mask, tissue_type="Muscle", show=show)

    # Step 6: Otsu thresholding and clean small connected components
    print("Step 10: Applying Otsu thresholding and cleaning small connected components to find marbling and muscle masks...")
    new_mask_marbling, new_mask_muscle = otsu_threshold_and_clean(enhanced_img=enhanced_marbling, enhanced_muscle_img=enhanced_muscle, mask=ld_mask)

    if show:
        # Show final marbling and muscle masks compared to original
        no_bkg = cv2.cvtColor(no_bkg, cv2.COLOR_BGR2RGB)
        segment_ld.show_images([no_bkg, new_mask_marbling, new_mask_muscle], ["Original Image", "Final Marbling Mask", "Final Muscle Mask"], filename=rel_path("temp/tissue-semgenation-output.png"))
    
    return {
        "no_bkg": no_bkg,
        "enhanced_marbling": enhanced_marbling,
        "enhanced_muscle": enhanced_muscle,
        "final_marbling": new_mask_marbling,
        "final_muscle": new_mask_muscle,
        "img_norm": img_norm,
        "cleaned_ld_mask": ld_mask
    }