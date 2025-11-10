import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import time
from sklearn.cluster import KMeans
import sys


def show_mask(mask, ax, random_color=False):
    """
    Description: Displays "mask" on axis "ax" with possible random color

    Args:
        mask (np.array): Binary mask to display
        ax (matplotlib axis): Axis to display the mask on
        random_color (bool): Whether to use a random color for the mask
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    """
    Description: Displays input coords on axis "ax" with green and red color for positive and negative labels respectively
    Args:
        coords (np.array): Array of (x, y) coordinates to display
        labels (np.array): Array of labels (1 for positive, 0 for negative)
        ax (matplotlib axis): Axis to display the points on
        marker_size (int): Size of the markers
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def test_prompt(path="images/sample6.jpg", checkpoint="sam_vit_b_01ec64.pth", foreground_input_points=None, background_input_points=None, show=True, min_size=0):
    """
    Description: Generates masks based on user input points (foreground vs background) using the Segment Anything Model (SAM)

    Args:
        path (str): Path to the input image
        checkpoint (str): Path to the SAM model checkpoint
        foreground_input_points (np.array): Array of (x, y) coordinates for foreground points
        background_input_points (np.array): Array of (x, y) coordinates for background points
        show (bool): Whether to display the generated masks
    """

    # print("Step 2: Read images and convert to RGB")
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Step 2: Load the Segment Anything Model (SAM) Checkpoint")
    if ("default" in checkpoint or "vit_h" in checkpoint):
        model_type = "vit_h"
    elif ("vit_l" in checkpoint):
        model_type = "vit_l"
    elif ("vit_b" in checkpoint):
        model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    if foreground_input_points is None:
        input_point = np.array([[len(image[0])//2, len(image)//2]])
        input_label = np.array([1]) # Label 1 = foreground, 0 = background
    else:
        input_point = foreground_input_points
        input_label = np.ones(len(foreground_input_points), dtype=np.int32)

    if background_input_points is not None:
        input_point = np.concatenate((input_point, background_input_points), axis=0)
        input_label = np.concatenate((input_label, np.zeros(len(background_input_points), dtype=np.int32)), axis=0)

    print("Step 3: Creating Predictor Object...")
    predictor = SamPredictor(sam)

    print("Step 4: Setting image in predictor...")
    predictor.set_image(image)

    print("Step 5: Generating masks...")
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    


    # Save best mask as black mask and white background in images/masks/ folder as last part of path befoe "/" + "_output.png"
    # Option 1: Best = smallest mask size that is bigger than min_size
    # best_mask_index = np.argmin([np.sum(mask) for mask in masks])
    # print("Best mask index: ", best_mask_index)
    # best_mask = masks[best_mask_index].astype(np.uint8) * 255
    # print(masks[best_mask_index].shape)

    print("Step 6: Selecting best mask: Smallest mask that is at least half of the total meat area")
    valid_masks = [(i, mask) for i, mask in enumerate(masks) if np.sum(mask) >= min_size]
    if not valid_masks:
        print("No valid masks found with size >= min_size. Using largest mask instead.")
        best_mask_index = np.argmin([np.sum(mask >= 1) for mask in masks])
        best_mask = masks[best_mask_index].astype(np.uint8) * 255
    else:
        best_mask_index = np.argmin([np.sum(mask >=1 ) for idx, mask in valid_masks])
        best_mask = valid_masks[best_mask_index][1].astype(np.uint8) * 255
    
    output_path = path.split("/")[-1].split(".")[0] + "_output.png"
    cv2.imwrite("images/masks/" + output_path, best_mask)

    # Save all masks to temp/output_ld_mask_{i}.png where i is the rank (1 = best, 2 = second best, 3 = third best)
    for i, mask in enumerate(masks):
        ranked_mask = mask.astype(np.uint8) * 255
        cv2.imwrite(f"temp/output_ld_mask_{i+1}.png", ranked_mask)
        # print(f"Saved mask {i+1} with Score: {scores[i]:.3f}, # of Pixels: {np.sum(mask)}")

    # Option 2: Set best mask to mask with highest score
    # best_mask_index = np.argmax(scores)
    # best_mask = masks[best_mask_index].astype(np.uint8) * 255
    # Overwrite previous best mask
    # cv2.imwrite("images/masks/" + "score_based_" + output_path, best_mask)


    if show:
        # print("Step 6: Showing masks")
        # Show in one figure in subplots all masks with scores and number of pixels
        # Highlight best mask title in red
        plt.figure(figsize=(10,10))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.subplot(1, len(masks), i+1)
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}\nScore: {score:.3f}\n# Pixels: {np.sum(mask)}", color='red' if i == best_mask_index else 'black')
            plt.axis('off')
        plt.show()

    return (best_mask)


# Generate masks for entire image:
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def test_automated_generator(path="images/sample1.jpg"):
    print("Step 1: Read images and convert to RGB")
    image = cv2.imread("images/sample1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Step 2: Load the model")
    sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Step 3: Generate masks")
    masks = mask_generator.generate(image)

    print("Step 4: Show masks")
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

def test_segment_tissue(path="images/sample1.jpg", image=None, meat_mask=None, show=True):
    print("Step 1: Finding center point of ribeye section to find Longissimus Dorsi muscle...")
    if image is None:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("     Step 1a: Convert to grayscale")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    print("     Step 1b: Apply Gaussian Blur to reduce noise and improve thresholding")
    # apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)


    if meat_mask is None:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Show this imshow:
        plt.figure(figsize=(10, 10))
        plt.imshow(thresh, cmap='gray')
        plt.title("Otsu Thresholded Image without Meat Mask")
        plt.axis('off')
        plt.show()
    else:
        # Ensure mask is binary
        print("     Step 1c: Calculate foreground points using Otsu thresholding")
        meat_mask = cv2.threshold(meat_mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Apply Otsu thresholding to the mask
        pixels = blurred[meat_mask == 255]
        masked_pixels_image = pixels.reshape(-1, 1).astype(np.uint8)

        if len(pixels) == 0:
            print("No pixels found in the mask.")
        
        ret, otsu_thresh = cv2.threshold(masked_pixels_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply thrshold globally but only keep maskd region
        thresh = np.zeros_like(gray, dtype=np.uint8)
        thresh[(blurred < ret) & (meat_mask == 255)] = 255

    # Create a kernel (you can tweak the size)
    kernel = np.ones((3, 3), np.uint8)

    # Optional: first clean with erosion, then expand back with dilation
    eroded = cv2.erode(thresh, kernel, iterations=1)
    cleaned = cv2.dilate(eroded, kernel, iterations=1)

    # Find contours (boundary tracking)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and take the largest (assumed ribeye)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        print("No contours found.")
        return image

    # Create a mask from the largest contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contours[0]], -1, 255, thickness=cv2.FILLED)

    foreground_M = cv2.moments(contours[0])
    if foreground_M["m00"] != 0:
        fx = int(foreground_M["m10"] / foreground_M["m00"])
        fy = int(foreground_M["m01"] / foreground_M["m00"])
    else:
        fx, fy = 0, 0

    background_M = cv2.moments(contours[1])
    if background_M["m00"] != 0:
        bx = int(background_M["m10"] / background_M["m00"])
        by = int(background_M["m01"] / background_M["m00"])
    else:
        bx, by = 0, 0

    # Use the mask to extract the ribeye section
    segmented = cv2.bitwise_and(image, image, mask=mask)

    # Display otsu beside the original image and the segmented image
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title("Otsu Thresholded Image")
    plt.axis('off')
    # plt.subplot(1, 4, 3)
    # plt.imshow(cleaned, cmap='gray')
    # plt.title("Cleaned Image")
    # plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(segmented)
    plt.scatter([fx], [fy], color='green', s=40, marker='*', label='Foreground')
    plt.scatter([bx], [by], color='green', s=40, marker='x', label='Background')
    plt.title("Segmented Meat with Selected Point")
    plt.axis('off')

    if show:
        plt.show()

    # Calculate number of pixels in thresh
    num_foreground_pixels = np.sum(thresh == 255)

    return np.array([[fx, fy]]), np.array([[bx, by]]), num_foreground_pixels


def remove_green_background(image_path="images/sample7.jpg", show=True):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    original = image.copy()


    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define green range in HSV
    lower_green = np.array([35, 40, 40])   # light green
    upper_green = np.array([85, 255, 255]) # dark green

    # Create mask for green regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Invert the mask to get non-green areas (meat)
    meat_mask = cv2.bitwise_not(green_mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(original, original, mask=meat_mask)

    # Display the result
    if show:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(meat_mask, cmap='gray')
        plt.title("Removed Green Background")
        plt.axis('off')
        plt.show()
    
    return result, meat_mask, green_mask
    
def calculate_background_points(green_mask, show=True):
    # print("Selecting background point 1...")
    distance_map = cv2.distanceTransform(green_mask, distanceType=cv2.DIST_L2, maskSize=5)

    # Get coordinates of the two farthest background pixels
    flat_indices = np.argpartition(distance_map.ravel(), -2)[-2:]  # get indices of top 2 distances
    y_coords, x_coords = np.unravel_index(flat_indices, distance_map.shape)

    # Create background points (x, y)
    background_input_point = np.stack((x_coords, y_coords), axis=1)

    return background_input_point




# ------------------------------------------- Marbling Segmentation Functions -------------------------------------------


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
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original {tissue_type} Image\nMean Brightness in Mask: {mean_val:.2f}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB))
        new_mean_val = np.mean(cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)[mask > 0])
        plt.title(f"Normalized {tissue_type} Image\nMean Brightness in Mask: {new_mean_val:.2f}")
        plt.show()

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

    cv2.imwrite("temp/output_new_thresholded_muscle_mask.png", new_thresholded_muscle_mask)

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
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Normalized {tissue_type} Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(enhanced, cmap='gray')
        plt.title(f"Enhanced Grayscale {tissue_type} Image")
        plt.axis('off')
        plt.show()

    return enhanced

def fill_holes(binary_mask):
    """
    Description: Fills holes in a binary mask using flood fill algorithm.
    Args:
        binary_mask (np.array): Binary mask to fill holes in

    Returns:
        out (np.array): Binary mask with holes filled
    """
    # Make a copy for flood filling
    h, w = binary_mask.shape
    mask = np.zeros((h+2, w+2), np.uint8)

    # Flood fill from top-left corner (background)
    filled = binary_mask.copy()
    cv2.floodFill(filled, mask, (0, 0), 255)

    # Invert floodfilled image
    floodfill_inv = cv2.bitwise_not(filled)

    # Combine original mask with inverted floodfill to fill holes
    out = binary_mask | floodfill_inv
    return out

def remove_background(image, ld_mask):
    """
    Description: Uses the ld_mask to remove background from image
    """

    result = cv2.bitwise_and(image, image, mask=ld_mask)

    return result

def find_marbling(image_path, ld_mask, show=True):
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
    ld_mask = pre_processing_ld_mask(ld_mask, show=show)

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
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(no_bkg, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(new_mask_marbling, cmap='gray')
        plt.title("Final Marbling Mask")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(new_mask_muscle, cmap='gray')
        plt.title("Final Muscle Mask")
        plt.axis('off')
        plt.show()
    

    return {
        "no_bkg": no_bkg,
        "enhanced_marbling": enhanced_marbling,
        "enhanced_muscle": enhanced_muscle,
        "final_marbling": new_mask_marbling,
        "final_muscle": new_mask_muscle,
        "img_norm": img_norm,
        "cleaned_ld_mask": ld_mask
    }

def pre_processing_ld_mask(ld_mask, min_size=5000, show=True):
    """
    Description: Removes small connected components from the LD mask
    Args:
        ld_mask (np.array): Binary mask for LD muscle
        min_size (int): Minimum size for connected components to keep

    Returns:
        ld_mask (np.array): Cleaned binary mask for LD muscle
    """
    # Fill in any holes, i.e. where chunks of black are surrounded by white
    ld_mask_new = fill_holes(ld_mask)

    cv2.imwrite("temp/output_filled_ld_mask1.png", ld_mask_new)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ld_mask_new, connectivity=8)
    sizes = stats[1:, -1]
    cleaned_mask = np.zeros_like(ld_mask_new)
    for i, size in enumerate(sizes):
        if size > min_size:
            cleaned_mask[labels == i + 1] = 255
    ld_mask_new = cleaned_mask

    cv2.imwrite("temp/output_cleaned_ld_mask2.png", ld_mask_new)

    # Keep only the biggest area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ld_mask_new, connectivity=8)
    sizes = stats[1:, -1]
    if len(sizes) > 0:
        largest_cc_index = np.argmax(sizes)
        ld_mask_new = np.zeros_like(ld_mask_new)
        ld_mask_new[labels == largest_cc_index + 1] = 255

    cv2.imwrite("temp/output_largest_ld_mask3.png", ld_mask_new)

    # shrink the overall ld_mask to avoid getting any of the fat
    ld_mask_new = cv2.erode(ld_mask_new, np.ones((10, 10), np.uint8), iterations=14)

    cv2.imwrite("temp/output_shrunk_ld_mask4.png", ld_mask_new)

    if show:
        # Show before and after
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(ld_mask, cmap='gray')
        plt.title("Original LD Mask")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(ld_mask_new, cmap='gray')
        plt.title("Cleaned LD Mask")
        plt.axis('off')
        plt.show()
    

    return ld_mask_new


if __name__ == "__main__":
    print("test")
    type = "real"
    sample_num = 2
    sample_side = "a" # or "" for fake images
    light_type = "both" # or "green" for fake images

    # Parse possible command line argument was give for sample_id="1a"
    if len(sys.argv) > 2:
        sample_num = sys.argv[1]
        sample_side = sys.argv[2]



    if type == "fake":
        path = f"images/samples_green/sample{sample_num}_green.jpg"
        ld_mask_path = f"images/masks/sample{sample_num}_green_output.png"
    else:
        path = f"real_images/sample{sample_num}{sample_side}_{light_type}.png"
        ld_mask_path = f"images/masks/sample{sample_num}{sample_side}_{light_type}_output.png"
    
    marbling_only = False
    show = True
    # test_prompt(path)
    # test_segment_tissue(path)
    # test_automated_generator(path)

    if not marbling_only:
    
        start = time.time()
        print("Step 0: Reading image and removing green background...")
        result, meat_mask, green_mask = remove_green_background(path, show=False)
        # Save meat_mask to temp/output_meat_mask.png
        cv2.imwrite(f"temp/output_meat_mask.png", meat_mask)
        # Set all parts of orig image where meat_mask is 0 to white - but rn it shows up as blue instead of red fix it
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"temp/output_no_green.png", result)

        # print("Remove green background time taken: ", time.time() - start)

        start = time.time()
        background_input_points1 = calculate_background_points(green_mask, show=show)
        foreground_midpoint, background_midpoint, thresh_mask_size = test_segment_tissue(path=path, meat_mask=meat_mask, show=show)
        print("     Finding center point of ribeye time taken: ", time.time() - start)

        bp = np.concatenate((background_input_points1, background_midpoint), axis=0)

        checkpoint_option = "L"  # Choose from "B", "H", "L"
        if checkpoint_option not in ["B", "H", "L"]:
            raise ValueError("Invalid checkpoint option. Choose from 'B', 'H', 'L'.")
        if checkpoint_option == "B":
            checkpoint_file = "sam_vit_b_01ec64.pth"
        elif checkpoint_option == "H":
            checkpoint_file = "sam_vit_h_4b8939.pth"
        else:
            # Lightest model
            checkpoint_file = "sam_vit_l_0b3195.pth"

        start = time.time()
        best_mask_ld = test_prompt(path, checkpoint_file, foreground_input_points=foreground_midpoint, background_input_points=bp, show=show, min_size = thresh_mask_size//2)
        # print(f"Checkpoint {checkpoint_option} Time taken: ", time.time() - start)

    else:
        best_mask_ld = cv2.imread(ld_mask_path, cv2.IMREAD_GRAYSCALE)
        if best_mask_ld is None:
            raise ValueError(f"Could not read mask image at {ld_mask_path}")


    # Use mask and original image to separate marbling fat (white) from muscle (red)
    start = time.time()
    results = find_marbling(path, best_mask_ld)
    for key, value in results.items():
        cv2.imwrite(f"temp/output_{key}.png", value)
    
    final_marbling = results["final_marbling"]
    # Write to images/masks/sample{sample_num}_marbling.png
    cv2.imwrite(f"images/masks/sample{sample_num}{sample_side}_{light_type}_marbling_mask.png", final_marbling)
    
    final_muscle = cv2.bitwise_and(best_mask_ld, cv2.bitwise_not(final_marbling))
    cv2.imwrite(f"images/masks/sample{sample_num}{sample_side}_{light_type}_muscle_mask.png", final_muscle)

    cv2.imwrite(f"images/masks/sample{sample_num}{sample_side}_{light_type}_muscle_mask_2.png", results["final_muscle"])

    cleaned_ld_mask = results["cleaned_ld_mask"]
    cv2.imwrite(f"images/masks/sample{sample_num}{sample_side}_{light_type}_ld_mask.png", cleaned_ld_mask)