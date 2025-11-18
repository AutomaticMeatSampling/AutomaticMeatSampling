from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve()
def rel_path(path):
    filepath = SCRIPT_DIR.parent / path
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    return filepath


"""
Helper Functions for graphing


"""

def show_mask(mask, ax, random_color=False):
    """
    Displays "mask" on axis "ax" with possible random color

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
    Displays input coords on axis "ax" with green and red color for positive and negative labels respectively
    
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

def show_images(images, titles=None, points=None, figsize=(12, 6), save=True, filename=None):
    """
    Displays a list of images in a single row.
    Automatically uses a grayscale colormap for 2D images.

    Args:
        images (list): List of images (numpy arrays or PIL images)
        titles (list or None): List of titles (same length as images)
        figsize (tuple): Figure size
    """
    n = len(images)
    plt.figure(figsize=figsize)

    if titles is None:
        titles = [""] * n

    if points is None:
        points = [None] * n

    for i, img in enumerate(images):
        plt.subplot(1, n, i + 1)

        # Auto-detect grayscale vs color
        use_gray = (np.array(img).ndim == 2)
        if use_gray:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)

        if points[i] is not None:
            p = points[i]
            for pt in p:
                plt.scatter(pt['x'], pt['y'], color=pt.get('color', 'red'),
                            s=40, marker='o')

        plt.title(titles[i])
        plt.axis('off')


    if save:
        if filename == None:
            fname = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = rel_path(f"temp/{fname}.png")

        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {filename}")

    plt.show()

"""
Functions for using Segment Anything Model (SAM) code


"""

def load_sam_model(checkpoint_file_path=rel_path("sam_vit_b_01ec64.pth")):
    """
    Get SamPredictor for specific checkpoint

    Args:
        checkpoint_file_path (str): File path to checkpoint

    Returns:
        SamPredictor
    """
    if "vit_h" in str(checkpoint_file_path):
        model_type = "vit_h"
    elif "vit_l" in str(checkpoint_file_path):
        model_type = "vit_l"
    else:
        model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_file_path)
    predictor = SamPredictor(sam)
    return predictor


def run_prompt_on_image(predictor, image, input_points, input_labels):
    """
    Sets image in predictor and gets masks based on input points (background & foreground)

    Args:
        predictor (SamPredictor): Loaded SamPredictor from specific checkpoint
        image (np.ndarray): Loaded cv2 image
        input_points (np.ndarray): Input foreground and backround points
        input_labels (np.ndarray): Labels of input_points (1 = foreground, 0 = background)

    Returns:
        masks (np.ndarray): Possible masks of meat sample
        scores (np.ndarray): IoU score returned by predictor
        logits (n/a)
    """

    print("Step 4: Setting image in predictor...")
    predictor.set_image(image)

    print("Step 5: Generating masks...")
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    return masks, scores, logits

def calculate_background_points(green_mask, show=True):
    """
    Given a rough mask (green_mask) of the steak, select background points - farthest from the mask
    """
    # print("Selecting background point 1...")
    distance_map = cv2.distanceTransform(green_mask, distanceType=cv2.DIST_L2, maskSize=5)

    # Get coordinates of the two farthest background pixels
    flat_indices = np.argpartition(distance_map.ravel(), -2)[-2:]  # get indices of top 2 distances
    y_coords, x_coords = np.unravel_index(flat_indices, distance_map.shape)

    # Create background points (x, y)
    background_input_point = np.stack((x_coords, y_coords), axis=1)

    return background_input_point

def format_input_points(image, foreground_input_points=None, background_input_points=None):
    """
    Combine foreground_input_points and background_input_points

    Args:
        image (np.ndarray): Used in case no foreground points are given, choose middle of image
        foreground_input_points
        background_input_points

    Returns:
        input_points
        input_labels (1 = foreground, 0 = background)
    """

    if foreground_input_points is None:
        input_points = np.array([[len(image[0])//2, len(image)//2]])
        input_labels = np.array([1]) # Label 1 = foreground, 0 = background
    else:
        input_points = foreground_input_points
        input_labels = np.ones(len(foreground_input_points), dtype=np.int32)

    if background_input_points is not None:
        input_points = np.concatenate((input_points, background_input_points), axis=0)
        input_labels = np.concatenate((input_labels, np.zeros(len(background_input_points), dtype=np.int32)), axis=0)

    return input_points, input_labels

def get_best_mask(masks, scores, option="smallest_mask", min_size=0):
    """
    Select best mask based on option provided

    """

    if option != "smallest_mask" and option != "highest_score":
        raise ValueError("Option is either smallest_mask or highest_score")
    
    if option == "smallest_mask":
        # Option 1: Best = smallest mask size that is bigger than min_size
    
        print("Step 6: Selecting best mask: Smallest mask that is at least half of the total meat area")
        valid_masks = [(i, mask) for i, mask in enumerate(masks) if np.sum(mask) >= min_size]
        if not valid_masks:
            print("No valid masks found with size >= min_size. Using largest mask instead.")
            best_mask_index = np.argmin([np.sum(mask >= 1) for mask in masks])
            best_mask = masks[best_mask_index].astype(np.uint8) * 255
        else:
            best_mask_index = np.argmin([np.sum(mask >=1 ) for idx, mask in valid_masks])
            best_mask = valid_masks[best_mask_index][1].astype(np.uint8) * 255

    else:
        # Option 2: Set best mask to mask with highest score
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index].astype(np.uint8) * 255

    return best_mask

def remove_green_background(image_path=rel_path("images/sample7.jpg"), show=True):
    """
    Roughly remove green background behind steak and return rough mask of steak
    """
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
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Display the result
    if show:
        show_images([original, meat_mask], ["Original Image", "Removed Green Background"], filename=rel_path("temp/remove_green_background-img.png"))
    
    return result, meat_mask, green_mask


def rough_segment_steak(path=rel_path("images/sample1.jpg"), image=None, meat_mask=None, show=True):
    """
    Roughly segment steak based on otsu thresholding and contour detection to find foreground (steak) and background (green/flat background)
    """
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
        if show:
            show_images([thresh], ["Otsu Thresholded Image without Meat Mask"], filename=rel_path("temp/rough_segment_steak-thresh.png"))
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

    # Find contours (boundary tracking)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and take the largest (assumed ribeye)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        print("No contours found.")
        return thresh

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
    if show:
        points = [None, None, [
            {   'x': [fx],
                'y': [fy],
                'color': 'green',
                'label': 'Foreground' },
            {   'x': [bx],
                'y': [by],
                'color': 'red',
                'label': 'Background'}
        ]]
        show_images([image, thresh, segmented], ["Original Image", "Otsu Thresholded Image", "Segmented Meat with Selected Point"], points=points, filename=rel_path("temp/rough_segment_steak-final.png"))

    # Calculate number of pixels in thresh
    num_foreground_pixels = np.sum(thresh == 255)

    return np.array([[fx, fy]]), np.array([[bx, by]]), num_foreground_pixels

def fill_holes(binary_mask):
    """
    Description: Fills holes in a binary mask using flood fill algorithm.
    Args:
        binary_mask (np.array): Binary mask to fill holes in

    Returns:
        out (np.array): Binary mask with holes filled
    """
    # Step 0: Ensure mask is binary (0 or 255)
    mask_bin = (binary_mask > 0).astype(np.uint8) * 255

    kernel_size=10
    dilation_iter=2

    # Step 1: Dilate to connect small gaps
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_bin, kernel, iterations=dilation_iter)

    binary_mask = dilated

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

def post_processing_ld_mask(ld_mask, min_size=5000, show=True):
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

    cv2.imwrite(rel_path("temp/post_processing_ld_mask-output_filled_ld_mask1.png"), ld_mask_new)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ld_mask_new, connectivity=8)
    sizes = stats[1:, -1]
    cleaned_mask = np.zeros_like(ld_mask_new)
    for i, size in enumerate(sizes):
        if size > min_size:
            cleaned_mask[labels == i + 1] = 255
    ld_mask_new = cleaned_mask

    cv2.imwrite(rel_path("temp/post_processing_ld_mask-output_cleaned_ld_mask2.png"), ld_mask_new)

    # Keep only the biggest area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ld_mask_new, connectivity=8)
    sizes = stats[1:, -1]
    if len(sizes) > 0:
        largest_cc_index = np.argmax(sizes)
        ld_mask_new = np.zeros_like(ld_mask_new)
        ld_mask_new[labels == largest_cc_index + 1] = 255

    cv2.imwrite(rel_path("temp/post_processing_ld_mask-output_largest_ld_mask3.png"), ld_mask_new)

    # shrink the overall ld_mask to avoid getting any of the fat
    ld_mask_new = cv2.erode(ld_mask_new, np.ones((10, 10), np.uint8), iterations=14)

    if show:
        # Show before and after
        show_images([ld_mask, ld_mask_new], ['Original LD Mask', 'Cleaned LD Mask'], filename=rel_path("temp/post_processing_ld_mask-cleaned-mask.png"))
    

    return ld_mask_new


def segment_ld(steak_img_path, predictor, show=True):
    # Create temp folder if it doesn't exist
    os.makedirs("temp", exist_ok=True)

    # Read images
    image = cv2.imread(steak_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 0: Removing green background...
    result, meat_mask, green_mask = remove_green_background(steak_img_path, show=False)
    cv2.imwrite(rel_path("temp/step_0_meat_mask.png"), meat_mask)
    cv2.imwrite(rel_path("temp/step_0_no_green.png"), result)

    # Get background points from green mask
    background_input_points1 = calculate_background_points(green_mask, show=show)

    # Get rough size of steak and more points from steak+background
    foreground_midpoint, background_midpoint, thresh_mask_size = rough_segment_steak(path=steak_img_path, meat_mask=meat_mask, show=show)
    bp = np.concatenate((background_input_points1, background_midpoint), axis=0)

    # Format input points for prompt
    input_points, input_labels = format_input_points(image, foreground_midpoint, bp)

    # Run predictor on image
    masks, scores, logits = run_prompt_on_image(predictor, image, input_points, input_labels)

    if show:
        titles = []
        images = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            titles.append(f"Mask {i+1}\nScore: {score:.3f}\n# Pixels: {np.sum(mask)}")
            images.append(mask.astype(np.uint8) * 255)
        show_images(images, titles, filename=rel_path("temp/segment_ld-all-masks.png"))


    # Get best LD mask
    best_option = "smallest_mask"
    min_size = thresh_mask_size//2
    best_mask = get_best_mask(masks, scores, option=best_option, min_size=min_size)

    # Post processing step:
    cleaned_best_mask = post_processing_ld_mask(best_mask, show=show)

    cv2.imwrite(rel_path("temp/step_6_best_ld_mask.png"), cleaned_best_mask)

    return cleaned_best_mask

if __name__ == "__main__":
    steak_img_path = rel_path("real_images/sample2a_both.png")

    # Create predictor
    print("!!!! Loading checkpoint into predictor... !!!!")
    predictor = load_sam_model()
    cleaned_mask = segment_ld(steak_img_path, predictor)