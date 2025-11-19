import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import segment_ld
import segment_marbling
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve()
def rel_path(path):
    filepath = SCRIPT_DIR.parent / path
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    return filepath

def get_sample_points(marbling_mask_img_path, muscle_mask_img_path, num_marbling_points, num_muscle_points, show=True):
    # Read binary masks
    muscle_mask = cv2.imread(muscle_mask_img_path, cv2.IMREAD_GRAYSCALE)
    marbling_mask = cv2.imread(marbling_mask_img_path, cv2.IMREAD_GRAYSCALE)

    if muscle_mask is None:
        raise FileNotFoundError(f"Image not found: {muscle_mask_img_path}")
    if marbling_mask is None:
        raise FileNotFoundError(f"Image not found: {marbling_mask_img_path}")
    
    _, muscle_mask = cv2.threshold(muscle_mask, 127, 255, cv2.THRESH_BINARY)
    _, marbling_mask = cv2.threshold(marbling_mask, 127, 255, cv2.THRESH_BINARY)

    # Initialize selected points
    selected_muscle_points = []
    selected_marbling_points = []

    # Total points to select
    total_points = num_muscle_points + num_marbling_points

    for i in range(total_points):
        if (i % 2 == 0 and len(selected_muscle_points) < num_muscle_points) or (len(selected_marbling_points) == num_marbling_points):
            # Recompute the distance transform for the muscle mask
            muscle_distance_map = cv2.distanceTransform(muscle_mask, cv2.DIST_L2, 5)

            # Distance Map of Muscles
            if show:
                plt.imshow(muscle_distance_map, cmap='hot')
                plt.colorbar()
                plt.title("Muscle Distance Transform Map")
                plt.show()

            max_dist_point = np.unravel_index(np.argmax(muscle_distance_map), muscle_distance_map.shape)

            # Add the selected point to the muscle list
            selected_muscle_points.append(max_dist_point)

            # Update the muscle mask to exclude the selected point
            cv2.circle(muscle_mask, (max_dist_point[1], max_dist_point[0]), radius=100, color=0, thickness=-1)
            cv2.circle(marbling_mask, (max_dist_point[1], max_dist_point[0]), radius=100, color=0, thickness=-1)

           
            # UNCOMMENT BELOW TO SHOW INTERMEDIATE STEPS:
            # Display muscle_mask at this point
            if show:
                plt.imshow(muscle_mask, cmap='gray')
                plt.axis("off")
                plt.title("Muscle Mask with Selected Point Blacked Out")
                plt.show()

        elif (i % 2 == 1 and len(selected_marbling_points) < num_marbling_points) or (len(selected_muscle_points) == num_muscle_points):
            # Recompute the distance transform for the marbling mask
            marbling_distance_map = cv2.distanceTransform(marbling_mask, cv2.DIST_L2, 5)

            # Distance Map of Muscles
            if show:
                plt.imshow(marbling_distance_map, cmap='hot')
                plt.colorbar()
                plt.title("Marbling Distance Transform Map")
                plt.show()

            # Select the point with the maximum distance
            max_dist_point = np.unravel_index(np.argmax(marbling_distance_map), marbling_distance_map.shape)

            # Add the selected point to the marbling list
            selected_marbling_points.append(max_dist_point)

            # Update the marbling mask to exclude the selected point
            cv2.circle(marbling_mask, (max_dist_point[1], max_dist_point[0]), radius=100, color=0, thickness=-1)
            cv2.circle(muscle_mask, (max_dist_point[1], max_dist_point[0]), radius=100, color=0, thickness=-1)

            # UNCOMMENT BELOW TO SHOW INTERMEDIATE STEPS:
            # Display muscle_mask at this point
            # plt.imshow(marbling_mask, cmap='gray')
            # plt.axis("off")
            # plt.title("Marbling Mask with Selected Point Blacked Out")
            # plt.show()

    if show:
        plt.imshow(muscle_mask, cmap='gray')
        plt.axis("off")
        plt.title("Muscle Mask with Selected Point Blacked Out")
        plt.show()
    
    if show:
        plt.imshow(marbling_mask, cmap='gray')
        plt.axis("off")
        plt.title("Marbling Mask with Selected Point Blacked Out")
        plt.show()

    return selected_muscle_points, selected_marbling_points

def show_final_selected(img_path, muscle_points, marbling_points):
    # Load the original image
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"Image not found: {orig_img}")

    # Convert BGR to RGB for plotting
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Plot the original image
    plt.imshow(orig_img_rgb)
    plt.axis("off")

     # Plot muscle points (red, open circles)
    for point in muscle_points:
        print(point)
        plt.scatter(
            point[1], point[0], 
            edgecolors="green", facecolors="none", 
            label="Muscle Point" if "Muscle Point" not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    # Plot marbling points (blue, open circles)
    for point in marbling_points:
        print(point)
        plt.scatter(
            point[1], point[0], 
            edgecolors="blue", facecolors="none", 
            label="Marbling Point" if "Marbling Point" not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    # Add legend
    plt.legend()

    # Show the image
    plt.show()

def segment_and_select_points(steak_img_path, predictor, num_marbling_pts=1, num_muscle_pts=0, show=True):
    ld_mask = segment_ld.segment_ld(steak_img_path, predictor, show)
    segment_tissue_results = segment_marbling.segment_tissue(steak_img_path, ld_mask, show)

    marbling_mask_path = rel_path("temp/final_marbling_mask.png")
    muscle_mask_path = rel_path("temp/final_muscle_mask.png")

    cv2.imwrite(marbling_mask_path, segment_tissue_results["final_marbling"])
    cv2.imwrite(muscle_mask_path, segment_tissue_results["final_muscle"])

    muscle_points, marbling_points = get_sample_points(marbling_mask_path, muscle_mask_path, num_marbling_pts, num_muscle_pts, show=show)

    if show:
        show_final_selected(steak_img_path, muscle_points, marbling_points)

    return muscle_points, marbling_points

if __name__ == "__main__":
    sample_id = 1
    sample_side = "a" # "a/b" or "" for fake images
    light_type = "both" # "both" or "" for fake images
    sample_type = "real"

    if len(sys.argv) > 2:
        sample_id = sys.argv[1]
        sample_side = sys.argv[2]

    if sample_type == "fake":
        ld_mask_path = rel_path(f"images/masks/sample{sample_id}_green_output.png")
        orig_img_path = rel_path(f"images/samples_green/sample{sample_id}_green.jpg")
        muscle_mask_path = rel_path(f"images/masks/sample{sample_id}_muscle_mask_2.png")
        marbling_mask_path = rel_path(f"images/masks/sample{sample_id}_marbling_mask.png")
    else:
        ld_mask_path = rel_path(f"images/masks/sample{sample_id}{sample_side}_{light_type}_ld_mask.png")
        orig_img_path = rel_path(f"real_images/sample{sample_id}{sample_side}_{light_type}.png")
        muscle_mask_path = rel_path(f"images/masks/sample{sample_id}{sample_side}_{light_type}_muscle_mask_2.png")
        marbling_mask_path = rel_path(f"images/masks/sample{sample_id}{sample_side}_{light_type}_marbling_mask.png")

    num_muscle_pts = 2
    num_marbling_pts = 2
    

    muscle_points, marbling_points = get_sample_points(marbling_mask_path, muscle_mask_path, num_marbling_pts, num_muscle_pts)

    show_final_selected(orig_img_path, marbling_points=marbling_points, muscle_points=muscle_points)
    