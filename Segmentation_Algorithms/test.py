import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import time
from sklearn.cluster import KMeans
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

def test_prompt(path=rel_path("images/sample6.jpg"), checkpoint=rel_path("sam_vit_b_01ec64.pth"), foreground_input_points=None, background_input_points=None, show=True, min_size=0):
    """
    Description: Generates masks based on user input points (foreground vs background) using the Segment Anything Model (SAM)

    Args:
        path (str): Path to the input image
        checkpoint (str): Path to the SAM model checkpoint
        foreground_input_points (np.array): Array of (x, y) coordinates for foreground points
        background_input_points (np.array): Array of (x, y) coordinates for background points
        show (bool): Whether to display the generated masks
    """

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Step 2: Load the Segment Anything Model (SAM) Checkpoint")
    predictor = segment_ld.load_sam_model(checkpoint)

    input_points, input_labels = segment_ld.format_input_points(image, foreground_input_points, background_input_points)

    masks, scores, logits = segment_ld.run_prompt_on_image(predictor, image, input_points, input_labels)

    best_option = "smallest_mask"
    best_mask = segment_ld.get_best_mask(masks, scores, option=best_option, min_size=min_size)
    
    output_path = path.split("/")[-1].split(".")[0] + "_output.png"
    if best_option == "smallest_mask":   
        cv2.imwrite(rel_path("images/masks/" + output_path), best_mask)
    elif best_option == "highest_score":
        cv2.imwrite(rel_path("images/masks/" + "score_based_" + output_path), best_mask)



    # Save all masks to temp/output_ld_mask_{i}.png where i is the rank (1 = best, 2 = second best, 3 = third best)
    for i, mask in enumerate(masks):
        ranked_mask = mask.astype(np.uint8) * 255
        cv2.imwrite(rel_path(f"temp/output_ld_mask_{i+1}.png"), ranked_mask)
        # print(f"Saved mask {i+1} with Score: {scores[i]:.3f}, # of Pixels: {np.sum(mask)}")



    return best_mask



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
        path = rel_path(f"images/samples_green/sample{sample_num}_green.jpg")
        ld_mask_path = rel_path(f"images/masks/sample{sample_num}_green_output.png")
    else:
        path = rel_path(f"real_images/sample{sample_num}{sample_side}_{light_type}.png")
        ld_mask_path = rel_path(f"images/masks/sample{sample_num}{sample_side}_{light_type}_output.png")
    
    marbling_only = False
    show = True
    # test_prompt(path)
    # test_segment_tissue(path)

    if not marbling_only:
    
        start = time.time()
        print("Step 0: Reading image and removing green background...")
        result, meat_mask, green_mask = segment_ld.remove_green_background(path, show=False)
        # Save meat_mask to temp/output_meat_mask.png
        cv2.imwrite(rel_path(f"temp/output_meat_mask.png"), meat_mask)
        # Set all parts of orig image where meat_mask is 0 to white - but rn it shows up as blue instead of red fix it
        cv2.imwrite(rel_path(f"temp/output_no_green.png"), result)

        # print("Remove green background time taken: ", time.time() - start)

        start = time.time()
        background_input_points1 = segment_ld.calculate_background_points(green_mask, show=show)
        foreground_midpoint, background_midpoint, thresh_mask_size = segment_ld.rough_segment_steak(path=path, meat_mask=meat_mask, show=show)
        print("     Finding center point of ribeye time taken: ", time.time() - start)

        bp = np.concatenate((background_input_points1, background_midpoint), axis=0)

        checkpoint_option = "B"  # Choose from "B", "H", "L"
        if checkpoint_option not in ["B", "H", "L"]:
            raise ValueError("Invalid checkpoint option. Choose from 'B', 'H', 'L'.")
        if checkpoint_option == "B":
            # Lightest model (base)
            checkpoint_file = rel_path("sam_vit_b_01ec64.pth")
        elif checkpoint_option == "H":
            checkpoint_file = rel_path("sam_vit_h_4b8939.pth")
        else:
            checkpoint_file = rel_path("sam_vit_l_0b3195.pth")

        predictor = segment_ld.load_sam_model(checkpoint_file)
        best_mask_ld = segment_ld.segment_ld(path, predictor)

    else:
        best_mask_ld = cv2.imread(ld_mask_path, cv2.IMREAD_GRAYSCALE)
        if best_mask_ld is None:
            raise ValueError(f"Could not read mask image at {ld_mask_path}")


    # Use mask and original image to separate marbling fat (white) from muscle (red)
    start = time.time()
    results = segment_marbling.segment_tissue(path, best_mask_ld)
    for key, value in results.items():
        cv2.imwrite(rel_path(f"temp/output_{key}.png"), value)
    
    final_marbling = results["final_marbling"]
    # Write to images/masks/sample{sample_num}_marbling.png
    cv2.imwrite(rel_path(f"images/masks/sample{sample_num}{sample_side}_{light_type}_marbling_mask.png"), final_marbling)
    
    final_muscle = cv2.bitwise_and(best_mask_ld, cv2.bitwise_not(final_marbling))
    cv2.imwrite(rel_path(f"images/masks/sample{sample_num}{sample_side}_{light_type}_muscle_mask.png"), final_muscle)

    cv2.imwrite(rel_path(f"images/masks/sample{sample_num}{sample_side}_{light_type}_muscle_mask_2.png"), results["final_muscle"])

    cleaned_ld_mask = results["cleaned_ld_mask"]
    cv2.imwrite(rel_path(f"images/masks/sample{sample_num}{sample_side}_{light_type}_ld_mask.png"), cleaned_ld_mask)