import cv2
import numpy as np
import os


# Function to compute accuracy
def compute_accuracy(true_mask, predicted_mask):
    # Ensure both masks have the same shape
    if true_mask.shape != predicted_mask.shape:
        raise ValueError("True mask and predicted mask must have the same dimensions")

    # Flatten the masks to 1D arrays for easier computation
    true_mask_flat = true_mask.flatten()
    predicted_mask_flat = predicted_mask.flatten()

    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
    TP = np.sum((true_mask_flat != 0) & (predicted_mask_flat != 0))
    TN = np.sum((true_mask_flat == 0) & (predicted_mask_flat == 0))
    FP = np.sum((true_mask_flat == 0) & (predicted_mask_flat != 0))
    FN = np.sum((true_mask_flat != 0) & (predicted_mask_flat == 0))

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return accuracy

# Calculate F1 score
def compute_f1_score(true_mask, predicted_mask):
    # Ensure both masks have the same shape
    if true_mask.shape != predicted_mask.shape:
        raise ValueError("True mask and predicted mask must have the same dimensions")

    # Flatten the masks to 1D arrays for easier computation
    true_mask_flat = true_mask.flatten()
    predicted_mask_flat = predicted_mask.flatten()

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = np.sum((true_mask_flat != 0) & (predicted_mask_flat != 0))
    FP = np.sum((true_mask_flat == 0) & (predicted_mask_flat != 0))
    FN = np.sum((true_mask_flat != 0) & (predicted_mask_flat == 0))

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

def compute_accuracy_marbling(true_marbling_mask, predicted_marbling_mask, ld_mask):
    # Ensure all masks have the same shape
    if true_marbling_mask.shape != predicted_marbling_mask.shape or true_marbling_mask.shape != ld_mask.shape:
        raise ValueError("All masks must have the same dimensions")

    # Flatten the masks to 1D arrays for easier computation
    true_marbling_flat = true_marbling_mask.flatten()
    predicted_marbling_flat = predicted_marbling_mask.flatten()
    ld_mask_flat = ld_mask.flatten()

    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
    TP = np.sum((true_marbling_flat != 0) & (predicted_marbling_flat != 0) & (ld_mask_flat != 0))
    TN = np.sum((true_marbling_flat == 0) & (predicted_marbling_flat == 0) & (ld_mask_flat != 0))
    FP = np.sum((true_marbling_flat == 0) & (predicted_marbling_flat != 0) & (ld_mask_flat != 0))
    FN = np.sum((true_marbling_flat != 0) & (predicted_marbling_flat == 0) & (ld_mask_flat != 0))

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return accuracy

def compute_f1_score_marbling(true_marbling_mask, predicted_marbling_mask, ld_mask):
    # Ensure all masks have the same shape
    if true_marbling_mask.shape != predicted_marbling_mask.shape or true_marbling_mask.shape != ld_mask.shape:
        raise ValueError("All masks must have the same dimensions")

    # Flatten the masks to 1D arrays for easier computation
    true_marbling_flat = true_marbling_mask.flatten()
    predicted_marbling_flat = predicted_marbling_mask.flatten()
    ld_mask_flat = ld_mask.flatten()

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = np.sum((true_marbling_flat != 0) & (predicted_marbling_flat != 0) & (ld_mask_flat != 0))
    FP = np.sum((true_marbling_flat == 0) & (predicted_marbling_flat != 0) & (ld_mask_flat != 0))
    FN = np.sum((true_marbling_flat != 0) & (predicted_marbling_flat == 0) & (ld_mask_flat != 0))

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score


if __name__ == "__main__":

    # Specify the sample ID
    sample_id = 5

    # Define paths to the true and predicted masks
    true_mask_path = os.path.join("images", "true", "LD",f"sample{sample_id}_true.jpg")
    predicted_mask_path = os.path.join("images", "masks", f"sample{sample_id}_green_output.png")

    true_marbling_mask_path = os.path.join("images", "true", "marbling", f"sample{sample_id}_true_marbling.jpg")
    predicted_marbling_mask_path = os.path.join("images", "masks", f"sample{sample_id}_marbling_mask.png")
    predicted_muscle_mask_path = os.path.join("images", "masks", f"sample{sample_id}_muscle_mask.png")

    # Load the true and predicted masks in grayscale mode
    true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
    true_marbling_mask = cv2.imread(true_marbling_mask_path, cv2.IMREAD_GRAYSCALE)

    predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

    # Flip values of true mask (0 to 255 and 255 to 0)
    true_mask = np.where(true_mask == 255, 0, 255).astype(np.uint8)
    true_marbling_mask = np.where(true_marbling_mask == 255, 0, 255).astype(np.uint8)

    # Calculate true muscle mask by subtracting true marbling mask from true mask
    true_muscle_mask = cv2.bitwise_and(cv2.bitwise_not(true_marbling_mask), true_mask)

    predicted_marbling_mask = cv2.imread(predicted_marbling_mask_path, cv2.IMREAD_GRAYSCALE)
    predicted_muscle_mask = cv2.imread(predicted_muscle_mask_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded properly
    if true_mask is None:
        raise FileNotFoundError(f"True mask not found at {true_mask_path}")
    if predicted_mask is None:
        raise FileNotFoundError(f"Predicted mask not found at {predicted_mask_path}")

    # Compute accuracy
    accuracy = compute_accuracy(true_mask, predicted_mask)
    f1_score = compute_f1_score(true_mask, predicted_mask)

    # Print the accuracy
    print(f"LD Accuracy for sample {sample_id}: {accuracy:.4f}")
    print(f"LD F1 Score for sample {sample_id}: {f1_score:.4f}")


    # Compute accuracy for marbling
    if true_marbling_mask is None:
        raise FileNotFoundError(f"True marbling mask not found at {true_marbling_mask_path}")
    if predicted_marbling_mask is None:
        raise FileNotFoundError(f"Predicted marbling mask not found at {predicted_marbling_mask_path}")
    if predicted_muscle_mask is None:
        raise FileNotFoundError(f"Predicted initial meat mask not found at {predicted_muscle_mask_path}")

    accuracy_marbling = compute_accuracy_marbling(true_marbling_mask, predicted_marbling_mask, predicted_mask)
    f1_score_marbling = compute_f1_score_marbling(true_marbling_mask, predicted_marbling_mask, predicted_mask)
    print(f"Marbling Accuracy for sample {sample_id}: {accuracy_marbling:.4f}")
    print(f"Marbling F1 Score for sample {sample_id}: {f1_score_marbling:.4f}")

    # Compute accuracy for muscle
    accuracy_muscle = compute_accuracy_marbling(true_muscle_mask, predicted_muscle_mask, predicted_mask)
    f1_score_muscle = compute_f1_score_marbling(true_muscle_mask, predicted_muscle_mask, predicted_mask)
    print(f"Muscle Accuracy for sample {sample_id}: {accuracy_muscle:.4f}")
    print(f"Muscle F1 Score for sample {sample_id}: {f1_score_muscle:.4f}")
