import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

# Get masks from a given prompt:

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def test_prompt():
    print("Step 1: Read images and convert to RGB")
    image = cv2.imread("images/sample6.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
    input_point = np.array([[len(image[0])//2, len(image)//2]])
    input_label = np.array([1]) # Label 1 = foreground, 0 = background
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show() 


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


def test_automated_generator():
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

if __name__ == "__main__":
    test_prompt()