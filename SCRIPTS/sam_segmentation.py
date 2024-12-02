import argparse
from datasets import DataLoader
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import os
from PIL import Image
import imgviz
import torch

parser = argparse.ArgumentParser(
    prog='CUB-SAM',
    description='Employ SAM for bird segmentation and ratio calculations'
)
parser.add_argument('--prompt', type=str, default='point', help="prompt type for segmentation from [box, point, click]")
parser.add_argument('--model', type=str, default='vit_l', help="SAM model type from [vit_b, vit_l, vit_h]")

args = parser.parse_args()

def merge_segmentation(part_masks, point_labels):
    """
    Merge segmentation results for different point prompts.
    """
    assert len(part_masks), 'No predictions from SAM'
    mask = np.zeros_like(part_masks[0], dtype=np.float32)
    areas = list(map(lambda x: x.sum(), part_masks))
    inds = sorted(range(len(point_labels)), key=lambda x: areas[x], reverse=True)
    for i in inds:
        m, l = part_masks[i], point_labels[i]
        mask[m] = l
    return mask

def save_colored_mask(mask, save_path):
    """
    Save a colored segmentation mask.
    """
    mask = Image.fromarray(mask.astype(np.uint8), mode='P')
    colormap = imgviz.label_colormap()
    mask.putpalette(colormap.flatten())
    mask.save(save_path)

def calculate_ratio(beak_mask, body_mask):
    """
    Calculate the ratio of the beak area to the body area.
    """
    beak_area = np.sum(beak_mask)
    body_area = np.sum(body_mask)
    return beak_area / body_area if body_area > 0 else 0

def ensure_dir_exists(file_path):
    """
    Ensure the directory for the given file path exists.
    If it doesn't exist, create it.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    if not os.path.exists('segmentations'):
        os.mkdir('segmentations')
    mask_dir = os.path.join('segmentations', 'mul_mask')
    ratio_dir = os.path.join('segmentations', 'ratios')
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    if not os.path.exists(ratio_dir):
        os.mkdir(ratio_dir)
    
    # Download pretrained weights
    if args.model == 'vit_b':
        model_path = 'sam_vit_b_01ec64.pth'
    elif args.model == 'vit_l':
        model_path = 'sam_vit_l_0b3195.pth'
    elif args.model == 'vit_h':
        model_path = 'sam_vit_h_4b8939.pth' 
    else:
        raise NameError
    
    if not os.path.isfile(model_path):
        os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/{model_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running model on {device}")
    
    sam = sam_model_registry[args.model](checkpoint=model_path)
    sam = sam.to(device=device)
    predictor = SamPredictor(sam)
    
    loader = DataLoader('data/CUB_200_2011')
    
    for index in tqdm(range(len(loader))):
        image_path, image, bboxes, point_coords, point_labels, click_coords, click_labels = loader(index)
        print(f"[DEBUG] Processing {image_path}")
        save_path_mask = os.path.join(mask_dir, image_path.replace('jpg', 'png'))
        save_path_ratio = os.path.join(ratio_dir, image_path.replace('jpg', '_ratio.txt'))
        
        # Ensure directories exist before saving
        ensure_dir_exists(save_path_mask)
        ensure_dir_exists(save_path_ratio)

        if os.path.isfile(save_path_mask) and os.path.isfile(save_path_ratio):
            continue
        
        predictor.set_image(image)
        part_masks = {}
        labels_to_process = {'body': 1, 'beak': 2}  # Flip these labels to correct assignment

        # Extract masks for specific parts
        for part_name, part_label in labels_to_process.items():
            coords = [c for c, l in zip(point_coords, point_labels) if l == part_label]
            if coords:
                masks, _, _ = predictor.predict(
                    point_coords=np.array(coords), point_labels=np.array([part_label] * len(coords))
                )
                part_masks[part_name] = masks[0]  # Take the first mask for simplicity

        # Ensure masks exist before processing
        if 'beak' in part_masks and 'body' in part_masks:
            beak_mask = part_masks['beak']
            body_mask = part_masks['body']

            # Calculate and save the ratio
            ratio = calculate_ratio(beak_mask, body_mask)
            with open(save_path_ratio, 'w') as f:
                f.write(f"Beak-to-Body Ratio: {ratio:.4f}\n")
            print(f"[DEBUG] Saved ratio file for {image_path}")

            # Merge and save the segmentation mask
            combined_mask = merge_segmentation([beak_mask, body_mask], [2, 1])  # Ensure beak is label 2, body is label 1
            save_colored_mask(combined_mask, save_path_mask)
        else:
            print(f"[DEBUG] Missing required masks for {image_path}")
