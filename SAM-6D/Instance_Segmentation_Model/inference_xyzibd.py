import torch
import numpy as np
import cv2
import os
import yaml
import json
import argparse
from omegaconf import OmegaConf

import sys
sys.path.append('SAM-6D')
from Instance_Segmentation_Model.preprocess.image_enhancement.image_utils import reduce_overexposure, enhance_image
from Instance_Segmentation_Model.preprocess.image_enhancement.pose_utils import crop_frame, update_intrinsics
from Instance_Segmentation_Model.preprocess.image_enhancement.HVI_net.CIDNet import CIDNet
from Instance_Segmentation_Model.preprocess.grounding_dino.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
import safetensors.torch as sf
from tqdm import tqdm
from hydra import initialize, compose
from hydra.utils import instantiate
import logging
import time

def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def nms_masks(all_masks, iou_threshold=0.5):
    all_masks = sorted(all_masks, key=lambda x: x['area'], reverse=True)
    keep_masks = []

    for i in range(len(all_masks)):
        current = all_masks[i]
        current_mask = current['segmentation']
        suppress = False

        for kept in keep_masks:
            kept_mask = kept['segmentation']
            iou = mask_iou(current_mask, kept_mask)
            if iou > iou_threshold:
                suppress = True
                break
        
        if not suppress:
            keep_masks.append(current)

    return keep_masks

def crop_image(res, rgb, depth):
    masks = res['masks']
    image_pil_bboxes = res['bboxes']

    bbox_frame = image_pil_bboxes[0]
    mask_frame = masks[0].cpu().numpy().astype(np.uint8)
    # Crop the image based on the bounding box
    x1, y1, x2, y2 = bbox_frame
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    # Crop
    mask_frame = mask_frame[0][y1:y2, x1:x2]
    cropped_rgb = rgb.copy()[y1:y2, x1:x2]  
    cropped_depth = depth.copy()[y1:y2, x1:x2]

    bbox = [x1, y1, x2, y2]
    
    return cropped_rgb, cropped_depth, mask_frame, bbox   

def image_aug(HVI_model, image):
    # Calculate mean intensity
    mean_intensity = np.mean(image) 
    # Set alpha_i and gamm based on mean intensity
    if mean_intensity > 50:
        enhanced_image = reduce_overexposure(image)
    else:
        alpha_s = 1.0
        alpha_i = 1.0 
        gamma = 1.0
        enhanced_image = enhance_image(HVI_model, image, alpha_s, alpha_i, gamma, mean_intensity)
    
    return enhanced_image

def draw_sam_seg(masks, area_threshold=None, alpha=0.5):
    """
    Draw SAM masks on a black background using numpy arrays directly.

    Args:
        masks (list of dicts): list of masks, each with 'segmentation', 'area', etc.
        area_threshold (int or None): ignore masks smaller than this.
        alpha (float): transparency level of masks.

    Returns:
        np.ndarray: (H, W, 3) RGB image
    """
    if len(masks) == 0:
        return None

    # Automatically infer shape from first mask
    first_mask = masks[0]['segmentation']
    H, W = first_mask.shape

    # Create black background
    canvas = np.zeros((H, W, 3), dtype=np.float32)

    # Sort masks by area descending
    sorted_anns = sorted(masks, key=lambda x: x['area'], reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']

        if area_threshold is not None and np.sum(m) < area_threshold:
            continue

        # Random color
        mask_color = np.random.random(3)

        # Blend mask into canvas
        for c in range(3):
            canvas[:, :, c] = np.where(
                m,
                canvas[:, :, c] * (1 - alpha) + mask_color[c] * alpha,
                canvas[:, :, c]
            )

    # Convert to uint8
    canvas = np.clip(canvas, 0, 1)
    canvas = (canvas * 255).astype(np.uint8)
    return canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection")
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="name of the dataset to use (e.g., ipd, xyzibd, itodd.)"
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="path to YAML config file"
    )
    parser.add_argument(
        "--sam_type",
        default="ISM_sam",
        choices=["ISM_fastsam", "ISM_sam"],
        help="which SAM variant to use"
    )

    args = parser.parse_args()
    dataset_name = args.dataset_name
    infer_cfg_path = args.cfg
    sam_type = args.sam_type

    with open(infer_cfg_path) as f:
        infer_cfg = yaml.safe_load(f)
    infer_dataset_cfg = infer_cfg["datasets"].get(dataset_name)
    infer_model_cfg = infer_cfg["model"]

    # load the data
    data_base = infer_dataset_cfg['data_base']
    scene_ids = sorted(int(id) for id in (os.listdir(data_base)))
    if "depth_range" in infer_dataset_cfg:
        depth_range = infer_dataset_cfg['depth_range']  

    # Initialize the Instance Segmentation Model v2
    inference_cfg_path = "configs"
    initialize(config_path=inference_cfg_path)
    cfg = compose(config_name="run_inference",     
                  overrides=[
                    f"model={sam_type}",
                    f"name_exp={sam_type}"
                ])

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    cfg.dataset_name = dataset_name
    model.ref_obj_names = cfg.data.datasets[cfg.dataset_name].obj_names
    model.dataset_name = cfg.dataset_name
    
    default_ref_dataloader_config = cfg.data.reference_dataloader
    ref_dataloader_config = default_ref_dataloader_config.copy()
    ref_dataloader_config.template_dir = f"SAM-6D/Data/BOP-Templates/{cfg.dataset_name}/"
    ref_dataset = instantiate(ref_dataloader_config)
    model.ref_dataset = ref_dataset

    # Initialize the HVI model
    HVI_model = CIDNet().cuda()
    model_ckpt = infer_model_cfg['hvi_ckpt']
    state_dict  = sf.load_file(model_ckpt)
    HVI_model.load_state_dict(state_dict, strict=False)
    HVI_model.eval()

    # Initialize the object detection models
    gdino = GroundingDINOObjectPredictor(threshold=0.25, config_file=infer_model_cfg['gdino_cfg'], ckpt_path=infer_model_cfg['gdino_ckpt'])
    Mobile_SAM = SegmentAnythingPredictor()

    # inference
    target_path = os.path.join(os.path.dirname(data_base), "test_targets_bop24.json")
    if os.path.exists(target_path):
        targets = json.load(open(target_path, "r"))
    else:
        raise FileNotFoundError(f"Target file not found: {target_path}")

    target_idx = 0
    for scene_id in tqdm(scene_ids):
        scene_dir = os.path.join(data_base, f"{scene_id:06d}")

        depth_dir = os.path.join(scene_dir, "depth")
        rgb_dir = os.path.join(scene_dir, "gray")
        camera_json_path = os.path.join(scene_dir, "scene_camera.json")
        with open(camera_json_path, "r") as f:
            camera_info = json.load(f)

        im_ids = sorted(os.listdir(depth_dir))
        for img in im_ids:
            im_id = int(os.path.splitext(img)[0])
            if not any(item["im_id"] == im_id and item["scene_id"] == scene_id for item in targets):
                continue

            cam_K = np.array(camera_info[str(im_id)]["cam_K"]).reshape(3, 3)
            rgb_path = os.path.join(rgb_dir, img)
            rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            depth_path = os.path.join(depth_dir, img)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) * 0.1

            # compute the time for preprocess
            start_time = time.time()

            # Apply image enhancement for the rgb image
            if len(rgb.shape) == 2:  # if grayscale
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

            # Apply image enhancement for the cropped rgb image
            enhanced_rgb = image_aug(HVI_model, rgb)

            res = crop_frame(gdino, Mobile_SAM, enhanced_rgb, scale=1.0)
            
            # # visualize the results
            # image_pil = res['image']
            # masks = res['masks']
            # phrases = res['phrases']
            # image_pil_bboxes = res['bboxes']
            # gdino_conf = res['gdino_conf']
            # bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)
            # # Save the image
            # save_path = os.path.join(output_base, f"{scene_id:06d}_{im_id:06d}.png")
            # bbox_annotated_pil.save(save_path)
            # continue

            # Crop the image based on the bounding box
            enhanced_cropped_rgb, cropped_depth, mask_frame, bbox = crop_image(res, enhanced_rgb, depth)
        
            # Update the cam_K
            x1, y1, x2, y2 = bbox
            update_cam_K = update_intrinsics(cam_K, x1, y1)
    
            # Apply the Instance Segmentation Model
            batch = {}
            # rgb to (0,1)
            enhanced_cropped_rgb = enhanced_cropped_rgb.astype(np.float32) / 255.0
            batch['image'] = torch.from_numpy(enhanced_cropped_rgb).permute(2, 0, 1).unsqueeze(0).float().cuda()
            batch['depth'] = torch.from_numpy(cropped_depth).unsqueeze(0).float().cuda() / 0.1
            batch['cam_intrinsic'] = torch.from_numpy(update_cam_K).unsqueeze(0).float().cuda()
            batch['depth_scale'] = torch.tensor([0.1], device="cuda")
            batch['bbox'] = bbox
            batch['original_rgb'] = rgb
            batch['scene_id'] = [f"{scene_id:06d}"]
            batch['frame_id'] = [im_id]

            end_time = time.time()
            preprecss_stage_time = end_time - start_time

            model.forward(batch, target_idx, enable_vis=infer_dataset_cfg["enable_vis"], preprecss_stage_time=preprecss_stage_time)
            target_idx += 1
    
    model.forward_end()


            


    