import os
import numpy as np
import torch
from PIL import Image
import glob
import cv2
from skimage.feature import canny
from skimage.morphology import binary_dilation
from omegaconf import OmegaConf
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad
from Instance_Segmentation_Model.model.utils import Detections
import colorsys

def label_to_color(obj_id: str) -> tuple[int, int, int]:
    """
    Given a label string, return an RGB tuple with
    maximally distinct colors by stepping the hue
    by the golden ratio conjugate each time.
    """
    # Convert your ID to an integer
    idx = int(obj_id)

    # Golden ratio conjugate ensures even spacing
    golden_ratio_conjugate = 0.618033988749895

    # Compute hue in [0,1)
    h = (idx * golden_ratio_conjugate) % 1.0

    # Optionally vary saturation/value slightly per-ID
    # to avoid too‐washed‐out or too‐dark colors:
    s = 0.5 + (idx * 0.13) % 0.5  # between 0.5 and 1.0
    v = 0.7 + (idx * 0.17) % 0.3  # between 0.7 and 1.0

    # Convert to RGB 0–1 floats, then to 0–255 ints
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def visualize_mask(rgb, masks, save_dir):
    rgb = Image.fromarray(rgb)
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    for idx, item in enumerate(masks):
        tmp = img.copy()
        mask = item['mask'] 
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))

        tmp[mask.astype(bool)] = (255, 0, 0)     # Red fill
        tmp[edge.astype(bool)] = (255, 255, 255) # White edges

        save_path = os.path.join(save_dir, f"mask_{idx}.png")
        img_pil = Image.fromarray(tmp)
        img_pil.save(save_path)
        
def visualize_masks(rgb, masks):
    rgb = Image.fromarray(rgb)
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    for item in masks:
        mask = item['mask'] 
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))

        img[mask.astype(bool)] = (255, 0, 0)     # Red fill
        img[edge.astype(bool)] = (255, 255, 255) # White edges

    img_pil = Image.fromarray(img)
    # Create side-by-side image
    concat = Image.new('RGB', (img_pil.width + rgb.width, img_pil.height))
    concat.paste(rgb, (0, 0))
    concat.paste(img_pil, (rgb.width, 0))
    return concat

def visualize_seg(rgb, seg, save_path):
    # Ensure rgb is a NumPy array (BGR for OpenCV)
    if isinstance(rgb, np.ndarray) and rgb.shape[2] == 3:
        rgb_vis = rgb.copy()
    else:
        raise ValueError("Input rgb must be a 3-channel NumPy array.")

    # check if rgb is 0-255, if not normalize to 0-255
    if rgb_vis.dtype != np.uint8:
        rgb_vis = (rgb_vis * 255).astype(np.uint8)                              
    
    # Convert to grayscale and back to 3-channel RGB for annotation
    gray = cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # build list of (area, obj_id, mask)
    all_masks = []
    for obj_id, masks_list in seg.items():
        for mask in masks_list:
            area = mask.sum()
            all_masks.append((area, obj_id, mask))

    # sort descending by area
    all_masks.sort(key=lambda t: t[0], reverse=True)

    # drop the largest one
    to_draw = all_masks[1:]

    for area, obj_id, mask in to_draw:
        obj_color = label_to_color(str(obj_id))          
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))

        img[mask.astype(bool)] = obj_color      # fill
        img[edge.astype(bool)] = (255, 255, 255)  # outline

    img_with_text = img.copy()

    # Resize if needed
    if rgb_vis.shape[0] != img_with_text.shape[0]:
        target_height = min(rgb_vis.shape[0], img_with_text.shape[0])
        rgb_vis = cv2.resize(rgb_vis, (int(rgb_vis.shape[1] * target_height / rgb_vis.shape[0]), target_height))
        img_with_text = cv2.resize(img_with_text, (int(img_with_text.shape[1] * target_height / img_with_text.shape[0]), target_height))

    # Side-by-side visualization
    concat = np.hstack((rgb_vis, img_with_text))

    # make the image smaller
    if concat.shape[1] > 1080:
        scale_factor = 1080 / concat.shape[1]
        new_size = (int(concat.shape[1] * scale_factor), int(concat.shape[0] * scale_factor))
        concat = cv2.resize(concat, new_size)

    cv2.imwrite(save_path, concat)

def SAM_rgb(model, rgb):
    detections = model.segmentor_model.generate_masks(rgb)
    detections = Detections(detections)
    # remove the very small masks
    mask_sums = detections.masks.sum(dim=(1, 2))
    idx_selected_proposals = mask_sums > 100
    detections.filter(idx_selected_proposals)

    return detections

def get_detections(model, detections, template_dir, mesh, cam_K, depth_scale, rgb, depth, device):
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data
    
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(rgb, detections)

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    # compute the geometric score
    batch = {}
    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)

    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
        
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)
    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )

    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
    
    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))
    detections.apply_nms_per_object_id(
        nms_thresh=0.25
    )      
    detections.to_numpy()

    return detections

