import torch
import cv2
import numpy as np
from PIL import Image
from Instance_Segmentation_Model.model.utils import Detections
from Instance_Segmentation_Model.preprocess.image_enhancement.utils import reduce_overexposure, enhance_image

def bbox_area(bbox):
    width = bbox[2] - bbox[0]     
    height = bbox[3] - bbox[1]   
    area = width * height
    return int(area)

def get_masks(detections):
    save_masks = []
    masks = detections.masks
    scores = detections.scores
    for score, mask in zip(scores, masks):
        save_masks.append({
        "mask": mask,
        "score": score
        })  
    return save_masks

def recover_resize(detections, downscale_ratio: float) -> Detections:
    """
    Recover the masks and boxes from a detection that was run
    on an image downscaled by downscale_ratio.
    
    :param detections: Detections instance containing masks, scores, boxes, object_ids.
                       Assumes these were produced on the *downscaled* image.
    :param downscale_ratio: The factor you used to resize the *original* image
                            before detection (e.g. 0.5 to go from 1080→540).
    :return: A new Detections instance where masks and boxes have been remapped
             back to the original image’s coordinate space.
    """
    # Compute the up‐scale factor
    upscale = 1.0 / downscale_ratio

    # Unpack
    masks       = detections.masks      # list or array of H'×W'
    boxes       = detections.boxes      # list of [x1, y1, x2, y2] in resized coords
    scores      = detections.scores
    object_ids  = detections.object_ids

    # 1) Upscale each mask back to original H×W
    masks_recovered = [
        cv2.resize(mask,
                   dsize=None,
                   fx=upscale,
                   fy=upscale,
                   interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]

    # 2) Scale box coordinates back
    boxes_recovered = [
        [int(coord * upscale) for coord in box]
        for box in boxes
    ]

    # Package into a new Detections
    return Detections({
        "masks":      np.array(masks_recovered),
        "scores":     np.array(scores),
        "boxes":      np.array(boxes_recovered),
        "object_ids": np.array(object_ids)
    })

def recover_masks(detections, crop_bbox, rgb):
    x1, y1, x2, y2 = map(int, crop_bbox)
    crop_h, crop_w = (y2 - y1, x2 - x1)
    H, W = rgb.shape[:2]

    masks_list     = []
    scores_list    = []
    boxes_list     = []
    object_ids_list= []

    for idx in range(len(detections)):
        # 1) unpack
        mask_crop = detections.masks[idx]       # (h', w')
        score     = detections.scores[idx]
        box_crop  = detections.boxes[idx]       # [bx1,by1,bx2,by2]
        obj_id    = detections.object_ids[idx]

        # 2) resize mask back into the cropped‐region size
        mask_resized = cv2.resize(
            mask_crop.astype(np.uint8),
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST
        )

        # 3) paste into full‐frame
        full_mask = np.zeros((H, W), dtype=mask_resized.dtype)
        full_mask[y1:y2, x1:x2] = mask_resized

        # 4) recover full‐frame box
        bx1, by1, bx2, by2 = box_crop
        full_box = [
            int(bx1) + x1,
            int(by1) + y1,
            int(bx2) + x1,
            int(by2) + y1,
        ]

        # 5) collect
        masks_list.append(full_mask)
        scores_list.append(score)
        boxes_list.append(full_box)
        object_ids_list.append(obj_id)

    # 6) pack into the flat dict that Detections wants
    data = {
        "masks":   np.stack(masks_list, axis=0),       # (N, H, W)
        "scores":  np.array(scores_list, dtype=float), # (N,)
        "boxes":   np.array(boxes_list, dtype=int),    # (N,4)
        "object_ids": np.array(object_ids_list)        # (N,)
    }

    # 7) return a Detections instance
    return Detections(data)

def crop_frame(gdino, SAM, image, scale=0.5):
    text_prompt = 'Parts frame where multiple parts inside it'
    # text_prompt = 'Parts frame where multiple parts on it and fixed by the robot arm'

    try:
        image_pil = Image.fromarray(image).convert("RGB")

        # downsample the image by scale
        image_pil = image_pil.resize((int(image_pil.width * scale), int(image_pil.height * scale)), Image.BILINEAR)

        with torch.no_grad():
            # logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
            bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

            # logging.info("GDINO post processing")
            w, h = image_pil.size # Get image width and height 
            # Scale bounding boxes to match the original image size
            image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

            # logging.info("SAM prediction")
            image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)
        
        # if multi_mask, only save the one with the highest confidence score
        if len(masks) > 1:
            # # Filter items with phrase "parts frame"
            # filtered_by_phrase = [i for i in range(len(phrases)) if phrases[i].lower() == "parts frame"]

            # if len(filtered_by_phrase) > 0:
            #     masks = masks[filtered_by_phrase]
            #     image_pil_bboxes = image_pil_bboxes[filtered_by_phrase]
            #     phrases = [phrases[i] for i in filtered_by_phrase]
            #     gdino_conf = gdino_conf[filtered_by_phrase]

            #     # Filter items by bbox area (between 2% and 90% of image area)
            #     image_area = image_pil.size[0] * image_pil.size[1]
            #     filtered_by_area = [
            #         i for i in range(len(image_pil_bboxes))
            #         if 0.02 * image_area < bbox_area(image_pil_bboxes[i]) < 0.9 * image_area
            #     ]

            #     if len(filtered_by_area) > 0:
            #         masks = masks[filtered_by_area]
            #         image_pil_bboxes = image_pil_bboxes[filtered_by_area]
            #         phrases = [phrases[i] for i in filtered_by_area]
            #         gdino_conf = gdino_conf[filtered_by_area]

            # Select the item with highest confidence
            max_conf_idx = gdino_conf.argmax()
            masks = masks[max_conf_idx:max_conf_idx + 1]
            image_pil_bboxes = image_pil_bboxes[max_conf_idx:max_conf_idx + 1]
            phrases = phrases[max_conf_idx:max_conf_idx + 1]
            gdino_conf = gdino_conf[max_conf_idx:max_conf_idx + 1]
        
        output = {
            "image": image_pil,
            "bboxes": image_pil_bboxes,
            "masks": masks,
            "phrases": phrases,
            "gdino_conf": gdino_conf
        }
        
        return output
    
    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")
        return None
    
def update_intrinsics(cam_K, x1, y1):
    # Create a copy of the original matrix
    new_cam_K = cam_K.copy()
    
    # Adjust the principal point coordinates
    new_cam_K[0, 2] = cam_K[0, 2] - x1  # cx = original cx - x1
    new_cam_K[1, 2] = cam_K[1, 2] - y1  # cy = original cy - y1
    
    return new_cam_K


def process_image(HVI_model, img_path):
    # Read the image in grayscale
    gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Calculate mean intensity
    mean_intensity = np.mean(gray_image)
    # Set alpha_i and gamm based on mean intensity
    if mean_intensity > 50:
        enhanced_image = reduce_overexposure(img_path)
    else:
        if mean_intensity < 20:
            alpha_s = 1.0
            alpha_i = 1.0 
            gamma = 1.0
        elif mean_intensity > 20 and mean_intensity < 40:
            alpha_s = 1.0
            alpha_i = 0.5
            gamma = 2.5
        else:
            alpha_s = 1.0
            alpha_i = 0.1
            gamma = 2.5
        enhanced_image = enhance_image(HVI_model, img_path, alpha_s, alpha_i, gamma, mean_intensity)
    
    return enhanced_image

def select_cam_view(gdino, SAM, enhanced_images, scale):
    gdino_res = []
    for idx, cam_img in enumerate(enhanced_images):
        # Perform inference
        result = crop_frame(gdino, SAM, cam_img, scale=scale)

        if result is None:
            print(f"Error processing image from camera {idx}. Skipping...")
            continue

        gdino_res.append({
            'idx': idx,
            'result': result
        })

    # remove the result without the phrase "parts frame"
    res = [res for res in gdino_res if res['result']['phrases'][0].lower() == "parts frame"]
    #  remove the result with too large or too small bboxes
    res = [res for res in res if 0.02 * res['result']['image'].size[0] * res['result']['image'].size[1] < bbox_area(res['result']['bboxes'][0]) < 0.9 * res['result']['image'].size[0] * res['result']['image'].size[1]]

    # check if there are any results left
    if len(res) == 0:
        # chooose the one with the higher confidence score
        best_result = max(gdino_res, key=lambda x: x['result']['gdino_conf'].max())
    else:
        # choose the one with the higher confidence score
        best_result = max(res, key=lambda x: x['result']['gdino_conf'].max())
    
    return best_result

def crop_image(best_result, rgb, depth):
    bbox_frame = best_result['result']['bboxes'][0]
    bbox_frame_score = best_result['result']['gdino_conf'][0] 
    mask_frame = best_result['result']['masks'][0].cpu().numpy().astype(np.uint8)
    # Crop the image based on the bounding box
    x1, y1, x2, y2 = bbox_frame
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    # Crop the mask frame
    if bbox_frame_score > 0.5:
        mask_frame = mask_frame[0][y1:y2, x1:x2]
    else:   
        mask_frame = None

    x1 = int(x1 * 2)
    y1 = int(y1 * 2)
    x2 = int(x2 * 2)
    y2 = int(y2 * 2)

    cropped_rgb = rgb[y1:y2, x1:x2]  
    cropped_depth = depth[y1:y2, x1:x2]

    bbox = [x1, y1, x2, y2]
    
    return cropped_rgb, cropped_depth, mask_frame, bbox