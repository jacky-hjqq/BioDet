import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import cv2
from PIL import Image

pil2tensor = transforms.Compose([transforms.ToTensor()])

def adaptive_brightness(image_path, brightness_factor=1.5, threshold=100, smoothing=5):
    """
    Selectively increase brightness in dark regions while preserving bright areas.
    
    Parameters:
    image_path (str): Path to the input image
    brightness_factor (float): Factor to increase brightness in dark areas
    threshold (int): Pixel value threshold to identify dark areas (0-255)
    smoothing (int): Size of Gaussian kernel for smoothing the mask
    
    Returns:
    Enhanced image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Create a mask for dark areas
    dark_mask = np.where(v < threshold, 1, 0).astype(np.float32)
    
    # Smooth the mask to create a gradual transition
    if smoothing > 0:
        dark_mask = cv2.GaussianBlur(dark_mask, (smoothing*2+1, smoothing*2+1), 0)
    
    # Calculate adjustment for each pixel
    # Dark areas get brightness_factor, bright areas get 1.0 (no change)
    adjustment = dark_mask * (brightness_factor - 1.0) + 1.0
    
    # Apply adjustment to value channel
    v_enhanced = np.clip(v * adjustment, 0, 255).astype(np.uint8)
    
    # Merge channels back and convert to BGR
    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    enhanced_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    return enhanced_img

def reduce_overexposure(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to LAB color space for better brightness adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Reduce brightness in overexposed areas with a non-linear function
    # Identify overexposed areas (very bright pixels)
    _, overexposed_mask = cv2.threshold(l, 220, 255, cv2.THRESH_BINARY)
    
    # Create a gamma correction for bright areas
    gamma = 1.5  # Darken bright areas with gamma > 1
    l_gamma = np.power(l_clahe / 255.0, gamma) * 255.0
    l_gamma = l_gamma.astype(np.uint8)
    
    # Apply gamma correction only to bright areas, proportional to brightness
    alpha = overexposed_mask / 255.0  # Normalization to [0,1]
    l_result = l_clahe * (1 - alpha) + l_gamma * alpha
    l_result = l_result.astype(np.uint8)
    
    # Merge with the a and b channels
    lab_result = cv2.merge((l_result, a, b))
    
    # Convert back to BGR color space
    result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
    
    # Reduce overall brightness slightly
    brightness_factor = 0.85
    result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
    
    # Add a slight contrast enhancement
    contrast_factor = 1.15
    mean_val = np.mean(result)
    output_img = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=mean_val * (1 - contrast_factor))
    
    return output_img
    

def enhance_image(model, img_path, alpha_s=1.0, alpha_i=1.0, gamma=1.0, mean_intensity=None):
    img = Image.open(img_path).convert('RGB')
    input = pil2tensor(img)

    # get the original size
    orig_h, orig_w = img.size

    # resize the image to be divisible by 2
    input = input.unsqueeze(0) 
    resized_input = F.interpolate(input, scale_factor=1/2, mode='bilinear', align_corners=False)
    resized_input = resized_input.squeeze(0)  

    factor = 8
    h, w = resized_input.shape[1], resized_input.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    resized_input = F.pad(resized_input.unsqueeze(0), (0,padw,0,padh), 'reflect')
    with torch.no_grad():
        model.trans.alpha_s = alpha_s
        model.trans.alpha = alpha_i
        output = model(resized_input.cuda()**gamma)
            
    output = torch.clamp(output.cuda(),0,1).cuda()
    output = output[:, :, :h, :w]

    # Resize output back to original size
    output_recover = F.interpolate(output, size=(orig_w, orig_h), mode='bilinear', align_corners=False)
    output_recover = output_recover.squeeze(0).permute(1,2,0).cpu().numpy()
    output_img = (output_recover*255).astype(np.uint8)
    output_mean_intensity = np.mean(output_img)
    
    ADD_Contrast = True
    if output_mean_intensity > 2 * mean_intensity and mean_intensity < 10:
        ADD_Contrast = False

    if output_mean_intensity > 2 * mean_intensity and mean_intensity > 10:
        # in this case most likeliy to have overexposured, so the orignal image is better
        output_img = adaptive_brightness(img_path)

    if ADD_Contrast:
        # Apply CLAHE
        lab = cv2.cvtColor(output_img, cv2.COLOR_RGB2LAB)  # Convert directly from RGB to LAB
        l, a, b = cv2.split(lab)  # Split into L, A, B channels
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Apply CLAHE to L channel only
        l_clahe = clahe.apply(l)
        lab_result = cv2.merge((l_clahe, a, b))  # Merge channels back
        output_img = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)  # Convert back to BGR

    return output_img
