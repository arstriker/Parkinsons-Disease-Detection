import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class ImagePreprocessor:
    """
    Standardized Image Preprocessing Pipeline for Early Parkinson's Detection.
    Ensures identical processing during training and real-world GUI inference.
    """
    def __init__(self, target_size=(224, 224), use_skeletonize=False):
        self.target_size = target_size
        self.use_skeletonize = use_skeletonize

    def load_image(self, input_source):
        """
        Loads an image from file path, numpy array, or bytes into RGB.
        """
        if isinstance(input_source, str):
            img = cv2.imread(input_source)
            if img is None:
                raise ValueError(f"Failed to load image from path: {input_source}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(input_source, np.ndarray):
            if len(input_source.shape) == 2:
                img_rgb = cv2.cvtColor(input_source, cv2.COLOR_GRAY2RGB)
            elif input_source.shape[2] == 4:
                img_rgb = cv2.cvtColor(input_source, cv2.COLOR_RGBA2RGB)
            elif input_source.shape[2] == 3:
                img_rgb = input_source.copy()
            else:
                raise ValueError("Unsupported array shape for image processing.")
        else:
            raise TypeError("input_source must be a file path string or numpy array.")
        return img_rgb

    def preprocess_pipeline(self, input_source):
        """
        Executes full preprocessing pipeline:
        1. RGB conversion
        2. Grayscale conversion
        3. Denoising (Gaussian Blur 3x3 / fastNlMeansDenoising)
        4. Adaptive Thresholding (Otsu fallback)
        5. Stroke extraction (morphology / skeletonization)
        6. Resizing to target_size (224x224)
        
        Returns:
            processed_rgb (np.ndarray): 224x224x3 uint8 array for display/GradCAM
            tensor (torch.Tensor): 3x224x224 float32 PyTorch tensor normalized for ResNet
            metrics (dict): quantitative stroke smoothness metrics
        """
        rgb_img = self.load_image(input_source)
        
        # Step 2: Grayscale Conversion
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
        # Step 3: Denoising
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Step 4: Adaptive Thresholding with Otsu fallback
        try:
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        except Exception:
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
        # Step 5: Stroke extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        if self.use_skeletonize:
            bool_img = morphed > 0
            skel = skeletonize(bool_img)
            stroke_img = (skel * 255).astype(np.uint8)
        else:
            stroke_img = morphed

        # Compute quantitative stroke smoothness metric (for report)
        stroke_metrics = self._calculate_stroke_metrics(stroke_img)

        # Step 6: Normalization & Resizing
        resized_stroke = cv2.resize(stroke_img, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Channel-replicated 3-channel image
        processed_rgb = cv2.cvtColor(resized_stroke, cv2.COLOR_GRAY2RGB)
        
        # Scale to [0, 1]
        norm_img = processed_rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        norm_img = (norm_img - IMAGENET_MEAN) / IMAGENET_STD
        
        # Convert to PyTorch Tensor (C, H, W)
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
        
        return processed_rgb, tensor, stroke_metrics

    def _calculate_stroke_metrics(self, binary_img):
        """
        Calculates stroke-consistency quantitative features (contour deviation & tremor index).
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {"stroke_count": 0, "smoothness_score": 0.0, "contour_perimeter": 0.0}
        
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Smooth polygon approximation
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Ratio of perimeter to approximate vertices deviation
        complexity_ratio = len(approx) / (perimeter + 1e-5) * 100
        smoothness_score = max(0.0, min(100.0, 100.0 - complexity_ratio * 10.0))
        
        return {
            "stroke_count": len(contours),
            "smoothness_score": round(float(smoothness_score), 2),
            "contour_perimeter": round(float(perimeter), 2),
            "vertices_count": len(approx)
        }


def preprocess_image(input_source, target_size=(224, 224)):
    preprocessor = ImagePreprocessor(target_size=target_size)
    return preprocessor.preprocess_pipeline(input_source)
