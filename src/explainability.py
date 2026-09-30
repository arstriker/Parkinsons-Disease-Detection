import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_PYTORCH_GRAD_CAM = True
except ImportError:
    HAS_PYTORCH_GRAD_CAM = False


class ExplainableAI:
    """
    Grad-CAM Heatmap Generator targeting layer4[-1] of ResNet-18.
    Highlights stroke tremor regions driving the Parkinson's prediction.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        # Target layer for ResNet-18
        if hasattr(model, 'resnet'):
            self.target_layers = [model.resnet.layer4[-1]]
        elif hasattr(model, 'layer4'):
            self.target_layers = [model.layer4[-1]]
        else:
            raise ValueError("Target layer 'layer4' not found in model architecture.")

    def generate_heatmap(self, input_tensor, rgb_img, target_category=None):
        """
        Generates Grad-CAM heatmap overlaid on original RGB image.

        Args:
            input_tensor (torch.Tensor): Shape (1, 3, 224, 224) or (3, 224, 224)
            rgb_img (np.ndarray): Shape (224, 224, 3) uint8 or float32 image [0, 1]
            target_category (int): Target class index (0 or 1). If None, uses model argmax.

        Returns:
            visualization (np.ndarray): RGB heatmap overlay array (224, 224, 3) uint8
            grayscale_cam (np.ndarray): Normalized 2D Grad-CAM map (224, 224) float32
        """
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        # Normalize RGB image to float [0, 1] for show_cam_on_image
        if rgb_img.dtype == np.uint8:
            rgb_img_float = rgb_img.astype(np.float32) / 255.0
        else:
            rgb_img_float = rgb_img.copy()

        if HAS_PYTORCH_GRAD_CAM:
            cam = GradCAM(model=self.model, target_layers=self.target_layers)
            
            targets = None
            if target_category is not None:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                targets = [ClassifierOutputTarget(target_category)]

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
            return visualization, grayscale_cam
        else:
            # Fallback Custom Grad-CAM implementation
            return self._custom_gradcam(input_tensor, rgb_img_float, target_category)

    def _custom_gradcam(self, input_tensor, rgb_img_float, target_category):
        gradients = []
        activations = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = self.target_layers[0]
        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        output = self.model(input_tensor)
        if target_category is None:
            target_category = output.argmax(dim=1).item()

        score = output[0, target_category]
        self.model.zero_grad()
        score.backward()

        h_fwd.remove()
        h_bwd.remove()

        grad = gradients[0].cpu().data.numpy()[0]
        act = activations[0].cpu().data.numpy()[0]

        weights = np.mean(grad, axis=(1, 2))
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * act, axis=0)
        cam = np.maximum(cam, 0)
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        cam_resized = cv2.resize(cam, (rgb_img_float.shape[1], rgb_img_float.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        visualization = cv2.addWeighted(rgb_img_float, 0.6, heatmap, 0.4, 0)
        visualization = np.uint8(255 * visualization)

        return visualization, cam_resized
