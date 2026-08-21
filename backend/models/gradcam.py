import torch
import torch.nn.functional as F
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class SwinGradCAM:
    def __init__(self, model, target_layer, grid_size=(7, 7)):
        """
        model: End-to-end multimodal model.
        target_layer: Final block or norm layer in the Swin backbone.
        grid_size: Spatial patch dimensions (H_patches, W_patches) at target layer.
                   Default (7, 7) for 224x224 input passing through 4 stages (224 / 32 = 7).
        """
        self.model = model
        self.target_layer = target_layer
        self.grid_size = grid_size
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _reshape_patch_tensor(self, tensor):
        """
        Transforms Swin patch sequences into standard CNN feature maps: [B, C, H, W]
        """
        # Case 1: Tensor is 3D [Batch, N_patches, Channels]
        if tensor.dim() == 3:
            B, N, C = tensor.shape
            
            # Remove [CLS] token if model variant prepends one
            if N == (self.grid_size[0] * self.grid_size[1]) + 1:
                tensor = tensor[:, 1:, :]
                N -= 1
                
            H, W = self.grid_size
            # Reshape [B, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
            tensor = tensor.reshape(B, H, W, C)
            tensor = tensor.permute(0, 3, 1, 2)

        # Case 2: Tensor is 4D [Batch, H, W, Channels] (Native Swin output)
        elif tensor.dim() == 4 and tensor.shape[1] != tensor.shape[3]:
            # Permute [B, H, W, C] -> [B, C, H, W]
            tensor = tensor.permute(0, 3, 1, 2)

        return tensor

    def _save_activations(self, module, input, output):
        # Handle tuple outputs from intermediate blocks
        if isinstance(output, tuple):
            output = output[0]
        self.activations = self._reshape_patch_tensor(output)

    def _save_gradients(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if isinstance(grad, tuple):
            grad = grad[0]
        self.gradients = self._reshape_patch_tensor(grad)

    def generate_heatmap(self, image_tensor, text_tensor, lab_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(image_tensor, text_tensor, lab_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Extract transformed activations and gradients [Channels, H, W]
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        # Channel importance weights via Global Average Pooling
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted sum of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and Normalisation
        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam, target_class


def overlay_heatmap(heatmap, original_image_np, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays the Grad-CAM heatmap onto the original original radiological image.
    """
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))
    
    # Convert heatmap to RGB image
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Overlay heatmap on original image
    overlaid_image = cv2.addWeighted(heatmap_colored, alpha, original_image_np, 1 - alpha, 0)
    return overlaid_image, heatmap_resized