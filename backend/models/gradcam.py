import torch
import numpy as np
import cv2


class SwinGradCAM:
    def __init__(self, model, target_layer, grid_size=(7, 7)):
        """
        model: End-to-end multimodal model.
        target_layer: Final stage/block in the Swin backbone whose spatial
                       feature map should be explained (e.g.
                       model.image_encoder.encoder.layers[-1]).
        grid_size: Fallback (H_patches, W_patches) used only when the target
                   layer's raw forward output doesn't carry its own spatial
                   dimensions. A Swinv2Stage reports its own (height, width)
                   per forward call as part of its output tuple, and that is
                   always preferred over this fixed guess -- input image size
                   or backbone variant changing the true grid (e.g. 8x8 for a
                   256x256 SwinV2 input, not the 7x7 you'd get at 224x224)
                   used to silently corrupt the reshape instead of erroring.
        """
        self.model = model
        self.target_layer = target_layer
        self.grid_size = grid_size
        self.gradients = None
        self.activations = None
        self._activation_grid = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _reshape_patch_tensor(self, tensor, grid_size=None):
        """
        Transforms Swin patch sequences into standard CNN feature maps: [B, C, H, W]
        """
        grid_size = grid_size or self.grid_size

        # Case 1: Tensor is 3D [Batch, N_patches, Channels]
        if tensor.dim() == 3:
            B, N, C = tensor.shape

            # Remove [CLS] token if model variant prepends one
            if N == (grid_size[0] * grid_size[1]) + 1:
                tensor = tensor[:, 1:, :]
                N -= 1

            H, W = grid_size
            # Reshape [B, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
            tensor = tensor.reshape(B, H, W, C)
            tensor = tensor.permute(0, 3, 1, 2)

        # Case 2: Tensor is 4D [Batch, H, W, Channels] (Native Swin output)
        elif tensor.dim() == 4 and tensor.shape[1] != tensor.shape[3]:
            # Permute [B, H, W, C] -> [B, C, H, W]
            tensor = tensor.permute(0, 3, 1, 2)

        return tensor

    @staticmethod
    def _extract_grid_size(output):
        """
        Swinv2Stage.forward returns (hidden_states, hidden_states_before_downsampling,
        output_dimensions), where output_dimensions is
        (height, width, height_downsampled, width_downsampled) -- the first two
        entries are the resolution of hidden_states itself. Any other module's
        output (e.g. the dummy test layer's plain tensor) falls through to None,
        so the caller-supplied grid_size is used instead.
        """
        if not (isinstance(output, tuple) and len(output) >= 3):
            return None
        dims = output[2]
        if not (isinstance(dims, tuple) and len(dims) == 4):
            return None
        try:
            return int(dims[0]), int(dims[1])
        except (TypeError, ValueError):
            return None

    def _save_activations(self, module, input, output):
        grid = self._extract_grid_size(output)
        # Handle tuple outputs from intermediate blocks
        if isinstance(output, tuple):
            output = output[0]
        self._activation_grid = grid
        self.activations = self._reshape_patch_tensor(output, grid)

    def _save_gradients(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if isinstance(grad, tuple):
            grad = grad[0]
        self.gradients = self._reshape_patch_tensor(grad, self._activation_grid)

    def generate_heatmap(self, image_tensor, text_tensor, lab_tensor, target_class=None, **model_kwargs):
        self.model.eval()
        self.model.zero_grad()

        if not image_tensor.requires_grad:
            raise ValueError(
                "image_tensor.requires_grad is False -- Grad-CAM needs gradients "
                "w.r.t. the pixel input. A frozen (requires_grad=False) backbone "
                "alone won't build an autograd graph back to the target layer; "
                "call image_tensor.requires_grad_(True) before this, and don't "
                "run it inside torch.no_grad()."
            )

        # Forward pass
        with torch.enable_grad():
            output = self.model(image_tensor, text_tensor, lab_tensor, **model_kwargs)
            if isinstance(output, tuple):
                output = output[0]

            if target_class is None:
                target_class = torch.argmax(output, dim=1).item() if output.shape[1] > 1 else 0

            score = output[0, target_class]
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks on the target layer never fired. Check that "
                "target_layer is actually reached during the forward pass "
                "(e.g. it isn't behind an unused branch)."
            )

        # Extract transformed activations and gradients [Channels, H, W]
        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()

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
    Overlays the Grad-CAM heatmap onto the original radiological image.
    Returns an RGB uint8 array (both inputs/output treated as RGB throughout --
    cv2.applyColorMap produces BGR, so it's converted back before returning).
    """
    heatmap_resized = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    overlaid_image = cv2.addWeighted(heatmap_colored, alpha, original_image_np, 1 - alpha, 0)
    return overlaid_image, heatmap_resized
