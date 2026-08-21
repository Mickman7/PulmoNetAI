import pytest
import torch
import torch.nn as nn
import numpy as np
from backend.models.gradcam import SwinGradCAM


class DummySwinLayer(nn.Module):
    def __init__(self, channels=64, num_patches=49):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_patches = num_patches

    def forward(self, x):
        # x is [B, C, H, W]
        x = self.conv(x)
        B, C, H, W = x.shape
        # Return sequence output like Swin Transformer: [B, H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return x


class DummyMultimodalModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.swin_layer = DummySwinLayer(channels=64, num_patches=49)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, image_tensor, text_tensor, lab_tensor):
        # Fake forward pass through Swin target layer
        swin_out = self.swin_layer(image_tensor)  # [B, 49, 64]
        pooled = swin_out.mean(dim=1)             # [B, 64]
        out = self.fc(pooled)                     # [B, num_classes]
        return out


@pytest.fixture
def test_setup():
    model = DummyMultimodalModel(num_classes=2)
    target_layer = model.swin_layer
    gradcam = SwinGradCAM(model, target_layer, grid_size=(7, 7))

    # Synthetic batch inputs (B=1)
    image_tensor = torch.randn(1, 64, 7, 7, requires_grad=True)
    text_tensor = torch.randn(1, 128)
    lab_tensor = torch.randn(1, 5)

    return gradcam, image_tensor, text_tensor, lab_tensor


def test_tensor_reshape(test_setup):
    gradcam, image_tensor, text_tensor, lab_tensor = test_setup
    
    # Sequence tensor [1, 49, 64]
    dummy_seq = torch.randn(1, 49, 64)
    reshaped = gradcam._reshape_patch_tensor(dummy_seq)

    # Check transformed output is [B, C, H, W]
    assert reshaped.shape == (1, 64, 7, 7)


def test_gradcam_hooks_and_heatmap_generation(test_setup):
    gradcam, image_tensor, text_tensor, lab_tensor = test_setup

    heatmap, predicted_class = gradcam.generate_heatmap(
        image_tensor, text_tensor, lab_tensor, target_class=0
    )

    # Validate output types and shapes
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (7, 7)
    assert predicted_class == 0
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_gradcam_hook_cleanup(test_setup):
    gradcam, _, _, _ = test_setup
    assert gradcam.activations is None
    assert gradcam.gradients is None