def summarize_attention(attn_weights):
    """Turns raw attention weights into a short interpretable label.
    attn_weights shape: [1, 1, num_patches]"""
    weights = attn_weights.squeeze().detach().cpu()
    max_weight = weights.max().item()
    
    if max_weight > 0.05:
        return "concentrated on specific image regions (suggests a localized finding)"
    else:
        return "spread broadly across the image (suggests a diffuse or less certain finding)"