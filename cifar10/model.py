"""Unconditional pixel-space DiT variants for CIFAR-10 MeanFlow.

The production variant uses the canonical DiT-B/4 dimensions and the upstream
SiT implementation of adaLN-Zero blocks, with MeanFlow's two time embeddings.
"""

from sit import SiT

from .dropout import IndexedDropout


def create_model(
    model_name: str = "DiT-B/4", augment_dim: int = 6, *,
    attn_projection_dropout: float = 0.0,
    mlp_output_dropout: float = 0.0,
    dropout_mask_mode: str = "indexed",
) -> SiT:
    """Build RGB32 DiT-B/4, or the explicitly named test-only tiny fixture."""
    variants = {
        "DiT-B/4": (12, 768, 12),
        "DiT-Tiny/4": (2, 32, 4),
    }
    if model_name not in variants:
        raise ValueError(f"Unknown CIFAR-10 model {model_name!r}; choose from {tuple(variants)}.")
    if augment_dim < 0:
        raise ValueError("augment_dim must be nonnegative.")
    if dropout_mask_mode != "indexed":
        raise ValueError("dropout_mask_mode must be 'indexed'.")
    depth, width, heads = variants[model_name]
    model = SiT(
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=width,
        decoder_hidden_size=width,
        depth=depth,
        num_heads=heads,
        mlp_ratio=4.0,
        num_classes=0,
        class_dropout_prob=0,
        use_cfg=False,
        augment_dim=augment_dim,
        qk_norm=False,
        fused_attn=False,
    )
    sample_shape = (model.pos_embed.shape[1], width)
    for block in model.blocks:
        block.attn.proj_drop = IndexedDropout(attn_projection_dropout, sample_shape)
        block.mlp.drop2 = IndexedDropout(mlp_output_dropout, sample_shape)
    return model
