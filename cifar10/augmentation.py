"""The MeanFlow CIFAR-10 augmentation recipe, sampled in the training process."""

import torch

from third_party.edm_augmentation.augment import AugmentPipe


AUGMENT_DIM = 6
_PAPER_AUGMENT = AugmentPipe(
    p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0, aniso=1, translate_frac=1
)


@torch.no_grad()
def augment_batch(images: torch.Tensor, enabled: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return augmented RGB images and six paper augmentation conditions.

    The base 50% horizontal flip is unconditioned, matching the official dataset
    transform. A second flip inside AugmentPipe contributes the first condition;
    the other five are scale, two anisotropy components, and two translations.
    All draws use the device's default Torch RNG so rank RNG checkpointing also
    restores augmentation. Workers only perform deterministic preprocessing.
    """
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32) or images.shape[0] == 0:
        raise ValueError("Augmentation expects a nonempty (N, 3, 32, 32) CIFAR-10 batch")
    if images.dtype != torch.float32:
        raise ValueError("The paper augmentation baseline expects float32 images")
    if not enabled:
        return images, images.new_zeros((images.shape[0], AUGMENT_DIM))
    horizontal_flip = torch.rand((images.shape[0], 1, 1, 1), device=images.device) < 0.5
    images = torch.where(horizontal_flip, images.flip(-1), images)
    augmented, labels = _PAPER_AUGMENT(images)
    return augmented, labels.to(dtype=images.dtype)
