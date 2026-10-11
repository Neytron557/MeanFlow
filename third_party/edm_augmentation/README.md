# MeanFlow / EDM augmentation attribution

`augment.py` is an unmodified copy of
[Gsunshine/py-meanflow, meanflow/models/augment.py](https://github.com/Gsunshine/py-meanflow/blob/1f6d72d94247c8fdeb89489acca1a8007a6baf6c/meanflow/models/augment.py),
pinned to commit `1f6d72d94247c8fdeb89489acca1a8007a6baf6c` (retrieved 2026-09-11).
Its SHA-256 is `092466c67c2b7f67938ec8071bac5b21a1712eed0c920e48f6b293ff48dc6fc9`.

The file identifies [NVIDIA's EDM](https://github.com/NVlabs/edm) as its original
source, modified by the MeanFlow project. Its original header is preserved.
It is licensed under **Creative Commons Attribution–NonCommercial–ShareAlike
4.0 International**, independently of this repository's MIT license. The full
upstream license texts are included as `LICENSE.edm` and `LICENSE.py-meanflow`.
`LICENSE.py-meanflow` is from the same pinned commit. `LICENSE.edm` is copied from
`https://raw.githubusercontent.com/NVlabs/edm/008a4e5316c8e3bfe61a62f874bddba254295afb/LICENSE.txt`
on 2026-09-11 (pinned EDM commit `008a4e5316c8e3bfe61a62f874bddba254295afb`);
its SHA-256 is `b7249043cca5e5c88f99730ddc40490a3284780dfe423b19f2c75bd5e2a8f03e`.

The new wrapper in `cifar10/augmentation.py` selects the official CIFAR-10
configuration from `meanflow/training/train_loop.py` at the pinned MeanFlow
commit: `p=0.12`, `xflip=1e8`, `yflip=0`, `scale=1`, `rotate_frac=0`, `aniso=1`,
`translate_frac=1`. The unconditioned base horizontal flip matches that commit's
`meanflow/training/data_transform.py`, moved into the training process for RNG
checkpoint reproducibility. The vendored resampling, reflection padding,
Symlet-6 filters, and augmentation-label definitions have not been altered.
