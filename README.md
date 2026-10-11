# MeanFlow on CIFAR-10

Unconditional 32×32 image generation with MeanFlow and a DiT-B/4 transformer.
The implementation includes the forward-mode JVP objective, indexed training
dropout, exponential moving averages, distributed training, complete resume
checkpoints, and one-step sampling with FID evaluation.

The provided configuration targets **four NVIDIA GeForce RTX 2080 GPUs**.
It starts a fresh run; pretrained checkpoints, datasets, and evaluation assets
are not bundled.

## Setup

Use Python 3.11 and a CUDA-capable PyTorch installation compatible with your
driver. The dependency files pin PyTorch 2.7.1 and torchvision 0.22.1.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cifar10.txt
python -m cifar10.data --data-dir data/cifar10
```

The data command downloads and verifies CIFAR-10. Training uses all 50,000
training images and discards class labels.

## Training

Run the CPU wiring check, then the full-model four-GPU preflight before
starting a long run:

```bash
bash scripts/smoke_cifar10_cpu.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/preflight_cifar10_4gpu.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_cifar10_4gpu.sh
```

The preflight exercises synthetic images with the production model, objective,
optimizer, and checkpointing. Validate memory availability and finite updates
on your server; the target configuration is not a measured memory guarantee.
Scripts use `.venv/bin/python` by default; set `PYTHON` to use another interpreter.

The recipe in [configs/cifar10_4x2080.json](configs/cifar10_4x2080.json) uses:

| Setting | Value |
|---|---|
| Global batch | 8 images × 32 accumulation steps × 4 GPUs = 1,024 |
| Prediction / JVP | FP16, with FP32 loss and finite checks |
| Optimizer | Reference Adam, learning rate 0.0002, betas 0.9 / 0.95 |
| Time sampler / dropout | v1 / 0.1 |
| Training length | 600,000 successful updates; 5,000-update warmup |
| Checkpoints | Every 5,000 updates; keep the latest three |

Logs, loss plots, and checkpoints are written under
`work_dir/cifar10_4x2080/`. Each complete checkpoint contains online and EMA
weights, optimizer/scheduler state, and per-rank random-generator/data state.

For a graceful stop and subsequent resume with the same four-GPU configuration:

```bash
touch work_dir/cifar10_4x2080/STOP
# Wait for training to finish saving its complete checkpoint.
rm work_dir/cifar10_4x2080/STOP
bash scripts/train_cifar10_4gpu.sh --resume latest
```

## Sampling and evaluation

Install the evaluation dependencies and download the Inception evaluation
weights once:

```bash
python -m pip install -r requirements-eval.txt
export TORCH_HOME="$PWD/.cache/torch"
python evaluate.py --prepare-metrics
```

After the configured run reaches 600,000 updates, evaluate its long EMA:

```bash
python evaluate.py \
  --ckpt work_dir/cifar10_4x2080/checkpoints/step-00600000/weights.pt \
  --weights ema_long --data-dir data/cifar10 \
  --sample-dir samples/cifar10-600000-ema-long \
  --num-fid-samples 50000 --num-steps 1 --nfe 1 \
  --seed 0 --compute-metrics
```

Choose an existing checkpoint if training stops earlier, and use an empty sample
directory for each evaluation. The one-step protocol uses the average field,
Euler integration from 1 to 0, FP32 sampling, and zero augmentation conditioning.
FID compares 50,000 generated images with the 50,000 unaugmented CIFAR-10
training images. The evaluator saves the metric and sampling metadata together.

## Tests

Install the development dependencies and run the core test suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q -m "not slow"
```

## References and licenses

The objective follows [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)
and [the reference MeanFlow implementation](https://github.com/Gsunshine/py-meanflow).
The transformer follows the [SiT architecture](https://github.com/willisma/SiT)
with canonical DiT-B/4 dimensions.

See [LICENSE](LICENSE) for the MIT license covering the project code.
The vendored EDM augmentation code has a separate **CC BY-NC-SA 4.0** license;
its source attribution and license texts are preserved in
[third_party/edm_augmentation](third_party/edm_augmentation/README.md).
