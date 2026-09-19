# Sigmoid Fitting and MLP Simulation (Single Hidden Layer, Quantized Device Response)

Code accompanying the manuscript on Pt/Co-Re analog memristor devices. This version extracts and fits the sigmoid-like transfer characteristics measured from the Pt/Co-Re devices, and evaluates them as hidden-layer activation functions in a single-hidden-layer multilayer perceptron (MLP), using only the experimentally measured (non-interpolated) device response for the device-level inference step.

## Overview

Each notebook runs the full pipeline:

1. Load the measured transfer curves from `measurement/Experiment_results.csv` (`Current` + `x30`, `x40`, `x50`, `x60`, `x70`, one column per Re composition).
2. Detect the significant sigmoid-like transition region in each curve.
3. Extract that region and shift / flip / scale it to a normalized form.
4. Fit a sigmoid function to each processed curve to obtain `x0` (center) and `k` (slope) parameters.
5. Train an MLP classifier with a single hidden layer on three benchmark datasets (MNIST Digits, Fashion-MNIST, EMNIST Characters), comparing:
   - an ideal `sigmoid` activation (baseline),
   - the fitted sigmoid activation for each Re composition,
   - a quantized activation built directly from the measured, non-interpolated device response (each input is assigned the response of its nearest experimentally measured state).
6. Report test accuracy in the same format as Table 1 (fitted sigmoid vs. ideal) and Table 2 (quantized device-response inference) of the manuscript.

### Difference from the multi-hidden-layer version

This version uses a **single hidden layer** (784 → *H* → num_classes) instead of two or three, to more clearly isolate the effect of the varied activation parameter (Re composition) on classification performance. The device-response evaluation also **quantizes each input to its nearest measured activation level** rather than linearly interpolating between measured points, so reported accuracy reflects only the discrete states the device can physically occupy.

## Repository contents

```
fin_snap/
├── measurement/
│   └── Experiment_results.csv         # measured transfer curves used for sigmoid fitting
├── mlp_simulation_128.ipynb           # MLP with 1 hidden layer (128 neurons)
├── mlp_simulation_192.ipynb           # MLP with 1 hidden layer (192 neurons)
├── mlp_simulation_384.ipynb           # MLP with 1 hidden layer (384 neurons)
└── README.md
```

The three notebooks are identical except for the hidden-layer width (`MLP784_*` class), which is reflected in each filename.

## Requirements

- Python 3.10+
- `torch`, `torchvision`
- `numpy`, `pandas`, `scipy`, `matplotlib`

```bash
pip install torch torchvision numpy pandas scipy matplotlib
```

A CUDA-capable GPU is used automatically if available (`torch.cuda.is_available()`); otherwise the notebook falls back to CPU. If multiple GPUs are available, the ideal model and the 5 fitted-composition models train in parallel, one group per GPU (reused round-robin if fewer GPUs than groups are available).

## How to run

1. Open one of the three notebooks in Jupyter / VS Code.
2. Run all cells from top to bottom. MNIST, Fashion-MNIST, and EMNIST are downloaded automatically via `torchvision` on first run (into a local `./data` folder, not included in this repository).
3. Training runs for all three benchmark datasets, under the ideal / fitted / quantized-device-response activation settings described above.

### Outputs

Running a notebook produces the following files in its working directory, tagged with the hidden-layer width (`128`, `192`, or `384`) so the three notebooks can be run in the same directory without overwriting each other's results:

| File | Description |
|---|---|
| `{dataset}_train_validation_accuracy_{H}.csv` | Per-epoch train/validation accuracy for the ideal and fitted-sigmoid models, one file per dataset (`mnist`, `fashion`, `emnist_letters`) |
| `table1_test_accuracy_{H}.csv` | Final test accuracy, ideal sigmoid vs. fitted sigmoid per Re composition — matches manuscript Table 1 |
| `lut_inference_results_{H}.csv` | Raw quantized device-response inference accuracy (long format) |
| `table2_lut_inference_accuracy_{H}.csv` | Quantized device-response inference accuracy per Re composition — matches manuscript Table 2 |

`{H}` is the hidden-layer width of the notebook that produced the file (128, 192, or 384).
