# Knee OA Structure Segmentation — Pre-trained Weights (Dataset002_OurOA)

Pre-trained **5-fold nnU-Net v2** weights for automatic segmentation of knee
osteoarthritis-related structures from MRI.

> ⚠️ **Data notice:** The training data is **not public**. This branch releases
> only the trained **model weights** for inference. Please contact the authors
> for data access.

## Model Details

- **Modalities (2 channels per case):**
  - `0000`: PDW
  - `0001`: T2W
- **Segmentation targets (5 structures + background):**

  | Label ID | Structure |
  |---------:|-----------|
  | 0        | background |
  | 1        | Patella |
  | 2        | Fat Pad |
  | 3        | Meniscus |
  | 4        | Femur |
  | 5        | Tibia |

- **Configuration:** `3d_fullres` (Trainer `nnUNetTrainer__nnUNetPlans__3d_fullres`)
- **Checkpoints:** 5-fold cross-validation `checkpoint_best.pth` (ensembled at inference)

## Repository Layout

```
.
├── README.md
└── nnUNet_results/                          # nnUNet_results data root (env: nnUNet_results)
    └── Dataset002_OurOA/
        └── nnUNetTrainer__nnUNetPlans__3d_fullres/
            ├── plans.json
            ├── dataset.json
            ├── dataset_fingerprint.json
            └── fold_{0..4}/checkpoint_best.pth
```

> Note: model weights are stored via [Git LFS](https://git-lfs.com/) (`*.pth`).

## Requirements

```bash
pip install nnunetv2        # nnU-Net v2 (see the official repo for details)
git lfs pull                # fetch the LFS weight files
```

Set the nnU-Net environment variable (adjust paths to your checkout):

```bash
export nnUNet_results=/path/to/this/repo/nnUNet_results
```

## Inference

Put your case files in an input directory, each named as
`<case_id>_0000.nii.gz` (PDW) and `<case_id>_0001.nii.gz` (T2W), then run:

```bash
nnUNetv2_predict \
  -i /path/to/input_dir \
  -o /path/to/output_dir \
  -d 002 \
  -c 3d_fullres
```

By default nnU-Net ensembles all 5 folds for more robust predictions.

## License & Usage

The released weights are provided for **research purposes only**. Please cite
this project if you use them, and contact the authors for data access or
questions about clinical/commercial use.
