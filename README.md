# Knee OA Structure Segmentation (Dataset002_OurOA)

Automatic segmentation of knee osteoarthritis-related structures from MRI using
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).

This repository releases a **trained 5-fold ensemble** and a **subset of 50
training cases** of our internal knee OA dataset. All imaging data has been
anonymized.

## Highlights

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
├── nnUNet_raw/                              # nnUNet_raw data root (env: nnUNet_raw)
│   └── Dataset002_OurOA/
│       ├── dataset.json
│       ├── imagesTr/                        # 50 training cases (PDW + T2W)
│       └── labelsTr/                        # 50 corresponding label maps
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

Set the nnU-Net environment variables (adjust paths to your checkout):

```bash
export nnUNet_raw=/path/to/this/repo/nnUNet_raw
export nnUNet_preprocessed=/path/to/your/preprocessed   # only needed for training
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

## (Optional) Retrain on the released subset

```bash
nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity

for fold in 0 1 2 3 4; do
  nnUNetv2_train 002 3d_fullres $fold
done
```

## License & Usage

The released data and weights are provided for **research purposes only**.
Please cite this project if you use them in your work, and contact the authors
for questions about clinical or commercial use.
