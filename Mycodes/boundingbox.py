import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import scipy.ndimage as ndi
from pathlib import Path
import os


def process_segmentation_and_mri(segmentation_path, mri_path, output_dir):
    """    
    :param segmentation_path: 分割掩码 nnUNet推理输出
    :param mri_path: 待分割mri
    :param output_dir: 输出路径

    """
    # 加载分割结果
    segmentation_nii = nib.load(segmentation_path)
    segmentation_data = segmentation_nii.get_fdata()

    # 加载原始 MRI 数据
    mri_nii = nib.load(mri_path)
    mri_data = mri_nii.get_fdata()

    name_without_ext = os.path.splitext(os.path.splitext(os.path.basename(mri_path))[0])[0]
    patient_num = name_without_ext[6:12]
    if name_without_ext.endswith("0000"):
        save_name = "pdsag"
    elif name_without_ext.endswith("0001"):
        save_name = "t2sag"
    elif name_without_ext.endswith("0002"):
        save_name = "t1sag"
    else:
        raise ValueError(f"无法识别的文件名: {name_without_ext}")
    
    # 验证分割结果和原始 MRI 的对齐
    assert segmentation_nii.affine.all() == mri_nii.affine.all(), "分割结果和原始 MRI 的仿射矩阵不一致！"
    if segmentation_data.shape != mri_data.shape:
        print(
            f"形状不一致，重采样原始MRI到分割结果shape: "
            f"{mri_data.shape} -> {segmentation_data.shape}"
        )
        mri_nii = resample_from_to(mri_nii, segmentation_nii, order=1)
        mri_data = mri_nii.get_fdata()

    # 获取分割结果中的所有类别（排除背景类别 0）
    unique_labels = np.unique(segmentation_data)
    unique_labels = unique_labels[unique_labels != 0]  # 排除背景

    # 存储每个类别的最小外接矩形
    bounding_boxes = {}

    for label in unique_labels:
        # 获取当前类别的掩码
        mask = segmentation_data == label

        # 仅保留最大连通域
        structure = np.ones((3, 3, 3), dtype=bool)
        labeled, num_features = ndi.label(mask, structure=structure)
        if num_features == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest_label = sizes.argmax()
        largest_cc = labeled == largest_label

        # 获取最大连通域的体素坐标
        coords = np.array(np.where(largest_cc))  # (3, N) 的数组，表示 (z, y, x)
        
        # 计算最小外接矩形
        z_min, y_min, x_min = coords.min(axis=1)
        z_max, y_max, x_max = coords.max(axis=1)

        # 调整边界，确保不会小于 0 或超过 320
        z_min = min(max(z_min - 10, 0), max(z_min - 15, 0))
        if label == 1: # 针对髌骨特调以覆盖髌骨关节
            z_max = max(min(z_max + 15, 319), min(z_max + 25, 319))
        else:
            z_max = max(min(z_max + 10, 319), min(z_max + 15, 319))
        y_min = min(max(y_min - 10, 0), max(y_min - 15, 0))
        y_max = max(min(y_max + 10, 319), min(y_max + 15, 319))
        x_min, x_max = max(x_min-2, 0), min(x_max+2, 19)

        # 存储结果
        bounding_boxes[label] = {
            "z_min": z_min, "z_max": z_max,
            "y_min": y_min, "y_max": y_max,
            "x_min": x_min, "x_max": x_max
        }

    # 裁剪每个类别的 ROI
    rois = {}

    for label, box in bounding_boxes.items():
        z_min, z_max = box["z_min"], box["z_max"]
        y_min, y_max = box["y_min"], box["y_max"]
        x_min, x_max = box["x_min"], box["x_max"]

        # 裁剪原始 MRI 数据
        roi = mri_data[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        rois[label] = roi

    # 打印每个类别的最小外接矩形并保存裁剪后的nii.gz
    for label, box in bounding_boxes.items():
        with open(os.path.join(output_dir, "bounding_boxes.log"), "a") as log_file:
            log_file.write(f"类别 {label} 的最小外接矩形: x({box['z_min']}, {box['z_max']}), "
                   f"y({box['y_min']}, {box['y_max']}), z({box['x_min']}, {box['x_max']})\n")

        # 保存裁剪后的 ROI 为新的 NIfTI 文件
        roi_nii = nib.Nifti1Image(rois[label], affine=mri_nii.affine)
        save_dir = Path(f"{output_dir}/area{int(label)}/M{patient_num}/")
        os.makedirs(save_dir, exist_ok=True)  # 确保目标目录存在
        nib.save(roi_nii, f"{save_dir}/{save_name}.nii.gz")


# 定义分割结果和原图目录
segmentation_dir = "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_segresult"
mri_dir = "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_forseg"
output_dir = "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/boundingboxROIs"

# 遍历分割结果目录
for segmentation_file in os.listdir(segmentation_dir):
    if segmentation_file.endswith(".nii.gz"):
        # 提取分割文件的基础名称
        base_name = os.path.splitext(os.path.splitext(segmentation_file)[0])[0]  # 去掉 .nii.gz 后缀

        # 构造原图文件名
        mri_file_0000 = f"{base_name}_0000.nii.gz"
        mri_file_0001 = f"{base_name}_0001.nii.gz"
        mri_file_0002 = f"{base_name}_0002.nii.gz"

        # 构造原图路径
        mri_path_0000 = os.path.join(mri_dir, mri_file_0000)
        mri_path_0001 = os.path.join(mri_dir, mri_file_0001)
        mri_path_0002 = os.path.join(mri_dir, mri_file_0002)

        # 检查原图是否存在
        if os.path.exists(mri_path_0000) & os.path.exists(mri_path_0001) & os.path.exists(mri_path_0002):
            with open(os.path.join(output_dir, "bounding_boxes.log"), "a") as log_file:
                log_file.write(f"Processing segmentation: {segmentation_file}\n")
        else:
            print(f"Original image not found for segmentation: {segmentation_file}")
            with open(os.path.join(output_dir, "bounding_boxes.log"), "a") as log_file:
                log_file.write(f"Original image not found for segmentation: {segmentation_file}\n")
        process_segmentation_and_mri(
            os.path.join(segmentation_dir, segmentation_file),
            mri_path_0000,
            output_dir
        )
        process_segmentation_and_mri(
            os.path.join(segmentation_dir, segmentation_file),
            mri_path_0001,
            output_dir
        )
        process_segmentation_and_mri(
            os.path.join(segmentation_dir, segmentation_file),
            mri_path_0002,
            output_dir
        )

        print(f"Processed segmentation: {segmentation_file}")


# if __name__ == "__main__":
#     # 分割出多连通域的例子测试330741
#     # 记得重新跑出t1序列0002 
#     process_segmentation_and_mri("/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_segresult/OurOA_330741.nii.gz",
#                                  "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_forseg/OurOA_330741_0000.nii.gz",
#                                  "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/boundingboxROIs")
#     process_segmentation_and_mri("/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_segresult/OurOA_330741.nii.gz",
#                                  "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_forseg/OurOA_330741_0001.nii.gz",
#                                  "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/boundingboxROIs")
#     process_segmentation_and_mri("/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_segresult/OurOA_330741.nii.gz",
#                                  "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_forseg/OurOA_330741_0002.nii.gz",
#                                  "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/boundingboxROIs")