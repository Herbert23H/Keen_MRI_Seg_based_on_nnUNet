from encodings import raw_unicode_escape
import os
import shutil
import nibabel as nib
import numpy as np
import nibabel.processing as nibproc

# 指定目录和目标形状
directory = "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_forseg"
backup_directory = "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_forseg_shapebackup"
backup_directory2 = "/root/autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/data_nii_3000_forseg_affinebackup"
target_shape = (320, 320, 20)

# 确保备份目录存在
os.makedirs(backup_directory, exist_ok=True)
os.makedirs(backup_directory2, exist_ok=True)
# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if not filename.endswith(".nii.gz"):
        raise ValueError(f"非nii.gz文件: {filename}")

    file_path = os.path.join(directory, filename)
    img = nib.load(file_path)
    current_shape = img.shape[:3]

    if tuple(current_shape) == tuple(target_shape):
        # print(f"{filename}: shape={current_shape} 已符合目标尺寸")
        continue

    print(f"{filename}: shape={current_shape} -> 重采样到 {target_shape}, 覆盖保存!")

    # 先备份原文件
    backup_path = os.path.join(backup_directory, filename)
    shutil.copy2(file_path, backup_path)

    # 计算新的仿射矩阵：按边缘对齐进行缩放（避免右下被裁剪）
    # 目标：new[0] -> old[0]，new[-1] -> old[-1]
    current = np.array(current_shape, dtype=float)
    target = np.array(target_shape, dtype=float)
    scale = np.ones(3, dtype=float)
    valid = target > 1
    scale[valid] = (current[valid] - 1.0) / (target[valid] - 1.0)
    new_affine = img.affine.copy()
    new_affine[:3, :3] = img.affine[:3, :3] @ np.diag(scale)
    # 重采样
    resampled = nibproc.resample_from_to(img, (target_shape, new_affine), order=1)
    # 保存（覆盖原文件）
    nib.save(resampled, file_path)

# 检查0001/2序列的仿射矩阵是否与0000序列一致
# 如果不一致则写字段与0000序列相同的原点和体素间距,切记不要重采样
for filename in os.listdir(directory):
    if filename.endswith("_0000.nii.gz"):
        continue
    else:
        file_path_toadj = os.path.join(directory, filename)
        file_path_0000 = os.path.join(directory, filename.replace("_0001.nii.gz", "_0000.nii.gz").replace("_0002.nii.gz", "_0000.nii.gz"))

        img_0000 = nib.load(file_path_0000)
        img_toadj = nib.load(file_path_toadj)
        new_affine = img_0000.affine
        raw_affine = img_toadj.affine
        if (new_affine == raw_affine).all():
            print(f"{filename}: 仿射矩阵已一致")
            continue
        
        print(f"{filename}: 仿射矩阵不一致 -> 仅写header统一为 {file_path_0000} 的仿射矩阵, 覆盖保存!")

        # 先备份原文件
        backup_path2 = os.path.join(backup_directory2, filename)
        shutil.copy2(file_path_toadj, backup_path2)

        # 仅写header（不重采样）
        hdr = img_toadj.header.copy()
        hdr.set_sform(new_affine, code=1)
        hdr.set_qform(new_affine, code=1)
        updated = nib.Nifti1Image(img_toadj.get_fdata(dtype=img_toadj.get_data_dtype()), new_affine, header=hdr)

        # 保存（覆盖原文件）
        nib.save(updated, file_path_toadj)
        print(f"已统一仿射矩阵: {file_path_toadj}")