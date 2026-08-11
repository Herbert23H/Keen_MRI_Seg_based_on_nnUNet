import os
import SimpleITK as sitk

# def reslice_and_save_mask(volume_path, sagittal_mask_path, coronal_image_path, axial_image_path, output_dir):
#     # 加载原始图像（用于参考元数据）
#     volume = sitk.ReadImage(volume_path)
#     coronal_image = sitk.ReadImage(coronal_image_path)
#     axial_image = sitk.ReadImage(axial_image_path)

#     # 加载分割掩码
#     sagittal_mask = sitk.ReadImage(sagittal_mask_path)

#     # 重新采样到冠状面（coronal plane）
#     coronal_mask = sitk.Resample(
#         sagittal_mask,
#         coronal_image.GetSize(),  # 使用冠状面图像的尺寸
#         sitk.Transform(),
#         sitk.sitkNearestNeighbor,  # 最近邻插值
#         coronal_image.GetOrigin(),  # 使用冠状面图像的原点
#         coronal_image.GetSpacing(),  # 使用冠状面图像的间距
#         coronal_image.GetDirection(),  # 使用冠状面图像的方向矩阵
#         0,  # 背景值
#         sagittal_mask.GetPixelID()
#     )

#     # 重新采样到轴状面（axial plane）
#     axial_mask = sitk.Resample(
#         sagittal_mask,
#         axial_image.GetSize(),  # 使用轴状面图像的尺寸
#         sitk.Transform(),
#         sitk.sitkNearestNeighbor,  # 最近邻插值
#         axial_image.GetOrigin(),  # 使用轴状面图像的原点
#         axial_image.GetSpacing(),  # 使用轴状面图像的间距
#         axial_image.GetDirection(),  # 使用轴状面图像的方向矩阵
#         0,  # 背景值
#         sagittal_mask.GetPixelID()
#     )

#     # 保存重新采样的掩码
#     name_without_ext = os.path.splitext(os.path.splitext(os.path.basename(volume_path))[0])[0]
#     os.makedirs(output_dir, exist_ok=True)
#     coronal_mask_path = os.path.join(output_dir, f"{name_without_ext}_cor_mask.nii.gz")
#     axial_mask_path = os.path.join(output_dir, f"{name_without_ext}_tra_mask.nii.gz")
#     sitk.WriteImage(coronal_mask, coronal_mask_path)
#     sitk.WriteImage(axial_mask, axial_mask_path)

#     print(f"Coronal mask saved to: {coronal_mask_path}")
#     print(f"Axial mask saved to: {axial_mask_path}")

# # 示例用法
# volume_path = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/imagesTs/OurOA_052_0001.nii.gz"
# sagittal_mask_path = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/inferTs/OurOA_052.nii.gz"
# coronal_image_path = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/transferOrigins/pdcor.nii.gz"
# axial_image_path = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/transferOrigins/pdtra.nii.gz"
# output_dir = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/transferredMasks"
# reslice_and_save_mask(volume_path, sagittal_mask_path, coronal_image_path, axial_image_path, output_dir)


def resample_with_transform(input_image_path, input_mask_path, target_image_path, output_path):
    # 加载输入图像和目标图像
    input_image = sitk.ReadImage(input_image_path)
    target_image = sitk.ReadImage(target_image_path)

    # 提取元数据
    input_origin = input_image.GetOrigin()
    input_spacing = input_image.GetSpacing()
    input_direction = input_image.GetDirection()

    target_origin = target_image.GetOrigin()
    target_spacing = target_image.GetSpacing()
    target_direction = target_image.GetDirection()

    # 创建仿射变换
    transform = sitk.AffineTransform(3)  # 3D 仿射变换

    # 设置方向矩阵（旋转部分）
    # 结合 input_direction 和 target_direction 计算新的方向矩阵
    combined_direction = [i * t for i, t in zip(input_direction, target_direction)]
    transform.SetMatrix(combined_direction)  # 设置结合后的方向矩阵

    # 设置缩放（间距差异）
    scale_factors = [i / t for i, t in zip(input_spacing, target_spacing)]
    transform.Scale(scale_factors)  # 设置缩放因子

    # 设置平移（原点差异）
    translation = [t - i for t, i in zip(target_origin, input_origin)]
    transform.SetTranslation(translation)

    # 重采样
    resampled_mask = sitk.Resample(
        input_mask_path,
        target_image.GetSize(),  # 使用目标图像的尺寸
        transform,               # 应用变换
        sitk.sitkLinear,         # 插值方法
        target_origin,           # 使用目标图像的原点
        target_spacing,          # 使用目标图像的体素间距
        target_direction,        # 使用目标图像的方向矩阵
        0,                       # 背景值
        input_image.GetPixelID()
    )

    # 保存结果
    sitk.WriteImage(resampled_mask, output_path)
    print(f"Resampled mask saved to: {output_path}")

# 示例用法
input_image_path = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/imagesTs/OurOA_052_0001.nii.gz"
input_mask_path = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/inferTs/OurOA_052.nii.gz"
target_image_path = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/transferOrigins/pdcor.nii.gz"
output_path = "autodl-tmp/DATASET/nnUNet_raw/Dataset002_OurOA/transferredMasks/resampled_mask.nii.gz"
resample_with_transform(input_image_path, input_mask_path, target_image_path, output_path)
