import os
import SimpleITK as sitk

def register_and_reslice_mask(sag_image_path, cor_image_path, tra_image_path, sag_mask_path, output_dir):
    # 加载原始图像
    sag_image = sitk.ReadImage(sag_image_path, sitk.sitkFloat32)
    cor_image = sitk.ReadImage(cor_image_path, sitk.sitkFloat32)
    tra_image = sitk.ReadImage(tra_image_path, sitk.sitkFloat32)

    # 加载分割掩码
    sag_mask = sitk.ReadImage(sag_mask_path)

    # 配准 sag 原图到 cor 原图
    initial_transform_cor = sitk.CenteredTransformInitializer(
        cor_image,
        sag_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method_cor = sitk.ImageRegistrationMethod()
    registration_method_cor.SetMetricAsMeanSquares()
    registration_method_cor.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-6, numberOfIterations=100)
    registration_method_cor.SetInitialTransform(initial_transform_cor, inPlace=False)
    registration_method_cor.SetInterpolator(sitk.sitkLinear)
    final_transform_cor = registration_method_cor.Execute(cor_image, sag_image)

    # 配准 sag 原图到 tra 原图
    initial_transform_tra = sitk.CenteredTransformInitializer(
        tra_image,
        sag_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method_tra = sitk.ImageRegistrationMethod()
    registration_method_tra.SetMetricAsMeanSquares()
    registration_method_tra.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-6, numberOfIterations=100)
    registration_method_tra.SetInitialTransform(initial_transform_tra, inPlace=False)
    registration_method_tra.SetInterpolator(sitk.sitkLinear)
    final_transform_tra = registration_method_tra.Execute(tra_image, sag_image)

    # 应用变换到分割掩码（cor）
    cor_mask = sitk.Resample(
        sag_mask,
        cor_image.GetSize(),
        final_transform_cor,
        sitk.sitkNearestNeighbor,  # 最近邻插值，保持掩码的整数标签
        cor_image.GetOrigin(),
        cor_image.GetSpacing(),
        cor_image.GetDirection(),
        0,  # 背景值
        sag_mask.GetPixelID()
    )

    # 应用变换到分割掩码（tra）
    tra_mask = sitk.Resample(
        sag_mask,
        tra_image.GetSize(),
        final_transform_tra,
        sitk.sitkNearestNeighbor,
        tra_image.GetOrigin(),
        tra_image.GetSpacing(),
        tra_image.GetDirection(),
        0,
        sag_mask.GetPixelID()
    )

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    cor_mask_path = os.path.join(output_dir, "coronal_mask.nii.gz")
    tra_mask_path = os.path.join(output_dir, "axial_mask.nii.gz")
    sitk.WriteImage(cor_mask, cor_mask_path)
    sitk.WriteImage(tra_mask, tra_mask_path)

    print(f"Coronal mask saved to: {cor_mask_path}")
    print(f"Axial mask saved to: {tra_mask_path}")

# 示例用法
sag_image_path = "path/to/sag_image.nii.gz"
cor_image_path = "path/to/cor_image.nii.gz"
tra_image_path = "path/to/tra_image.nii.gz"
sag_mask_path = "path/to/sag_mask.nii.gz"
output_dir = "path/to/output_masks"
register_and_reslice_mask(sag_image_path, cor_image_path, tra_image_path, sag_mask_path, output_dir)