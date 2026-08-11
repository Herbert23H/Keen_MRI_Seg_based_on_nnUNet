for fold in {0..4}
do 
    # echo "nnUNetv2_train 2 3d_fullres $fold"
    nnUNetv2_train 2 3d_fullres $fold
done

