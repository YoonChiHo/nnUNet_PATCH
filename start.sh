today=$(date "+%Y%m%d")
echo ${today}
echo "UUID GPU List - target"
nvidia-smi -L

export nnUNet_raw_data_base="/data/nnUNet/nnUNet_raw_data_base"
export nnUNet_preprocessed="/data/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/nnUNet/nnUNet_trained_models"

#전처리 단계
#1. -t는 501 고정 (같은 raw/crop 데이터 사용)
#2. --target_task에 저장하고자 하는 Preprocessing Task 이름 설정. (Task000_ 형태여야함)
#3. --target_patch_size에 원하는 patch size를 설정 (띄어쓰기로 x, y, z 크기 구분)

#nnUNet_plan_and_preprocess -t 501 --target_task Task601_P80 --verify_dataset_integrity
#nnUNet_plan_and_preprocess -t 501 --target_task Task602_P112 --target_patch_size 112 224 256 --verify_dataset_integrity
#nnUNet_plan_and_preprocess -t 501 --target_task Task603_P128 --target_patch_size 128 256 320 --verify_dataset_integrity
#[120, 240, 278][144, 288, 320]
#훈련 단계
#1. 원하는 PREPROCESSED task idx로 설정 
#2. --result_name 에 저장하고자 하는 결과 이름 설정 (없다면 기본 세팅으로 설정됨)
#3. --patch_select에 원하는 patch 개수 설정 (positive, negative 순으로 설정)

#nnUNet_train 3d_fullres nnUNetTrainerV2 601 0 --result_name ${today}_original --deterministic
#nnUNet_train 3d_fullres nnUNetTrainerV2 601 0 --result_name ${today}_3p7n --patch_select 3 7 --deterministic

nnUNet_train 3d_fullres nnUNetTrainerV2 602 0 --result_name ${today}_1p2n --patch_select 1 2 --deterministic   #slurm-48659
#nnUNet_train 3d_fullres nnUNetTrainerV2 602 0 --result_name ${today}_2p1n --patch_select 2 1 --deterministic  #slurm-48639

#nnUNet_train 3d_fullres nnUNetTrainerV2 603 0 --result_name ${today}_1p1n --patch_select 1 1 --deterministic 
#nnUNet_train 3d_fullres nnUNetTrainerV2 603 0 --result_name ${today}_1p1n_500e --patch_select 1 1 --deterministic #slurm-48657
#nnUNet_train 3d_fullres nnUNetTrainerV2 603 0 --result_name ${today}_1p1n_bs4 --patch_select 1 1 --deterministic #slurm-48425 
#nnUNet_train 3d_fullres nnUNetTrainerV2 603 0 --result_name ${today}_2p1n --patch_select 2 1 --deterministic