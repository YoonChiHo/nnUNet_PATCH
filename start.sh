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


#훈련 단계
#1. 원하는 PREPROCESSED task idx로 설정 
#2. --result_name 에 저장하고자 하는 결과 이름 설정 (없다면 기본 세팅으로 설정됨)
#3. --patch_select에 원하는 patch 개수 설정 (positive, negative 순으로 설정)

#slurm-47866
#nnUNet_train 3d_fullres nnUNetTrainerV2 601 0 --result_name 230627_original --deterministic
#nnUNet_train 3d_fullres nnUNetTrainerV2 601 0 --result_name 230627_original_randomness --deterministic
#slurm-47867
nnUNet_train 3d_fullres nnUNetTrainerV2 601 0 --result_name 230627_original_norandom2 --deterministic
#nnUNet_train 3d_fullres nnUNetTrainerV2 601 0 --result_name 230627_1p1n_randomness --patch_select 1 1 --deterministic
