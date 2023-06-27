# nnUNet_PATCH

## 1. Folder Setting
다음 폴더에 사용하고자 하는 raw 데이터를 Task000_name 포맷으로 배치  
```
nnUNet_PATCH/  
├── nnUNet_raw_data_base  
│   └── nnUNet_raw_data  
```
## 2. Docker Setting    
nnUNet_PATCH/slurm.sh 의 Our Docker Setting 코드를 Local Root에 맞게 수정    
```python
# 기존 코드    
docker run --rm --name nnunet_batch_run --shm-size 16G --gpus ${GPULIST}     
-v /home2/ych000/data/nnUNet_PATCH/nnUNet_trained_models:/data/nnUNet/nnUNet_trained_models     
-v /home2/ych000/data/nnUNet_PATCH/nnUNet_preprocessed:/data/nnUNet/nnUNet_preprocessed     
-v /home2/ych000/data/nnUNet_PATCH/nnUNet_raw_data_base:/data/nnUNet/nnUNet_raw_data_base     
nnunet_batch    
```
```python
# 수정 코드    
docker run --rm --name nnunet_batch_run --shm-size 16G --gpus ${GPULIST}     
-v {Local PATH}/nnUNet_PATCH/nnUNet_trained_models:/data/nnUNet/nnUNet_trained_models     
-v {Local PATH}/nnUNet_PATCH/nnUNet_preprocessed:/data/nnUNet/nnUNet_preprocessed     
-v {Local PATH}/nnUNet_PATCH/nnUNet_raw_data_base:/data/nnUNet/nnUNet_raw_data_base     
nnunet_batch    
```
## 3. RUN CODE Setting    
nnUNet_PATCH/start.sh 의 preprocess 및 train 코드를 원하는 훈련에 맞게 수정    
```python
#전처리 단계    
#1. -t는 501 고정 (같은 raw/crop 데이터 사용)    
#2. --target_task에 저장하고자 하는 Preprocessing Task 이름 설정. (Task000_ 형태여야함)    
#3. --target_patch_size에 원하는 patch size를 설정 (띄어쓰기로 x, y, z 크기 구분)    

#nnUNet_plan_and_preprocess -t 501 --target_task Task601_P80 --verify_dataset_integrity    
#nnUNet_plan_and_preprocess -t 501 --target_task Task602_P112 --target_patch_size 112 224 256 --verify_dataset_integrity    
```
```python
#훈련 단계    
#1. 원하는 PREPROCESSED task idx로 설정     
#2. --result_name 에 저장하고자 하는 결과 이름 설정 (없다면 기본 세팅으로 설정됨)    
#3. --patch_select에 원하는 patch 개수 설정 (positive, negative 순으로 설정)    

#nnUNet_train 3d_fullres nnUNetTrainerV2 601 0 --result_name 230627_original --deterministic    
#nnUNet_train 3d_fullres nnUNetTrainerV2 601 0 --result_name 230627_1p1n --patch_select 1 1 --deterministic    
```

## 4. RUN SBATCH Code
* 주의: slurm.sh 코드가 포함되어 있는 폴더로 이동해서 아래 코드를 실행하여야함.  
```python  
cd {LOCAL PATH}/nnUNet_PATCH  
sbatch slurm.sh  
```

## Updated List  
1. Preprocessing 입력 추가   
   1.1 raw, preprocess결과 폴더 구분   
   1.2 패치 사이즈 입력으로 받아오기   
3. Training 입력 추가   
   2.1 입력, 결과 폴더 구분   
   2.2 패치 개수 입력으로 받아오기   
4. Randomness 세팅   
5. Validation Set 설정   
6. SBATCH GPU세팅   
   
