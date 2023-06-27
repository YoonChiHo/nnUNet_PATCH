FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
WORKDIR /data/nnUNet
COPY ["nnunet", "/data/nnUNet/nnunet"]
COPY ["setup.py", "/data/nnUNet/setup.py"]
COPY ["setup.cfg", "/data/nnUNet/setup.cfg"]
COPY ["start.sh", "/data/nnUNet/start.sh"]
RUN pip install -e.
VOLUME ["/nnUNet_preprocessed", "/data/nnUNet/nnUNet_preprocessed"]
VOLUME ["/nnUNet_raw_data_base", "/data/nnUNet/nnUNet_raw_data_base"]
VOLUME ["/nnUNet_trained_models", "/data/nnUNet/nnUNet_trained_models"]
CMD ["sh", "start.sh"]