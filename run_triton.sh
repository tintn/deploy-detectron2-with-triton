docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 --shm-size=1gb\
    -v /home/tin/Workspace/triton/deploy-detectron2-with-triton/models:/models \
    nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver --model-repository=/models
