docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v <full-path-to-model-repository>:/models \
    nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver --model-repository=/models
