# OCTCube-M 本機開發/驗證環境
# base image 已含 Python + torch + CUDA runtime, 與 L40/Linux 同環境, 躲掉 Windows DLL segfault。
#
# 若要改 torch/CUDA 版本, 換下面 tag 即可 (driver 610.47 向後相容 CUDA 12.x):
#   https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# 非互動 apt, 裝最小系統套件 (h5py/opencv-headless 的 runtime 依賴)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 先裝相依 (利用 layer cache: requirements 不變時不重裝)
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# 程式碼與資料用 -v 掛載進來 (不 COPY: 倉庫含 1.3G pilot h5 + 4G checkpoint)
CMD ["bash"]
