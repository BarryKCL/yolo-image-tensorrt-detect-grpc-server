# 使用 NVIDIA 官方包含 TensorRT 环境的镜像 (标签请根据 NVCR 实际包含 TRT11 的版本调整)
# 默认的 nvcr.io/nvidia/tensorrt 镜像已经配置好了完整的 CUDA 和 C++ 编译环境
FROM nvcr.io/nvidia/tensorrt:26.05-py3

ENV DEBIAN_FRONTEND=noninteractive

# 1. 安装 OpenCV、gRPC 和 Protobuf 依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/*

# 2. 设置工作目录并将代码复制到容器中
WORKDIR /app
COPY yolo_service.proto server.cpp CMakeLists.txt ./
COPY models ./models

# 3. 编译 C++ 服务端项目
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# 4. 暴露端口
EXPOSE 8080

# 5. 启动命令 (挂载 engine 模型)
CMD ["./build/yolo_server", "./models/yolo26x-fp16.engine"]