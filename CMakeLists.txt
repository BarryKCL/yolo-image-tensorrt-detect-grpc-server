cmake_minimum_required(VERSION 3.10)
project(YoloTensorRTGrpcService)

set(CMAKE_CXX_STANDARD 17)

# 查找系统依赖
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)

# 查找 grpc 插件
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

# 查找 CUDA 和 TensorRT
find_package(CUDAToolkit REQUIRED)
# 在 NVIDIA 容器中，nvinfer 通常在标准路径下
find_library(TRT_LIBRARY nvinfer REQUIRED)

# 生成 Protobuf 和 gRPC 代码
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/yolo_service.proto")
set(PROTO_SRC "${CMAKE_CURRENT_BINARY_DIR}/yolo_service.pb.cc")
set(PROTO_HEADER "${CMAKE_CURRENT_BINARY_DIR}/yolo_service.pb.h")
set(GRPC_SRC "${CMAKE_CURRENT_BINARY_DIR}/yolo_service.grpc.pb.cc")
set(GRPC_HEADER "${CMAKE_CURRENT_BINARY_DIR}/yolo_service.grpc.pb.h")

add_custom_command(
    OUTPUT ${PROTO_SRC} ${PROTO_HEADER} ${GRPC_SRC} ${GRPC_HEADER}
    COMMAND protoc
    ARGS --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
         --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
         -I ${CMAKE_CURRENT_SOURCE_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
)

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})

add_executable(yolo_server server.cpp ${PROTO_SRC} ${GRPC_SRC})

# 链接 CUDA, TensorRT 和其他基础库
target_link_libraries(yolo_server
    CUDA::cudart
    ${TRT_LIBRARY}
    ${OpenCV_LIBS}
    ${Protobuf_LIBRARIES}
    grpc++
    grpc
    gpr        # <=== 【核心修复】：增加 gpr 底层库链接
    absl_synchronization   # <=== 【新增】解决 Mutex 报错
    absl_time              # <=== 【新增】配合同步库使用的时间库
    absl_base              # <=== 【新增】absl 基础库
    Threads::Threads
)