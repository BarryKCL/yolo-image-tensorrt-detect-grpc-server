#include <iostream>
#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <fstream>
#include <algorithm>

#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "yolo_service.grpc.pb.h"

using namespace grpc;
using namespace yolo;

// 内部检测结果结构体
struct DetResult {
    float x, y, width, height, conf;
    int class_id;
};

// 推理任务结构体
struct InferTask {
    cv::Mat blob; // [1, 3, 640, 640] 的单张预处理结果
    float scale;
    std::promise<std::vector<DetResult>> promise; // 用于异步返回结果
};

// TensorRT 日志捕获器
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

class YoloDetectorServiceImpl final : public YoloDetector::Service {
public:
    YoloDetectorServiceImpl(const std::string& model_path) {
        LoadEngine(model_path);
        
        // 分配 MAX_BATCH 大小的连续内存空间
        cudaMalloc(&d_input_, max_batch_size_ * 3 * 640 * 640 * sizeof(float));
        cudaMalloc(&d_output_, max_batch_size_ * num_detections_ * 6 * sizeof(float));
        h_input_.resize(max_batch_size_ * 3 * 640 * 640);
        h_output_.resize(max_batch_size_ * num_detections_ * 6);
        
        cudaStreamCreate(&stream_);
        
        // 启动后台消费者线程
        stop_worker_ = false;
        worker_thread_ = std::thread(&YoloDetectorServiceImpl::BatchWorker, this);
        
        std::cout << "🚀 Dynamic Batching Engine Initialized! Max Batch: " << max_batch_size_ << std::endl;
    }

    ~YoloDetectorServiceImpl() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_worker_ = true;
        }
        queue_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        cudaFree(d_input_);
        cudaFree(d_output_);
        cudaStreamDestroy(stream_);
    }

    // gRPC 接口：这里只负责预处理和排队，不执行推理
    Status Detect(ServerContext* context, const DetectRequest* request, DetectResponse* response) override {
        cv::Mat img = cv::imdecode(std::vector<uchar>(request->image_data().begin(), request->image_data().end()), cv::IMREAD_COLOR);
        if (img.empty()) {
            response->set_status_code(400);
            return Status::OK;
        }

        // 1. 预处理
        float scale = std::min(640.0f / img.cols, 640.0f / img.rows);
        cv::Mat resized_img;
        cv::Size resize_shape(int(img.cols * scale), int(img.rows * scale));
        cv::resize(img, resized_img, resize_shape);
        cv::copyMakeBorder(resized_img, resized_img, 0, 640 - resized_img.rows, 0, 640 - resized_img.cols, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        
        cv::Mat blob = cv::dnn::blobFromImage(resized_img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

        // 2. 封装任务并入队
        auto task = std::make_shared<InferTask>();
        task->blob = blob;
        task->scale = scale;
        auto future = task->promise.get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push(task);
        }
        queue_cv_.notify_one(); // 唤醒后台线程

        // 3. 阻塞等待后台线程把结果塞进 promise
        std::vector<DetResult> results = future.get();

        // 4. 将结果打包为 gRPC Response
        for (const auto& r : results) {
            Detection* det = response->add_detections();
            det->set_x(r.x); det->set_y(r.y);
            det->set_width(r.width); det->set_height(r.height);
            det->set_confidence(r.conf); det->set_class_id(r.class_id);
        }

        response->set_status_code(0);
        return Status::OK;
    }

private:
    // 后台消费者线程：负责收集图片并执行 GPU 批处理
    void BatchWorker() {
        while (true) {
            std::vector<std::shared_ptr<InferTask>> batch_tasks;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                // 等待条件：队列达到最大 batch 数量，或者等待了超过 5 毫秒，或者收到停止信号
                queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms_), [this] {
                    return task_queue_.size() >= max_batch_size_ || stop_worker_;
                });

                if (stop_worker_ && task_queue_.empty()) break;

                // 取出最多 max_batch_size_ 个任务
                while (!task_queue_.empty() && batch_tasks.size() < max_batch_size_) {
                    batch_tasks.push_back(task_queue_.front());
                    task_queue_.pop();
                }
            }

            if (batch_tasks.empty()) continue;

            int current_batch = batch_tasks.size();

            // 1. 将散落的多个 cv::Mat blob 拼接到一段连续的内存中 (CPU 端)
            size_t single_image_size = 3 * 640 * 640 * sizeof(float);
            for (int i = 0; i < current_batch; ++i) {
                memcpy(h_input_.data() + i * 3 * 640 * 640, batch_tasks[i]->blob.ptr<float>(0), single_image_size);
            }

            // 2. CPU -> GPU 拷贝
            cudaMemcpyAsync(d_input_, h_input_.data(), current_batch * single_image_size, cudaMemcpyHostToDevice, stream_);

            // 3. TRT 10 动态 Batch API：设置当前运行的真实 Shape
            nvinfer1::Dims4 input_dims{current_batch, 3, 640, 640};
            context_->setInputShape(input_name_, input_dims);

            context_->setTensorAddress(input_name_, d_input_);
            context_->setTensorAddress(output_name_, d_output_);
            
            // 4. 执行推理
            context_->enqueueV3(stream_);

            // 5. GPU -> CPU 拷贝
            cudaMemcpyAsync(h_output_.data(), d_output_, current_batch * num_detections_ * 6 * sizeof(float), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);

            // 6. 分发结果给各个等待的客户端
            for (int b = 0; b < current_batch; ++b) {
                std::vector<DetResult> res;
                float* batch_ptr = h_output_.data() + b * num_detections_ * 6;
                float scale = batch_tasks[b]->scale;

                for (int i = 0; i < num_detections_; ++i) {
                    float conf = batch_ptr[i * 6 + 4];
                    if (conf > 0.25f) {
                        res.push_back({
                            batch_ptr[i * 6 + 0] / scale,
                            batch_ptr[i * 6 + 1] / scale,
                            batch_ptr[i * 6 + 2] / scale,
                            batch_ptr[i * 6 + 3] / scale,
                            conf,
                            static_cast<int>(batch_ptr[i * 6 + 5])
                        });
                    }
                }
                // 唤醒对应的 gRPC 线程
                batch_tasks[b]->promise.set_value(res);
            }
        }
    }

    void LoadEngine(const std::string& model_path) {
        std::ifstream file(model_path, std::ios::binary);
        if (!file.good()) exit(-1);

        uint32_t first_4_bytes;
        file.read(reinterpret_cast<char*>(&first_4_bytes), sizeof(first_4_bytes));

        size_t engine_offset = 0;
        if (first_4_bytes != 1953657958 && first_4_bytes < 1000000) {
            std::cout << "[INFO] Skipping " << first_4_bytes << " bytes of JSON header..." << std::endl;
            engine_offset = sizeof(first_4_bytes) + first_4_bytes;
        }

        file.seekg(0, file.end);
        size_t engine_size = file.tellg() - static_cast<std::streampos>(engine_offset);
        file.seekg(engine_offset, file.beg);
        std::vector<char> engineStream(engine_size);
        file.read(engineStream.data(), engine_size);
        file.close();

        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engineStream.data(), engine_size));
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

        input_name_ = engine_->getIOTensorName(0);
        output_name_ = engine_->getIOTensorName(1);

        auto output_dims = engine_->getTensorShape(output_name_);
        num_detections_ = output_dims.d[1]; // 通常为 300
    }

    // TensorRT 对象
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    const char* input_name_;
    const char* output_name_;
    cudaStream_t stream_;
    int num_detections_;

    // 显存和主机内存
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    std::vector<float> h_input_;
    std::vector<float> h_output_;

    // --- 动态 Batch 核心成员 ---
    int max_batch_size_ = 8;  // 最大 Batch 尺寸 (可根据显存调整)
    int timeout_ms_ = 5;      // 攒图最大等待时间 (毫秒)
    
    std::queue<std::shared_ptr<InferTask>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread worker_thread_;
    bool stop_worker_;
};

void RunServer(const std::string& model_path) {
    ServerBuilder builder;
    YoloDetectorServiceImpl service(model_path);
    builder.AddListeningPort("0.0.0.0:8080", InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    server->Wait();
}

int main(int argc, char** argv) {
    std::string model_path = "./models/yolo26x-fp16.engine";
    if (argc > 1) model_path = argv[1];
    RunServer(model_path);
    return 0;
}