import grpc
import cv2
import time
import numpy as np
import concurrent.futures
from collections import deque
import argparse

import yolo_service_pb2
import yolo_service_pb2_grpc

def make_request(stub, request_data):
    """单个线程执行的请求函数"""
    start_time = time.time()
    try:
        response = stub.Detect(request_data)
        latency = (time.time() - start_time) * 1000  # 转换为毫秒
        if response.status_code == 0:
            return True, latency, len(response.detections)
        else:
            return False, latency, 0
    except grpc.RpcError as e:
        latency = (time.time() - start_time) * 1000
        return False, latency, 0

def run_stress_test(image_path, server_address, num_requests, num_threads):
    print(f"[{time.strftime('%X')}] 准备压测...")
    print(f"目标服务: {server_address}")
    print(f"测试图片: {image_path}")
    print(f"并发线程数: {num_threads}")
    print(f"总请求数: {num_requests}\n")

    # 1. 读取并预处理图片（在客户端只做一次，避免把本地磁盘 I/O 算入压测时间）
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return
    _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    request_data = yolo_service_pb2.DetectRequest(image_data=img_encoded.tobytes())

    # 2. 建立连接
    channel = grpc.insecure_channel(server_address)
    stub = yolo_service_pb2_grpc.YoloDetectorStub(channel)

    # 3. 预热 (Warm-up)
    # 预热可以确保服务端的模型加载到内存最佳位置，并完成 JIT 编译
    print("正在进行预热 (发送 5 个请求)...")
    for _ in range(5):
        stub.Detect(request_data)
    print("预热完成。\n")

    # 4. 开始多线程压测
    print(f"[{time.strftime('%X')}] 🚀 开始高并发压测，请稍候...")
    
    latencies = []
    success_count = 0
    fail_count = 0
    total_detections = 0

    start_time = time.time()

    # 使用线程池发起并发请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        futures = [executor.submit(make_request, stub, request_data) for _ in range(num_requests)]
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            success, latency, dets = future.result()
            latencies.append(latency)
            if success:
                success_count += 1
                total_detections += dets
            else:
                fail_count += 1

    total_time = time.time() - start_time

    # 5. 统计与打印结果
    if not latencies:
        print("未收到任何有效响应！")
        return

    latencies = np.array(latencies)
    qps = num_requests / total_time
    
    print("================ 压测报告 ================")
    print(f"总耗时:         {total_time:.3f} 秒")
    print(f"总请求数:       {num_requests}")
    print(f"成功请求:       {success_count}")
    print(f"失败请求:       {fail_count}")
    print(f"总检测目标数:   {total_detections}")
    print("-" * 40)
    print(f"🔥 QPS (吞吐量): {qps:.2f} 请求/秒")
    print("-" * 40)
    print("延迟分布 (毫秒):")
    print(f"  - 最小延迟:   {np.min(latencies):.2f} ms")
    print(f"  - 平均延迟:   {np.mean(latencies):.2f} ms")
    print(f"  - P50 延迟:   {np.percentile(latencies, 50):.2f} ms (50%的请求在此时间内完成)")
    print(f"  - P90 延迟:   {np.percentile(latencies, 90):.2f} ms (90%的请求在此时间内完成)")
    print(f"  - P99 延迟:   {np.percentile(latencies, 99):.2f} ms (99%的请求在此时间内完成)")
    print(f"  - 最大延迟:   {np.max(latencies):.2f} ms")
    print("==========================================")

if __name__ == '__main__':
    # 使用 argparse 让运行更灵活
    parser = argparse.ArgumentParser(description="YOLO gRPC 服务多线程压测工具")
    parser.add_argument('--ip', type=str, default='localhost:8036', help='服务端地址 (默认: localhost:8080)')
    parser.add_argument('--img', type=str, default='/home//YOLO/ultralytics/ultralytics/assets/bus.jpg', help='测试图片路径')
    parser.add_argument('-n', '--num', type=int, default=1000, help='压测总请求数 (默认: 100)')
    parser.add_argument('-t', '--threads', type=int, default=10, help='并发线程数 (默认: 10)')
    
    args = parser.parse_args()

    run_stress_test(
        image_path=args.img,
        server_address=args.ip,
        num_requests=args.num,
        num_threads=args.threads
    )