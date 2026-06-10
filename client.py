import grpc
import cv2
import numpy as np
import time

# 导入刚刚编译生成的 protobuf 接口
import yolo_service_pb2
import yolo_service_pb2_grpc

# 完整的 80 个 COCO 类别映射
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# 预生成随机且固定的颜色列表，保证每个类别的颜色不同
# 固定随机种子为42，确保每次运行脚本生成的颜色映射是不变的
np.random.seed(42) 
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype="uint8")

def run(image_path, server_address='localhost:8033'):
    # 1. 建立与 gRPC 服务端的连接
    print(f"Connecting to server at {server_address}...")
    channel = grpc.insecure_channel(server_address)
    stub = yolo_service_pb2_grpc.YoloDetectorStub(channel)

    # 2. 读取原始图片并将其编码为字节流
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 将 OpenCV 的 Mat 编码为 JPEG 格式的 bytes
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    # 3. 构造请求并发送
    request = yolo_service_pb2.DetectRequest(image_data=img_bytes)
    
    print("Sending request for inference...")
    start_time = time.time()
    
    try:
        response = stub.Detect(request)
    except grpc.RpcError as e:
        print(f"gRPC call failed: {e.code()} - {e.details()}")
        return

    end_time = time.time()
    print(f"Inference complete in {(end_time - start_time) * 1000:.2f} ms")

    # 4. 解析响应结果
    if response.status_code != 0:
        print(f"Server returned error: {response.message}")
        return

    detections = response.detections
    print(f"Found {len(detections)} objects.")

    # 5. 在原图上绘制检测框
    for det in detections:
        xmin, ymin = int(det.x), int(det.y)
        xmax, ymax = int(det.width), int(det.height)
        conf = det.confidence
        cls_id = det.class_id
        
        # 获取标签文字
        label = COCO_CLASSES.get(cls_id, f"Class {cls_id}")
        text = f"{label}: {conf:.2f}"

        # 获取该类别的专属颜色 (BGR格式)
        color = tuple(int(c) for c in COLORS[cls_id % len(COLORS)])

        # 画矩形框 (线条颜色)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        
        # 画标签底色
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + tw, ymin), color, -1)
        
        # 智能计算文字颜色：计算背景色亮度，亮度高用黑字，亮度低用白字
        # OpenCV 中颜色通道顺序为 BGR
        luminance = color[2] * 0.299 + color[1] * 0.587 + color[0] * 0.114
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

        # 绘制文字
        cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # 6. 保存并显示结果
    output_path = "result.jpg"
    cv2.imwrite(output_path, img)
    print(f"Result saved to {output_path}")

if __name__ == '__main__':
    # 测试图片路径
    test_image = "/home//YOLO/ultralytics/ultralytics/assets/bus.jpg"
    run(test_image)