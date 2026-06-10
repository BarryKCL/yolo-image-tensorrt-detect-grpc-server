import grpc
import cv2
import os
import time
import numpy as np

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

def compute_iou(box1, box2):
    """计算两个框的交并比 (IoU)。格式: [xmin, ymin, xmax, ymax]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def load_ground_truth(label_path, img_width, img_height):
    """读取 YOLO 格式的 txt 标签，并转换为绝对坐标"""
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
        
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
                abs_w = w * img_width
                abs_h = h * img_height
                xmin = (x_c * img_width) - (abs_w / 2)
                ymin = (y_c * img_height) - (abs_h / 2)
                xmax = xmin + abs_w
                ymax = ymin + abs_h
                gt_boxes.append([class_id, xmin, ymin, xmax, ymax])
    return gt_boxes

def evaluate_grpc_coco8(server_address='localhost:8080'):
    dataset_dir = "/home//YOLO/datasets/coco8"
    images_dir = os.path.join(dataset_dir, "images/val")
    labels_dir = os.path.join(dataset_dir, "labels/val")

    # 创建保存结果的文件夹
    output_dir = "runs/grpc_val_results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(images_dir):
        print(f"找不到图片目录: {images_dir}")
        return

    channel = grpc.insecure_channel(server_address)
    stub = yolo_service_pb2_grpc.YoloDetectorStub(channel)

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_threshold = 0.5

    print(f"================ 开始 gRPC 服务端 COCO8 验证 ================\n")
    print(f"提示: 验证结果图片将保存在目录 [{output_dir}] 中\n")

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.rsplit('.', 1)[0] + '.txt')

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # 为了画图清晰，我们拷贝一份原图，避免污染原始数据
        draw_img = img.copy()

        # 1. 读取 Ground Truth 标签并画图 (绿色)
        gt_boxes = load_ground_truth(label_path, img_w, img_h)
        for gt in gt_boxes:
            cls_id = int(gt[0])
            xmin, ymin, xmax, ymax = map(int, gt[1:5])
            label_name = COCO_CLASSES.get(cls_id, f"Cls{cls_id}")
            
            # 画真实框 (绿色)
            cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"GT: {label_name}"
            # 真实框标签显示在框的上方
            cv2.putText(draw_img, text, (xmin, max(ymin - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. 发起 gRPC 推理
        _, img_encoded = cv2.imencode('.jpg', img)
        request = yolo_service_pb2.DetectRequest(image_data=img_encoded.tobytes())
        
        start_time = time.time()
        response = stub.Detect(request)
        infer_time = (time.time() - start_time) * 1000

        # 3. 解析预测结果并画图 (红色)
        pred_boxes = []
        if response.status_code == 0:
            for det in response.detections:
                if det.confidence >= 0.25: 
                    pred_boxes.append([det.class_id, det.x, det.y, det.width, det.height, det.confidence])
                    
                    cls_id = int(det.class_id)
                    xmin, ymin, xmax, ymax = map(int, [det.x, det.y, det.width, det.height])
                    label_name = COCO_CLASSES.get(cls_id, f"Cls{cls_id}")
                    
                    # 画预测框 (红色)
                    cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    text = f"PR: {label_name} {det.confidence:.2f}"
                    # 预测框标签显示在框的下方 (避免和绿色 GT 标签重叠)
                    cv2.putText(draw_img, text, (xmin, min(ymax + 18, img_h - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        pred_boxes.sort(key=lambda x: x[5], reverse=True)

        # 4. 保存画好框的图片
        output_img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_img_path, draw_img)

        # 5. 匹配预测框与真实框 (计算 TP, FP, FN)
        matched_gt = [False] * len(gt_boxes)
        image_tp = 0
        image_fp = 0

        for pred in pred_boxes:
            pred_class = pred[0]
            pred_bbox = pred[1:5]
            
            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(gt_boxes):
                if gt[0] == pred_class and not matched_gt[i]:
                    iou = compute_iou(pred_bbox, gt[1:5])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            if best_iou >= iou_threshold:
                matched_gt[best_gt_idx] = True
                image_tp += 1
            else:
                image_fp += 1

        image_fn = len(gt_boxes) - image_tp
        
        total_tp += image_tp
        total_fp += image_fp
        total_fn += image_fn

        print(f"🖼️  图片: {img_name} (保存至: {output_img_path})")
        print(f"   - 推理耗时: {infer_time:.1f}ms")
        print(f"   - 匹配情况: TP: {image_tp} | FP: {image_fp} | FN: {image_fn}\n")

    # 6. 计算最终全局指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("================ 全局评估指标 (IoU 阈值: 0.5) ================")
    print(f"🎯 精确率 (Precision): {precision:.4f}")
    print(f"🎯 召回率 (Recall):    {recall:.4f}")
    print(f"🎯 F1-Score:         {f1_score:.4f}")
    print(f"📂 结果图片已保存至: {os.path.abspath(output_dir)}")
    print("==============================================================")

if __name__ == '__main__':
    evaluate_grpc_coco8(server_address='localhost:8033')  # 使用你刚才压测的端口 8033