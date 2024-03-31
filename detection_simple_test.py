import time

import numpy as np
import cv2
import os
from tqdm import tqdm

from car_detection.simple.car_detection import CarDetection as detection


def warm_up(model, base_dir, gt_path, warm_number):
    with open(gt_path, 'r') as gt_f_:
        gt_ = gt_f_.readlines()
    count = 0
    for i_ in gt_:
        count += 1
        if count >= warm_number:
            break
        info_ = i_.split(' ')
        pic_path_ = os.path.join(base_dir, info_[0])
        frame_ = cv2.imread(pic_path_)
        model([frame_])


def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def compute_ap(recalls, precisions):
    # Compute AP by numerical integration of the precision-recall curve
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """
    predictions: list of dicts with keys {'bbox': [x1, y1, x2, y2], 'prob': confidence, 'class': class_id}
    ground_truths: list of dicts with keys {'bbox': [x1, y1, x2, y2], 'class': class_id}
    """
    aps = []
    for class_id in set([gt['class'] for gt in ground_truths]):
        # Filter predictions and ground truths by class
        preds = [p for p in predictions if p['class'] == class_id]
        gts = [gt for gt in ground_truths if gt['class'] == class_id]

        # Sort predictions by confidence
        preds.sort(key=lambda x: x['prob'], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        used_gts = set()

        for i, pred in enumerate(preds):
            max_iou = 0
            max_gt_idx = -1
            for j, gt in enumerate(gts):
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou and j not in used_gts:
                    max_iou = iou
                    max_gt_idx = j

            if max_iou >= iou_threshold:
                tp[i] = 1
                used_gts.add(max_gt_idx)
            else:
                fp[i] = 1

        # Calculate precision and recall
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        recalls = tp_cumsum / len(gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    # Mean AP
    mAP = np.mean(aps)
    return mAP


def detection_tracking_delay_test():
    detector_simple = detection({
        'weights': 'yolov5s.pt',
        'device': 0
    })

    video_dir = '/data/edge_computing_dataset/UA-DETRAC/Insight-MVT_Annotation_Train'
    gt_file = '/data/edge_computing_dataset/UA-DETRAC/train_gt.txt'

    # warm_up(detector_trt, video_dir, gt_file, 100)
    warm_up(detector_simple, video_dir, gt_file, 100)

    result_detection = None
    prob = None

    detector_simple_delay = []
    detector_simple_acc = []

    with open(gt_file, 'r') as gt_f:
        gt = gt_f.readlines()
        gt = gt[:100]

    for i in tqdm(gt):
        info = i.split(' ')
        pic_path = os.path.join(video_dir, info[0])
        frame = cv2.imread(pic_path)

        bbox_gt = [float(b) for b in info[1:]]
        boxes_gt = np.array(bbox_gt, dtype=np.float32).reshape(-1, 4)
        frame_gt = []
        for box in boxes_gt.tolist():
            frame_gt.append({'bbox': box, 'class': 1})

        start_time = time.time()
        response = detector_simple([frame])
        result_detection = response['result'][0]
        # print(len(result))
        prob = response['probs'][0]
        prediction_detection = []
        for box, score in zip(result_detection, prob):
            prediction_detection.append({'bbox': box, 'prob': score, 'class': 1})
        detector_simple_acc.append(calculate_map(prediction_detection, frame_gt))

        end_time = time.time()
        detector_simple_delay.append(end_time - start_time)

    print(f'【detector】 delay:{np.mean(detector_simple_delay):.4f}s    map:{np.mean(detector_simple_acc)}')


# def batch_delay_test():
#     server = RoadSurveillance


if __name__ == '__main__':
    detection_tracking_delay_test()
