import random
import time

import numpy as np
import cv2
import os
from tqdm import tqdm

# from car_tracking.simple.deep_sort import build_tracker
from car_detection.simple.car_detection import CarDetection as detection
from car_detection.trt.car_detection_trt import CarDetection as detection_trt
# from car_tracking.simple.utils.parser import get_config
from car_tracking.trt import car_tracking_trt

# from car_tracking.trt.deep_sort.utils.parser import get_config
# from car_tracking.trt.deep_sort.deep_sort import DeepSort
from car_tracking.optical_flow import optical_flow


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
    detector_trt = detection_trt({
        'weights': '/home/nvidia/zwh/Auto-Edge/batch_test/yolov5s_batch1.engine',
        'plugin_library': '/home/nvidia/zwh/Auto-Edge/batch_test/libbatch1plugins.so',
        'batch_size': 1,
        'device': 0
    })

    # cfg = get_config()
    # cfg.merge_from_file('car_tracking/simple/configs/deep_sort.yaml')
    # cfg.USE_FASTREID = False
    # tracker_simple = build_tracker(cfg, use_cuda=True)
    #
    # cfg_trt = get_config()
    # cfg_trt.merge_from_file("car_tracking/trt/deep_sort/configs/deep_sort.yaml")
    # tracker_trt = DeepSort(cfg.DEEPSORT.REID_CKPT,
    #                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    #                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    #                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    #                        use_cuda=True)

    video_dir = '/data/edge_computing_dataset/UA-DETRAC/Insight-MVT_Annotation_Train'
    gt_file = '/data/edge_computing_dataset/UA-DETRAC/train_gt.txt'

    warm_up(detector_trt, video_dir, gt_file, 100)

    result = None
    prob = None

    detector_trt_delay = []
    tracker_trt_delay = []
    tracker_simple_delay = []

    with open(gt_file, 'r') as gt_f:
        gt = gt_f.readlines()
        gt = gt[:100]
        # gt = random.sample(gt, 100)
    frame = None
    for i in tqdm(gt):
        info = i.split(' ')
        pic_path = os.path.join(video_dir, info[0])
        prev_frame = frame
        frame = cv2.imread(pic_path)
        print(f'frame size:{frame.shape}')
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if result and prob:
            start_time = time.time()

            # tracker_trt.update(car_tracking_trt.xyxy_to_xywh(np.asarray(result)), prob, frame)
            optical_flow.tracking(prev_frame, result, [frame])
            end_time = time.time()
            tracker_trt_delay.append(end_time - start_time)

        # if result and prob:
        #     start_time = time.time()
        #     tracker_simple.update(np.asarray(result), np.asarray(prob), frame)
        #     end_time = time.time()
        #     tracker_simple_delay.append(end_time - start_time)

        start_time = time.time()
        response = detector_trt([frame])
        result = response['result'][0]
        prob = response['probs'][0]
        print(f'object number: {len(result)}')

        end_time = time.time()
        detector_trt_delay.append(end_time - start_time)

    print(f'【detector】 trt:{np.mean(detector_trt_delay):.4f}s ')
    # print(f'【tracker】  simple:{np.mean(tracker_simple_delay):.4f}s')
    print(f'【tracker】  trt:{np.mean(tracker_trt_delay):.4f}s')


def batch_delay_test():
    server = RoadSurveillance


if __name__ == '__main__':
    detection_tracking_delay_test()
