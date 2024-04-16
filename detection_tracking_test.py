import json
import random
import time

import numpy as np
import cv2
import os
from tqdm import tqdm

from car_detection.trt.car_detection_trt import CarDetection as detection_trt

from car_tracking.optical_flow import optical_flow

from road_surveillance import RoadSurveillance
from road_surveillance_baseline import RoadSurveillanceBaseline


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


def detection_tracking_delay_test(detector_trt=None, resolution=None):
    if not detector_trt:
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

    result_detection = None
    prob = None
    result_tracking = None

    detector_trt_delay = []
    tracker_trt_delay = []
    # tracker_simple_delay = []

    detector_trt_acc = []
    tracker_trt_acc = []

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
        if resolution:
            raw_length = frame.shape[1]
            raw_height = frame.shape[0]
            new_length = resolution[0]
            new_height = resolution[1]
            frame = cv2.resize(frame, resolution)
        # print(f'frame size:{frame.shape}')
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bbox_gt = [float(b) for b in info[1:]]
        boxes_gt = np.array(bbox_gt, dtype=np.float32).reshape(-1, 4)
        frame_gt = []
        for box in boxes_gt.tolist():
            if resolution:
                box[0] *= new_length / raw_length
                box[1] *= new_height / raw_height
                box[2] *= new_length / raw_length
                box[3] *= new_height / raw_height
            frame_gt.append({'bbox': box, 'class': 1})

        if result_detection and prob:
            start_time = time.time()

            # tracker_trt.update(car_tracking_trt.xyxy_to_xywh(np.asarray(result)), prob, frame)
            response_tracking = optical_flow.tracking(prev_frame, result_detection, [frame])
            end_time = time.time()
            tracker_trt_delay.append(end_time - start_time)
            # print(f'tracker time: {end_time - start_time}')

            result_tracking = response_tracking[0]
            prediction_tracking = []
            for box, score in zip(result_tracking, prob):
                prediction_tracking.append({'bbox': box, 'prob': score, 'class': 1})
            tracker_trt_acc.append(calculate_map(prediction_tracking, frame_gt))

        # if result and prob:
        #     start_time = time.time()
        #     tracker_simple.update(np.asarray(result), np.asarray(prob), frame)
        #     end_time = time.time()
        #     tracker_simple_delay.append(end_time - start_time)

        start_time = time.time()
        response = detector_trt([frame])
        result_detection = response['result'][0]
        prob = response['probs'][0]
        # print(f'object number: {len(result_detection)}')
        prediction_detection = []
        for box, score in zip(result_detection, prob):
            prediction_detection.append({'bbox': box, 'prob': score, 'class': 1})
        detector_trt_acc.append(calculate_map(prediction_detection, frame_gt))

        end_time = time.time()
        detector_trt_delay.append(end_time - start_time)

    print(f'【detector】 delay:{np.mean(detector_trt_delay):.4f}s    map:{np.mean(detector_trt_acc)}')
    # print(f'【tracker】  simple:{np.mean(tracker_simple_delay):.4f}s')
    print(f'【tracker】  delay:{np.mean(tracker_trt_delay):.4f}s    map:{np.mean(tracker_trt_acc)}')
    return np.mean(detector_trt_delay), np.mean(detector_trt_acc), np.mean(tracker_trt_delay), np.mean(tracker_trt_acc)


def batch_delay_test():
    batch_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    frame_num = 1024
    server = RoadSurveillance({
        'weights': '/home/nvidia/zwh/Auto-Edge/batch_test/yolov5s_batch1.engine',
        'plugin_library': '/home/nvidia/zwh/Auto-Edge/batch_test/libbatch1plugins.so',
        'batch_size': 1,
        'device': 0
    })

    video_dir = '/data/edge_computing_dataset/UA-DETRAC/Insight-MVT_Annotation_Train'
    gt_file = '/data/edge_computing_dataset/UA-DETRAC/train_gt.txt'

    with open(gt_file, 'r') as gt_f:
        gt_file = gt_f.readlines()
        gt_file = gt_file[:frame_num]

    batch_delay = []
    batch_acc = []
    for batch in batch_list:
        print(f'test for batch size of {batch}..')
        frame_buffer = []
        delay_buffer = []
        acc_buffer = []
        gt_buffer = []
        for i in tqdm(gt_file):
            info = i.split(' ')

            pic_path = os.path.join(video_dir, info[0])
            frame = cv2.imread(pic_path)

            bbox_gt = [float(b) for b in info[1:]]
            boxes_gt = np.array(bbox_gt, dtype=np.float32).reshape(-1, 4)
            frame_gt = []
            for box in boxes_gt.tolist():
                frame_gt.append({'bbox': box, 'class': 1})

            frame_buffer.append(frame)
            gt_buffer.append(frame_gt)
            if len(frame_buffer) == batch:
                start_time = time.time()
                response = server(frame_buffer)
                # print(f'response:{response}')
                end_time = time.time()
                delay_buffer.append((end_time - start_time) * 1000 / batch)

                prediction = []
                for bbox in response['bbox']:
                    pred = []
                    # print(f'bbox:{bbox}, probs:{probs}')
                    for box, score in zip(bbox, response['prob']):
                        pred.append({'bbox': box, 'prob': score, 'class': 1})
                    prediction.append(pred)

                acc = []
                for pred, gt in zip(prediction, gt_buffer):
                    # print(f'pred: {pred}   gt:{gt}')
                    acc.append(calculate_map(pred, gt))
                acc_buffer.append(np.mean(acc))

                frame_buffer = []
                gt_buffer = []

        batch_acc.append(np.mean(acc_buffer))
        batch_delay.append(np.mean(delay_buffer))
        print(f'batch {batch}:  delay:{np.mean(delay_buffer):.2f}ms    acc:{np.mean(acc_buffer):.4f}')
    with open('detection_tracking_batch.json', 'w') as f:
        json.dump({'delay': batch_delay, 'acc': batch_acc}, f)


def batch_delay_test_baseline():
    batch_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    frame_num = 1024
    server = RoadSurveillanceBaseline({
        'weights': '/home/nvidia/zwh/Auto-Edge/batch_test/yolov5s_batch1.engine',
        'plugin_library': '/home/nvidia/zwh/Auto-Edge/batch_test/libbatch1plugins.so',
        'batch_size': 1,
        'device': 0
    })

    video_dir = '/data/edge_computing_dataset/UA-DETRAC/Insight-MVT_Annotation_Train'
    gt_file = '/data/edge_computing_dataset/UA-DETRAC/train_gt.txt'

    with open(gt_file, 'r') as gt_f:
        gt_file = gt_f.readlines()
        gt_file = gt_file[:frame_num]

    batch_delay = []
    batch_acc = []
    for batch in batch_list:
        print(f'test for batch size of {batch}..')
        frame_buffer = []
        delay_buffer = []
        acc_buffer = []
        gt_buffer = []
        for i in tqdm(gt_file):
            info = i.split(' ')

            pic_path = os.path.join(video_dir, info[0])
            frame = cv2.imread(pic_path)

            bbox_gt = [float(b) for b in info[1:]]
            boxes_gt = np.array(bbox_gt, dtype=np.float32).reshape(-1, 4)
            frame_gt = []
            for box in boxes_gt.tolist():
                frame_gt.append({'bbox': box, 'class': 1})

            frame_buffer.append(frame)
            gt_buffer.append(frame_gt)
            if len(frame_buffer) == batch:
                start_time = time.time()
                response = server(frame_buffer)
                # print(f'response:{response}')
                end_time = time.time()
                delay_buffer.append((end_time - start_time) * 1000 / batch)

                prediction = []
                for bbox in response['bbox']:
                    pred = []
                    # print(f'bbox:{bbox}, probs:{probs}')
                    for box, score in zip(bbox, response['prob']):
                        pred.append({'bbox': box, 'prob': score, 'class': 1})
                    prediction.append(pred)

                prediction = []
                for bbox, probs in zip(response['bbox'], response['prob']):
                    pred = []
                    # print(f'bbox:{bbox}, probs:{probs}')
                    for box, score in zip(bbox, probs):
                        pred.append({'bbox': box, 'prob': score, 'class': 1})
                    prediction.append(pred)

                acc = []
                for pred, gt in zip(prediction, gt_buffer):
                    # print(f'pred: {pred}   gt:{gt}')
                    acc.append(calculate_map(pred, gt))
                acc_buffer.append(np.mean(acc))

                frame_buffer = []
                gt_buffer = []

        batch_acc.append(np.mean(acc_buffer))
        batch_delay.append(np.mean(delay_buffer))
        print(f'batch {batch}:  delay:{np.mean(delay_buffer):.2f}ms    acc:{np.mean(acc_buffer):.4f}')
    with open('detection_tracking_batch_baseline.json', 'w') as f:
        json.dump({'delay': batch_delay, 'acc': batch_acc}, f)


def detection_tracking_resolution_test():
    detector_trt = detection_trt({
        'weights': '/home/nvidia/zwh/Auto-Edge/batch_test/yolov5s_batch1.engine',
        'plugin_library': '/home/nvidia/zwh/Auto-Edge/batch_test/libbatch1plugins.so',
        'batch_size': 1,
        'device': 0
    })

    resolution_dict = {
        "HVGA_360p(4:3)": (480, 360),
        "nHD_360p(16:9)": (640, 360),

        "VGA_480p(4:3)": (640, 480),

        "SVGA_600p(4:3)": (800, 600),

        "qHD_540p(16:9)": (960, 540),

        "DVCPRO-HD_720p(4:3)": (960, 720),  # 691200
        "HD_720p(16:9)": (1280, 720),  # 921600
        "WallpaperHD_720p(18:9)": (1440, 720),  # 1036800

        "WXGA_800p(16:10)": (1280, 800),  # 1024000
        "QuadVGA_960p(4:3)": (1280, 960),  # 1228800
        "WXGA+_900p(16:10)": (1440, 900),  # 1296000
        "FWXGA+_960p(3:2)": (1440, 960),  # 1382400
        "HD+_900p(16:9)": (1600, 900),  # 1440000

        "DVCPRO-HD_1080p(16:9)": (1440, 1080),  # 1555200
        "FHD_1080p(16:9)": (1920, 1080)  # 2073600
    }
    detection_delay_list = []
    detection_acc_list = []
    tracker_delay_list = []
    tracker_acc_list = []
    for resolution_name in resolution_dict:
        resolution = resolution_dict[resolution_name]
        print(f'start test on resolution of {resolution}..')
        detection_delay, detection_acc, tracker_delay, tracker_acc = detection_tracking_delay_test(detector_trt, resolution)
        detection_delay_list.append(detection_delay)
        detection_acc_list.append(detection_acc)
        tracker_delay_list.append(tracker_delay)
        tracker_acc_list.append(tracker_acc)
    with open('detection_tracking_resolution.json', 'w') as f:
        json.dump({'detection_delay':detection_delay_list,
                   'detection_acc':detection_acc_list,
                   'tracker_delay':tracker_delay_list,
                   'tracker_acc':tracker_acc_list},f)


if __name__ == '__main__':
    detection_tracking_delay_test()
    # batch_delay_test()
    # batch_delay_test_baseline()
    # detection_tracking_resolution_test()
