import cv2

from car_detection import CarDetection
from car_tracking.deep_sort import build_tracker
from car_tracking.utils.parser import get_config


class RoadSurveillance:
    def __init__(self):
        self.detector = CarDetection({
            'weights': '',
            'plugin_library': '',
            'batch_size': 0,
            'device': 0
        })

        cfg = get_config()
        cfg.merge_from_file('car_tracking/configs/deep_sort.yaml')
        cfg.USE_FASTREID = False
        self.tracker = build_tracker(cfg, use_cuda=True)

    def __call__(self, images):
        assert images, 'image list is empty'
        detection_list = images[0:1]
        tracking_list = images[1:]
