import cv2

from car_detection.trt.car_detection_trt import CarDetection
from car_tracking.optical_flow import optical_flow


class RoadSurveillance:
    def __init__(self, args):
        self.detector = CarDetection(args)

    def __call__(self, images):
        assert images, 'image list is empty'
        detection_list = images[0:1]
        tracking_list = images[1:]

        response = self.detector(detection_list)
        result_detection = response['result'][0]
        prob = response['probs'][0]
        result_tracking = optical_flow.tracking(detection_list[0], result_detection, tracking_list)
        result_tracking.insert(0, result_detection)

        return {'bbox': result_tracking, 'prob': prob}
