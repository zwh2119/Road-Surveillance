from car_detection.trt.car_detection_trt import CarDetection


class RoadSurveillanceBaseline:
    def __init__(self, args):
        self.detector = CarDetection(args)

    def __call__(self, images):
        assert images, 'image list is empty'
        detection_list = images
        result_detection = []
        probs = []
        for image in detection_list:
            response = self.detector([image])
            result = response['result'][0]
            prob = response['probs'][0]
            result_detection.append(result)
            probs.append(prob)

        return {'bbox': result_detection, 'prob': probs}
