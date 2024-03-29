# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on 2021/5/24 13:46
@Author: Wang Cong
@Email : iwangcong@outlook.com
@Version : 0.1
@File : tracker_trt.py
"""
import cv2
import numpy as np
import torch


def xyxy_to_xywh(boxes_xyxy):
    if isinstance(boxes_xyxy, torch.Tensor):
        boxes_xywh = boxes_xyxy.clone()
    elif isinstance(boxes_xyxy, np.ndarray):
        boxes_xywh = boxes_xyxy.copy()
    else:
        assert None, 'error'

    boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
    boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
    boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    return boxes_xywh


def draw_bboxes(image, bboxes, line_thickness):
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    list_pts = []
    point_radius = 4

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)

        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
        list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y-point_radius])

        ndarray_pts = np.array(list_pts, np.int32)

        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

        list_pts.clear()

    return image


# def update(bboxes, confidence,image):
#     bbox_xywh = []
#     bboxes2draw = []
#
#     if len(bboxes) > 0:
#         for x1, y1, x2, y2, in bboxes:
#             obj = [
#                 int((x1 + x2) / 2), int((y1 + y2) / 2),
#                 x2 - x1, y2 - y1
#             ]
#             bbox_xywh.append(obj)
#
#         xywhs = np.array(bbox_xywh)
#         confss = np.array(confidence)
#
#         outputs = deepsort.update(xywhs, confss, image)
#
#         for value in list(outputs):
#             x1, y1, x2, y2, track_id = value
#             bboxes2draw.append((int(x1), int(y1), int(x2), int(y2),  int(track_id)))
#         pass
#     pass
#
#     return bboxes2draw
