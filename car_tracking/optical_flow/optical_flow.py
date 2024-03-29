import cv2
import numpy as np


def select_key_points(bounding_boxes, gray_image, max_corners=100, quality_level=0.01, min_distance=10):
    """
    Selects key points within car bounding boxes using the Shi-Tomasi corner detection algorithm.

    Parameters:
    - bounding_boxes: A list of bounding boxes, where each box is represented as (x, y, w, h).
    - gray_image: The grayscale version of the current frame.
    - max_corners: The maximum number of corners to return. If there are more corners than are found,
      the strongest of them will be returned.
    - quality_level: Parameter characterizing the minimal accepted quality of image corners;
      the value of the parameter is multiplied by the best corner quality measure, which is the minimum eigenvalue
      (see cornerMinEigenVal) or the Harris function response (see cornerHarris). The corners with the quality measure
      less than the product are rejected. For example, if the best corner has the quality measure = 1500,
      and the quality_level=0.01, then all the corners with the quality measure less than 15 are rejected.
    - min_distance: Minimum possible Euclidean distance between the returned corners.

    Returns:
    - An array of points to track, in the format needed for cv2.calcOpticalFlowPyrLK.
    """
    points = []
    for (x, y, w, h) in bounding_boxes:
        # Defining the ROI for the current bounding box
        roi = gray_image[y:y + h, x:x + w]
        # Detecting corners in the ROI
        corners = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners, qualityLevel=quality_level,
                                          minDistance=min_distance)
        if corners is not None:
            # Adjusting corner positions to the full image coordinates
            corners = corners + np.array([x, y], dtype=np.float32)
            # for corner in corners:
            points.extend(corners.tolist())

    # Converting the list of points to a numpy array with the shape (n, 1, 2)
    if points:
        points = np.array(points, dtype=np.float32)
    else:
        # If no points were found, return an empty array with the correct shape
        points = np.empty((0, 1, 2), dtype=np.float32)

    return points


def tracking(prev_frame, bbox, present_frame):
    pass
