import cv2
import numpy as np


def KeepMaxContour(mask):
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    ###no contour
    if len(contour_sizes) == 0:
        return None
    
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    blank_image = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(blank_image, pts=[biggest_contour], color= (1,1,1))
    return blank_image