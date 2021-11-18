import os

import cv2.cv2

from images_align import align
from exposure_fusion_alg import exposure_fusion

if __name__ == "__main__":
    src_path = "images"
    images = []
    for file in os.listdir(src_path):
        images.append(cv2.cv2.imread(os.path.join(src_path, file)))

    aligned = align(images)
    _, res = exposure_fusion(images)
    cv2.cv2.imwrite("res.jpg", res)