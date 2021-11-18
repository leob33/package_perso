from pathlib import Path

import cv2
import numpy as np

from tf_object_detection_model import ObjectDetectionModel


def test_object_detection_model_return_image():
    model = ObjectDetectionModel(Path("/Users/leo.babonnaud/lab/package_perso/models/saved_model"),
                             Path("/Users/leo.babonnaud/lab/package_perso/models/label_map.txt"))
    image = cv2.imread("data/day_A_IMG_0055.jpg")
    img_annotated_with__prediction_results = model.predict(image, 0.4, True)
    cv2.imshow("prediction", img_annotated_with__prediction_results)
    cv2.waitKey()


test_object_detection_model_return_image()

