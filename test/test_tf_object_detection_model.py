from pathlib import Path

import cv2

from package_perso.tf_object_detection_model import ObjectDetectionModel


def test_object_detection_model_return_image():
    model = ObjectDetectionModel(Path("models/saved_model"),
                                 Path("models/label_map.txt"))
    image = cv2.imread("data/night_K_RCNX0305 (1).jpg")
    img_annotated_with_prediction_results = model.predict(image, 0.4, True)
    cv2.imshow("prediction", img_annotated_with_prediction_results)
    cv2.waitKey()


def test_object_detection_model_return_correct_output():
    model = ObjectDetectionModel(Path("models/saved_model"),
                                 Path("models/label_map.txt"))
    image = cv2.imread("data/night_K_RCNX0305 (1).jpg")
    results = model.predict(image, 0.4, False)
    print(results)


test_object_detection_model_return_correct_output()