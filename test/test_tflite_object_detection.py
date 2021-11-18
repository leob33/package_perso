from pathlib import Path

import cv2

from tflite_object_detection import ObjectDetectionModelLite


def test_model_return_an_imge_with_labels():
    model = ObjectDetectionModelLite(labels_path=Path("/Users/leo.babonnaud/lab/package_perso/models/label_map.txt"),
                                     model_path=Path(
                                     "/Users/leo.babonnaud/lab/package_perso/models/ssd_mobilenet_v2_6.tflite"))

    for path_img in Path("data").glob("*.jpg"):
        print(path_img)
        img = cv2.imread(str(path_img))
        img_annotated_with__prediction_results = model.predict(img, 0.4, True)
        cv2.imshow("prediction", img_annotated_with__prediction_results)
        cv2.waitKey()


test_model_return_an_imge_with_labels()
