from pathlib import Path
from typing import Tuple, Union, List, Dict, Any

import numpy as np
import tensorflow as tf
import cv2


class ObjectDetectionModel():

    def __init__(self, model_path: Path, labels_path: Path):
        self.model_path = model_path
        self.labels_path = labels_path
        self.input_tensor_dtype_expected = np.float32

    @property
    def _detection_function(self):
        _detection_function = tf.saved_model.load(str(self.model_path))
        return _detection_function

    @property
    def _labels_list(self):
        labels_list = open(self.labels_path).read().strip().split("\n")
        return labels_list

    def predict(self, image: np.ndarray, filter_threshold: float, draw_boxes: bool, text_size: float = 0.45) \
            -> Union[np.ndarray, List[Dict[str, Union[str, int]]]]:

        input_tensor_expected = self._prepare_input_as_required_by_model(image)
        boxes, classes, scores, count = self._run_inference(input_tensor_expected)
        results = self._parse_inference_results(boxes, classes, scores, count, filter_threshold)
        if draw_boxes:
            results = self._annotate_raw_image_with_prediction_results(results, image, text_size=text_size)
        else:
            results = self._decode_class_id_in_results(results)

        return results

    def _prepare_input_as_required_by_model(self, image: np.ndarray) -> np.ndarray:
        return np.expand_dims(image, 0)

    def _run_inference(self, input_tensor_expected: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        detections = self._detection_function(input_tensor_expected)
        num_detections = int(detections.pop('num_detections'))
        detections = {key_to_keep: detections[key_to_keep] for key_to_keep in
                      ['detection_boxes', 'detection_scores', 'detection_classes']}
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        return detections['detection_boxes'], detections['detection_classes'], detections['detection_scores'], \
               detections['num_detections']

    def _parse_inference_results(self, bounding_boxes_coordinates_list,
                                 predicted_classes_indexes_list,
                                 proba_of_class_predicted_list,
                                 number_of_predictions,
                                 filter_threshold) -> List[Dict[str, Any]]:

        results = []
        for i in range(number_of_predictions):
            if proba_of_class_predicted_list[i] >= filter_threshold:
                result = {
                    'bounding_box': bounding_boxes_coordinates_list[i],
                    'class_id': predicted_classes_indexes_list[i],
                    'score': proba_of_class_predicted_list[i]
                }
                results.append(result)

        return results

    def _decode_class_id_in_results(self, results_of_prediction: List[Dict[str, Any]]) \
            -> List[Dict[str, Union[str, int]]]:

        info = []
        if len(results_of_prediction) > 0:
            for result in results_of_prediction:
                dic = {self._labels_list[int(result["class_id"])]: result["score"]}
                info.append(dic)
        return info

    def _annotate_raw_image_with_prediction_results(self,
                                                    results_of_prediction: List[Dict[str, Any]],
                                                    image: np.ndarray,
                                                    text_size: float = 0.45) \
            -> np.ndarray:

        if len(results_of_prediction) > 0:

            for result in results_of_prediction:
                xmin, ymin, xmax, ymax = self._get_bounding_boxes_coordinates(image, result)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                self._add_class_label_to_image(image, result, xmin, ymin, text_size)

        return image

    def _get_bounding_boxes_coordinates(self, image, result_of_prediction) -> tuple:

        imh, imw = image.shape[0:-1]

        ymin = int(max(1, (result_of_prediction["bounding_box"][0] * imh)))
        xmin = int(max(1, (result_of_prediction["bounding_box"][1] * imw)))
        ymax = int(min(imh, (result_of_prediction["bounding_box"][2] * imh)))
        xmax = int(min(imw, (result_of_prediction["bounding_box"][3] * imw)))

        return xmin, ymin, xmax, ymax

    def _add_class_label_to_image(self, image, results_of_prediction, xmin, ymin, text_size) -> None:

        object_name = self._labels_list[int(results_of_prediction["class_id"])]
        label = f'{object_name}: {int(results_of_prediction["score"] * 100)}%'
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10),
                      (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                      cv2.FILLED)
        cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0),
                    2)