from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import json

import numpy as np
import cv2


class BaseObjectDetectionModel(ABC):

    def __init__(self, model_path: Path, labels_path: Path):
        self.model_path = model_path
        self.labels_path = labels_path

    @property
    @abstractmethod
    def input_tensor_dtype_expected(self):
        pass

    @property
    def _labels_dict(self):
        with open(self.labels_path) as f:
            labels_dict = json.load(f)
        return labels_dict

    def predict(self, image: np.ndarray, filter_threshold: float, draw_boxes: bool, text_size: float = 0.45) \
            -> Union[np.ndarray, List[Dict[str, Union[str, int]]]]:

        input_tensor_expected = self.prepare_input_as_required_by_model(image)
        raw_results = self.run_inference(input_tensor_expected)
        parsed_and_filtered_results = self._parse_inference_results(*raw_results, filter_threshold)
        if draw_boxes:
            annotated_image = self._parse_results_of_prediction_to_get_an_image_annotated(
                parsed_and_filtered_results, image, text_size=text_size)
            return annotated_image
        else:
            human_readable_results = self._parse_results_of_prediction_to_get_humain_readable_format(parsed_and_filtered_results)
            return human_readable_results

    @abstractmethod
    def prepare_input_as_required_by_model(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def run_inference(self, input_tensor_expected: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        pass

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

    def _parse_results_of_prediction_to_get_humain_readable_format(self, results_of_prediction: List[Dict[str, Any]]) \
            -> List[Dict[str, Union[str, float]]]:

        info = []
        if len(results_of_prediction) > 0:
            for result in results_of_prediction:
                dic = {self._labels_dict[str(result["class_id"])]: round(float(result["score"]), 2)}
                info.append(dic)
        return info

    def _parse_results_of_prediction_to_get_an_image_annotated(self,
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

    def _get_bounding_boxes_coordinates(self, image: np.ndarray, result_of_prediction: Dict[str, Any]) -> tuple:

        imh, imw = image.shape[0:-1]

        ymin = int(max(1, (result_of_prediction["bounding_box"][0] * imh)))
        xmin = int(max(1, (result_of_prediction["bounding_box"][1] * imw)))
        ymax = int(min(imh, (result_of_prediction["bounding_box"][2] * imh)))
        xmax = int(min(imw, (result_of_prediction["bounding_box"][3] * imw)))

        return xmin, ymin, xmax, ymax

    def _add_class_label_to_image(self, image: np.ndarray, result_of_prediction: Dict[str, Any],
                                  xmin: float, ymin: float, text_size: float) -> None:

        label_name = self._labels_dict[int(result_of_prediction["class_id"])]
        label = f'{label_name}: {int(result_of_prediction["score"] * 100)}%'
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, label_size[1] + 10)
        cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10),
                      (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                      cv2.FILLED)
        cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0),
                    2)


