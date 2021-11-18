from typing import Tuple
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.package_perso.object_detection_model import BaseObjectDetectionModel


class TfLiteObjectDetectionModel(BaseObjectDetectionModel):

    def __init__(self, model_path: Path, labels_path: Path):
        super().__init__(model_path, labels_path)
        self.input_details_dic = self._interpreter.get_input_details()[0]
        self.output_details_list_dic = self._interpreter.get_output_details()
        self.input_tensor_size_expected = tuple(self.input_details_dic['shape'][1:-1])

    @property
    def input_tensor_dtype_expected(self):
        return self.input_details_dic['dtype']

    @property
    def _interpreter(self):
        _interpreter = tf.lite.Interpreter(str(self.model_path))
        _interpreter.allocate_tensors()
        return _interpreter

    def prepare_input_as_required_by_model(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_tensor_size_expected)
        if self.input_tensor_dtype_expected == np.float32:
            input_mean = 127.5
            input_std = 127.5
            image = (np.float32(image) - input_mean) / input_std
        image = image.astype(self.input_tensor_dtype_expected)

        return np.expand_dims(image, 0)

    def run_inference(self, input_tensor_expected: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        self._interpreter.set_tensor(self.input_details_dic['index'], input_tensor_expected)
        self._interpreter.invoke()
        bounding_boxes_coordinates_array = self._interpreter.get_tensor(self.output_details_list_dic[0]['index'])[0]
        predicted_classe_indexes_array = self._interpreter.get_tensor(self.output_details_list_dic[1]['index'])[0]
        proba_of_class_predicted_array = self._interpreter.get_tensor(self.output_details_list_dic[2]['index'])[0]
        number_of_predictions = int(self._interpreter.get_tensor(self.output_details_list_dic[3]['index'])[0])

        return bounding_boxes_coordinates_array, predicted_classe_indexes_array, proba_of_class_predicted_array, \
               number_of_predictions
