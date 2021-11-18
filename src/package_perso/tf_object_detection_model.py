from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from src.package_perso.object_detection_model import BaseObjectDetectionModel


class TfObjectDetectionModel(BaseObjectDetectionModel):

    def __init__(self, model_path: Path, labels_path: Path):
        super().__init__(model_path, labels_path)

    @property
    def input_tensor_dtype_expected(self):
        return np.float32

    @property
    def _detection_function(self):
        _detection_function = tf.saved_model.load(str(self.model_path))
        return _detection_function

    def prepare_input_as_required_by_model(self, image: np.ndarray) -> np.ndarray:
        return np.expand_dims(image, 0)

    def run_inference(self, input_tensor_expected: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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

