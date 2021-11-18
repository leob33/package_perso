
class BaseObjectDetectionModel:

    def __init__(self):
        pass

    def predict(self):
        pass

    def _prepare_input_as_required_by_model(self):
        pass

    def _run_inference(self):
        pass

    def _parse_inference_results(self):
        pass

    def _parse_results_of_prediction_to_get_humain_readable_format(self):
        pass

    def _parse_results_of_prediction_to_get_an_image_annotated(self):
        pass

    def _get_bounding_boxes_coordinates(self):
        pass

    def _add_class_label_to_image(self):
        pass

