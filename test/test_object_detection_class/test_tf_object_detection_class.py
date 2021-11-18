from pathlib import Path
import cv2

from src.package_perso.tf_object_detection_model import TfObjectDetectionModel

LABELS = Path('/Users/leo.babonnaud/lab/package_perso/test/models/label_map.json')
SAVED_MODEL = Path('/Users/leo.babonnaud/lab/package_perso/test/models/saved_model')
IMAGE_PATH = Path('/Users/leo.babonnaud/lab/package_perso/test/data')
model = TfObjectDetectionModel(SAVED_MODEL, LABELS)


def test_object_detection_model_is_properly_instanciated():
    expected = {'1': 'Chevreuil Européen', '2': 'Renard roux', '3': 'Martre des pins', '4': "Sanglier d'Eurasie"}
    assert model._labels_dict == expected


def test_object_detection_model_run_inference_properly_for_a_given_img_of_sanglier():
    image = cv2.imread(str(IMAGE_PATH/'night_K_RCNX0305 (1).jpg'))
    image = model.prepare_input_as_required_by_model(image)
    results = model.run_inference(input_tensor_expected=image)
    results = model._parse_inference_results(*results, filter_threshold=0.9)
    results = model._parse_results_of_prediction_to_get_humain_readable_format(results)

    expected = [
        {"Sanglier d'Eurasie": 0.91}
    ]
    assert results == expected


def test_object_detection_model_run_inference_properly_for_a_given_img_of_fox():
    image = cv2.imread(str(IMAGE_PATH / 'day_A_IMG_0055.jpg'))
    image = model.prepare_input_as_required_by_model(image)
    results = model.run_inference(input_tensor_expected=image)
    results = model._parse_inference_results(*results, filter_threshold=0.9)
    results = model._parse_results_of_prediction_to_get_humain_readable_format(results)

    expected = [
        {"Renard roux": 0.93}
    ]
    assert results == expected


def test_object_detection_model_run_inference_properly_for_a_given_img_of_martre():
    image = cv2.imread(str(IMAGE_PATH / 'unknown_2020-10-04_Martes_foina_01_6 (1).jpg'))
    image = model.prepare_input_as_required_by_model(image)
    results = model.run_inference(input_tensor_expected=image)
    results = model._parse_inference_results(*results, filter_threshold=0.5)
    results = model._parse_results_of_prediction_to_get_humain_readable_format(results)

    expected = [
        {"Martre des pins": 0.9}
    ]
    assert results == expected


def test_object_detection_model_run_inference_properly_for_a_given_img_of_chevreuil():
    image = cv2.imread(str(IMAGE_PATH / 'unknown_2020-12-08_Capreolus_capreolus_02_6.jpg'))
    image = model.prepare_input_as_required_by_model(image)
    results = model.run_inference(input_tensor_expected=image)
    results = model._parse_inference_results(*results, filter_threshold=0.5)
    results = model._parse_results_of_prediction_to_get_humain_readable_format(results)

    expected = [
        {"Chevreuil Européen": 0.85},
        {"Sanglier d'Eurasie": 0.55}
    ]
    assert results == expected