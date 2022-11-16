from typing import Any, List

import numpy as np
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

import detection.postprocess as det_postproc
from segmentation.inference import FINAL_CLASS_INFO as SEG_CLASS_INFO


# Parameters
class CFG:
    _IMG_SIZE = 160
    _IOU_THR = 0.3
    _SAHI_OVERLAP_RATIO = 0.3
    _SEG_POSTPROCESS_TH = {
        SEG_CLASS_INFO['ignore']: 1,
        SEG_CLASS_INFO['tumor']: 1,
        SEG_CLASS_INFO['stroma']: 0.,
        SEG_CLASS_INFO['rest']: .6,
        4: .8
    }
    _CONF_ADJUST_HPARAM = [0.0, 0.08, 0.17, -0.15, -0.04, -0.04, 0.]
    _ENSEMBLE_POST_HPARAM = {
        'boost_3+': 1.0,
        'boost_2': .95,
        'decrease': 0.65
    }
    _ECC_TH = 0.9
    _ECC_CONF_MULT = 0.8


global MODEL_POOL

MODEL_POOL = None


def build_detect_model_pool(
    config: dict,
    image_size: int = 160,
    device: str = "cuda:0"
) -> None:

    global MODEL_POOL
    MODEL_POOL = _build_detection_models(config,
                                         image_size,
                                         device=device)


def _build_detection_models(
    config: dict,
    image_size: int,
    device: str = "cuda:0",
) -> List:

    detection_models = []

    for i, (model_name, _) in enumerate(config.items()):
        checkpoint_path = config[model_name]['path']
        threshold = config[model_name]['th']
        if config[model_name]['type'] == "yolov5":
            detection_model = AutoDetectionModel.from_pretrained(model_type="yolov5",
                                                                 model_path=checkpoint_path,
                                                                 confidence_threshold=threshold,
                                                                 image_size=image_size,
                                                                 device=device)
        else:
            assert 'not define model path or need to insert path in list'

        detection_models.append(detection_model)

    return detection_models


def detect_cells(
    image: np.ndarray,
    mask: np.ndarray = None,
    visualize_fname: str = None,
    use_center_nms: bool = True,
) -> Any:
    cfg = CFG

    # Check image is not empty
    assert image is not None

    # Create variables to save results
    boxes_list, scores_list, labels_list = [], [], []
    total_df = []

    # Get image height and width
    height, width, _ = image.shape

    # Run inference for each model
    for detection_model in MODEL_POOL:
        result = get_sliced_prediction(image,
                                       detection_model,
                                       image_size=cfg._IMG_SIZE,
                                       slice_height=cfg._IMG_SIZE,
                                       slice_width=cfg._IMG_SIZE,
                                       overlap_height_ratio=cfg._SAHI_OVERLAP_RATIO,
                                       overlap_width_ratio=cfg._SAHI_OVERLAP_RATIO,
                                       verbose=0)
        result = det_postproc.sahi_result_postprocess(result)

        # Format result
        data = [({'bbox': list(row['bbox']), 'score':row['score']}) for row in result.to_coco_annotations()]
        df = pd.DataFrame(data)
        total_df.append(df)

    # Convert data frame to list
    for fold_df in total_df:
        try:
            scores_list.append(np.array(list(fold_df['score'])))
        except:
            continue
        bboxes = np.array(list(fold_df['bbox'])).astype(float)
        boxes_list.append(bboxes)

        labels_list.append(list([1] * len(fold_df)))

    postprocess = det_postproc.TNBCPostprocess(cfg, boxes_list, scores_list, labels_list, image, mask)

    postprocess.conf_adjust()

    if use_center_nms:
        postprocess.center_nms()
    else:
        postprocess.normal_nms()

    if mask is not None:
        postprocess.seg_postprocess()

    postprocess.eccentricity_filtering()

    if visualize_fname:
        postprocess.visualize(visualize_fname)

    # Compute probabilties
    probs = list(postprocess.scores)

    return list(map(round, postprocess.x_coords)), list(map(round, postprocess.y_coords)), probs
