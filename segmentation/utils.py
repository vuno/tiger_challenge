from importlib import import_module
import logging
from typing import Any, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def classmap_dict2lut(class_map: dict, default_value: int, max_class_idx: int = 255):
    return np.array([class_map.get(class_idx, default_value) for class_idx in range(max_class_idx + 1)], dtype=np.uint8)


def make_lowsatval_img(img: np.ndarray, saturation_target: int = 50, value_target: int = 150, min_saturation_to_count: int = 20) -> np.ndarray:
    """
        Make an RGB image with low saturation / value to the extent of arg settings.

    Args:
        img (np.ndarray): Input RGB image array having [H * W * 3] np.uint8 type.
        saturation_target (int): Mean saturation over naive tissue area will be lowered (if needed)
            to meet saturation_target as an upper limit.
        value_target (int): Mean value over naive tissue area will be lowered (if needed) to meet
            value_target as an upper limit.
        min_saturation_to_count (int): A naive tissue area will be defined as `naive_tissue = img > min_saturation_to_count`

    Returns:
        np.ndarray: A numpy array applied by sat/val lowering, having the same dimension to the input's.
    """
    img_lowsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_saturation = img_lowsv[..., 1][img_lowsv[..., 1] > min_saturation_to_count].mean()
    mean_value = img_lowsv[..., 2][img_lowsv[..., 1] > min_saturation_to_count].mean()
    if mean_saturation > saturation_target:
        sat_subtract_amount = mean_saturation - saturation_target
        img_lowsv[..., 1] = cv2.subtract(img_lowsv[..., 1], sat_subtract_amount)
    if mean_value > value_target:
        val_subtract_amount = mean_value - value_target
        img_lowsv[..., 2] = cv2.subtract(img_lowsv[..., 2], val_subtract_amount)
    return cv2.cvtColor(img_lowsv, cv2.COLOR_HSV2RGB)


def replace_small_object(pred: np.ndarray, src_mpp: float, src_idx: int, dst_idx: int, object_size_lowerbound: float = 16**2) -> np.ndarray:
    """Suppress small objects in class==src_idx and make them into dst_idx.

    Args:
        pred (np.ndarray): Pixel-wise prediction class idx map after argmax & class-remap
        src_mpp (float): MPP of 'pred' array's pixel
        src_idx (int): Class idx of the target of size investigation which would be replaced
        dst_idx (int): Class idx to be assigned to the pixel meeting such conditions
        object_size_lowerbound (float): Physical area threshold under which the object will be replaced
    """
    pred = pred.copy()  # avoiding inplace op
    target_mask_with_small_objects = (pred == src_idx).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        target_mask_with_small_objects.astype(np.uint8))
    labels = _reduce_dtype_by_max_value(labels)
    object_size_lowerbound_in_px = object_size_lowerbound / (src_mpp ** 2)
    candid_idx_list = np.nonzero(stats[:, cv2.CC_STAT_AREA] < object_size_lowerbound_in_px)[0]
    candid_idx_list = candid_idx_list[candid_idx_list > 0]
    replace_mask = np.isin(labels, candid_idx_list)
    pred[replace_mask] = dst_idx
    return pred, target_mask_with_small_objects


def _reduce_dtype_by_max_value(arr: np.ndarray) -> np.ndarray:
    target_dtype_seq = [np.uint8, np.uint16, np.uint32]
    for dtype in target_dtype_seq:
        if arr.max() <= np.iinfo(dtype).max:
            return arr.astype(dtype)
    return arr


def boost_suboptimal_class(prob: np.ndarray, competing_idx_list: list, boosting_idx: int, confidence_lowerbound: int = 205) -> np.ndarray:
    """Boost 'boosting_idx' class on the pixels where Top-K classes are 'competing_idx_list'.

    Args:
        prob (np.ndarray): Probability map ndarray in dtype=np.uint8 with [H * W * C]
        competing_idx_list (list): Indices of topK competing classes (len == K)
        boosting_idx (int): The index to be replacing the competing opponent(s)
        confidence_lowerbound (int): A confidence threshold below which the pixels are considered as target

    Returns:
        np.ndarray: Modified prob, applied by boosting procedures.
    """
    boost_mask = _find_competing_area(prob=prob, competing_idx_list=competing_idx_list,
                                      confidence_lowerbound=confidence_lowerbound)
    prob = _boost_class(prob=prob, boost_mask=boost_mask, boosting_idx=boosting_idx)
    return prob


def _find_competing_area(prob: np.ndarray, competing_idx_list: list, confidence_lowerbound: int = 205) -> np.ndarray:
    assert confidence_lowerbound < 255
    K = len(competing_idx_list)  # TopK competing classes indices
    prob_argsort = np.argsort(prob, axis=-1)[..., -K:].astype(np.uint8)
    competing_area = prob.max(axis=-1) < confidence_lowerbound
    for i in competing_idx_list:
        competing_area &= ((prob_argsort == i).max(axis=-1))
    return competing_area


def _boost_class(prob: np.ndarray, boost_mask: np.ndarray, boosting_idx: int) -> np.ndarray:
    assert list(prob.shape[:2]) == list(boost_mask.shape[:2])
    assert 0 < boosting_idx < prob.shape[-1]
    booster = np.zeros_like(prob)
    booster[..., boosting_idx] = boost_mask * 255
    prob = cv2.add(prob, booster).astype(np.float16)
    prob = (prob / prob.sum(axis=-1, keepdims=True) * 255).astype(np.uint8)
    return prob


def load_yaml_config(config_fname):
    with open(str(config_fname), 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def import_object(target: str):
    """Import and getattr to import target string module names
    """
    def _import_deepmost_module(target: str) -> Tuple[Any, str, str]:
        """Import deepmost importable module

        Returns:
        (<imported_obj>, <imported_obj_dot_path>, <remaining_attr_dot_path>)
        """
        parts = target.split('.')
        assert parts

        # Import deepmost importable module - longest to shortest, back to front
        for import_pos in range(0, -len(parts), -1):
            if import_pos == 0:
                module_str = '.'.join(parts)
                remaining_attr = ''
            else:
                module_str = '.'.join(parts[:import_pos])
                remaining_attr = '.'.join(parts[import_pos:])
            try:
                return import_module(module_str), module_str, remaining_attr
            except ImportError:
                pass
        raise ImportError(f"Cannot import {module_str} (part from {target})")

    def _load_attribute(module_obj: Any, module_str: str, attr_path: str):
        attr_parts = attr_path.split('.')
        assert attr_parts

        # Try loading attribute by consuming one-by-one
        for attr_name in attr_parts:
            try:
                module_obj = getattr(module_obj, attr_name)
                module_str = '.'.join([module_str, attr_name])
            except AttributeError:
                raise ImportError(f"Cannot load {attr_name} from {module_str}")
        return module_obj

    parts = target.split('.')
    assert parts

    # Base import - check installation
    try:
        obj = import_module(parts[0])
        if len(parts) == 1:
            return obj
    except ImportError:
        raise ImportError(f"Cannot import base package '{parts[0]}'. Please check pip list or $PYTHONPATH.")

    # Dive further
    module_obj, module_str, attr_path = _import_deepmost_module(target)
    obj = _load_attribute(module_obj, module_str, attr_path)
    return obj


def resolve_module_instantiation(subconfig: dict):
    """Search _target_ key and instantiate recursively.
    """
    assert isinstance(subconfig, dict)

    module = None
    if subconfig.get('_target_'):
        module = import_object(subconfig.pop('_target_'))
    for k in subconfig.keys():
        if not isinstance(subconfig[k], dict):
            continue
        subconfig[k] = resolve_module_instantiation(subconfig[k])  # with _target_ popped off
    if module:
        return module(**subconfig)
    return subconfig
