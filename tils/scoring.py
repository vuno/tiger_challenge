from typing import List, Dict

import numpy as np


def calculate_pseudo_tils_score(
    patch_scalar_vals: List[Dict]
) -> float:
    # patch-wise ratio
    patch_ratio_sum = 0

    # wSI-wise ratio
    detect_count_sum = 0
    stroma_area_sum = 0
    for patch_scalar_val in patch_scalar_vals:
        patch_ratio_sum += (patch_scalar_val['detect_count'] / patch_scalar_val['peritumor_stroma_area']
                            if patch_scalar_val['peritumor_stroma_area'] != 0 else 0)

        detect_count_sum += patch_scalar_val['detect_count']
        stroma_area_sum += patch_scalar_val['peritumor_stroma_area']

    avg_of_patch_ratio = patch_ratio_sum / len(patch_scalar_vals) if len(patch_scalar_vals) != 0 else 0
    wsi_ratio = detect_count_sum / stroma_area_sum if stroma_area_sum != 0 else 0

    pseudo_tils_score = (avg_of_patch_ratio + wsi_ratio) / 2

    return pseudo_tils_score


def rescale_pseudo_tils_score_to_final_tils_score(
    pseudo_tils_score: float,
    tils_score_min: int = 1,
    tils_score_max: int = 95,
    tils_score_weight_factor: float = 671,
) -> int:

    if pseudo_tils_score == float('inf'):
        tils_score = tils_score_max
    elif pseudo_tils_score == float('-inf'):
        tils_score = tils_score_min
    else:
        tils_score = round(pseudo_tils_score * tils_score_weight_factor)
        tils_score = np.clip(tils_score, tils_score_min, tils_score_max)

    return tils_score
