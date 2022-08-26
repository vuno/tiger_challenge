from typing import List, Tuple

import cv2
import numpy as np


def get_n_boundary_pts(
    mask: np.ndarray,
    num_of_boundary_pts: int,
) -> List[Tuple[int, int]]:
    # find boundary
    cntrs, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Collect points at different contours into a single list.
    pts = []
    for cntr in cntrs:
        pts.extend(cntr.squeeze().tolist())

    # get margin of contour points to skip
    total_cntr_pts = len(pts)
    skip_pts = round(total_cntr_pts / num_of_boundary_pts)

    # select N points at equal spacing
    selected_pts = pts[::skip_pts]

    return selected_pts


def get_top_k_high_area_patches(
    info_mask: np.ndarray,
    candid_center_pts: List[Tuple[int, int]],
    patch_size: int,
    k: int,
) -> List[Tuple[int, int]]:
    # calculate mask patch area for each center point
    info_areas = []
    for center_x, center_y in candid_center_pts:
        info_mask_patch = extract_patch(
            info_mask,
            center_x,
            center_y,
            patch_size)
        info_area = np.sum(info_mask_patch)
        info_areas.append(info_area)

    # sort center point list based on mask area - high area comes first.
    patch_rank = np.argsort(info_areas)[::-1]
    candid_center_pts_sorted = (np.array(candid_center_pts)[patch_rank]).tolist()

    # select top k
    top_k_pts = candid_center_pts_sorted[:k]

    return top_k_pts


def extract_patch(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    patch_size: int,
) -> np.ndarray:
    # Calculate top left coordinates of the patch
    top_left_x = round(center_x - (patch_size / 2))
    top_left_y = round(center_y - (patch_size / 2))
    bottom_right_x = round(center_x + (patch_size / 2))
    bottom_right_y = round(center_y + (patch_size / 2))

    # Clip patch coordinates within image
    image_h, image_w = image.shape[:2]

    top_left_x = np.clip(top_left_x, 0, image_w)
    top_left_y = np.clip(top_left_y, 0, image_h)
    bottom_right_x = np.clip(bottom_right_x, 0, image_w)
    bottom_right_y = np.clip(bottom_right_y, 0, image_h)

    return image[top_left_y: bottom_right_y,
                 top_left_x: bottom_right_x]
