from pathlib import Path
import cv2

from tqdm import tqdm
import multiresolutionimageinterface as mir
import numpy as np
import torch

import configuration.configuration_loading as config_load
import detection.detection_inference as detect_infer
import pipeline.timer as timer
import segmentation.inference as seg_infer
import tils.patch_selection as patch_select
import tils.scoring as scoring

_CFG = {
    'CFG_PATH': {
        'DETECT': Path("/vuno/configuration/detection.yaml"),
        'SEG': Path("/vuno/segmentation/inference-final-model.yaml"),
    },
    'CKPT_FOLDER': {
        'DETECT': Path("/vuno/pretrained_weights/detection/"),
        'SEG': Path("/vuno/pretrained_weights/segmentation/"),
    },
    'MPP': {
        'L0': 0.5,
        'SEG_INFER': 4,
    },
    'TILS_SCORE': {
        'MIN': 1,
        'MAX': 95,
        'WEIGHT_FACTOR': 671,
    },
    'PERITUMOR_BAND_WIDTH': 100,  # um
    'INFO_PATCH_SELECT': {
        'PATCH_SIZE_AT_L0_MPP': 2048,
        'NUM_OF_CAND_PATCHES': 50,
        'TOP_K': 20,
    },
    'TIME_LIMIT': "01:45:00",
}


def run_tils_pipeline(
    wsi_mri: mir.MultiResolutionImage,
) -> None:
    loop_timer = timer.Timer(_CFG['TIME_LIMIT'], auto_start=True)

    seg_pipeline = _build_models()

    wsi_at_seg_infer_mpp = _read_wsi_at_seg_infer_mpp(wsi_mri)
    seg_mask = seg_pipeline(img=wsi_at_seg_infer_mpp)

    tumor_mask, psb_mask, has_valid_tils_region = seg_pipeline.analyze_tumor_stroma_area(
        seg_mask=seg_mask,
        seg_mask_mpp=_CFG['MPP']['SEG_INFER'],
        dist_from_tumor_in_um=_CFG['PERITUMOR_BAND_WIDTH'],
    )

    if not has_valid_tils_region:
        print("has no valid tils region")

        return _CFG['TILS_SCORE']['MIN']
    else:
        # get tumor boundary points
        candid_center_pts = patch_select.get_n_boundary_pts(
            mask=tumor_mask,
            num_of_boundary_pts=_CFG['INFO_PATCH_SELECT']['NUM_OF_CAND_PATCHES'],
        )

        patch_size_at_l0_mpp = _CFG['INFO_PATCH_SELECT']['PATCH_SIZE_AT_L0_MPP']
        patch_size_at_seg_infer_mpp = _rescale_l0_to_seg_infer(patch_size_at_l0_mpp)

        # filter center points
        info_patch_center_pts_at_seg_infer_mpp = patch_select.get_top_k_high_area_patches(
            info_mask=psb_mask,
            candid_center_pts=candid_center_pts,
            patch_size=patch_size_at_seg_infer_mpp,
            k=_CFG['INFO_PATCH_SELECT']['TOP_K'])

        wsi_w_at_l0_mpp, wsi_h_at_l0_mpp = wsi_mri.getDimensions()

        tils_related_info_of_patches = []

        # compute patch-wise scalar values
        for (center_x_at_seg_infer_mpp, center_y_at_seg_infer_mpp) in tqdm(info_patch_center_pts_at_seg_infer_mpp):
            psb_patch = patch_select.extract_patch(
                image=psb_mask,
                center_x=center_x_at_seg_infer_mpp,
                center_y=center_y_at_seg_infer_mpp,
                patch_size=patch_size_at_seg_infer_mpp)

            # skip if there is no peritumoral stroma area
            if not np.any(psb_patch):
                continue

            # extract segmentation mask for detection postprocessing
            seg_mask_patch_at_seg_infer_mpp = patch_select.extract_patch(
                image=seg_mask,
                center_x=center_x_at_seg_infer_mpp,
                center_y=center_y_at_seg_infer_mpp,
                patch_size=patch_size_at_seg_infer_mpp,
            )
            seg_mask_patch_at_seg_l0_mpp = cv2.resize(
                src=seg_mask_patch_at_seg_infer_mpp,
                dsize=(patch_size_at_l0_mpp, patch_size_at_l0_mpp),
                interpolation=cv2.INTER_NEAREST)

            # extract image
            center_x_at_l0_mpp = _rescale_seg_infer_to_l0(center_x_at_seg_infer_mpp)
            center_y_at_l0_mpp = _rescale_seg_infer_to_l0(center_y_at_seg_infer_mpp)
            start_x_at_l0 = center_x_at_l0_mpp - round(patch_size_at_l0_mpp / 2)
            start_y_at_l0 = center_y_at_l0_mpp - round(patch_size_at_l0_mpp / 2)
            start_x_at_l0 = np.clip(start_x_at_l0, 0, wsi_w_at_l0_mpp - 1).item()
            start_y_at_l0 = np.clip(start_y_at_l0, 0, wsi_h_at_l0_mpp - 1).item()

            image_patch_at_l0_mpp = wsi_mri.getUCharPatch(
                startX=start_x_at_l0,
                startY=start_y_at_l0,
                width=patch_size_at_l0_mpp,
                height=patch_size_at_l0_mpp,
                level=0,
            )

            if image_patch_at_l0_mpp.shape[:2] != seg_mask_patch_at_seg_l0_mpp.shape[:2]:
                continue

            detection = _run_detection(image_patch_at_l0_mpp, seg_mask_patch_at_seg_l0_mpp)

            tils_related_info_of_patches.append({
                "detect_count": len(detection),
                "peritumor_stroma_area": np.sum(psb_patch),
            })

            if loop_timer.exceeded_time_limit():
                print("Time limit exceeded")
                break

        # compute final tils score
        pseudo_tils_score = scoring.calculate_pseudo_tils_score(tils_related_info_of_patches)
        final_tils_score = scoring.rescale_pseudo_tils_score_to_final_tils_score(
            pseudo_tils_score=pseudo_tils_score,
            tils_score_min=_CFG['TILS_SCORE']['MIN'],
            tils_score_max=_CFG['TILS_SCORE']['MAX'],
            tils_score_weight_factor=_CFG['TILS_SCORE']['WEIGHT_FACTOR'],
        )

        print(f"{pseudo_tils_score}")
        print(f"{final_tils_score}")

    return final_tils_score


def _build_models(
) -> None:
    print(f"Pytorch GPU available: {torch.cuda.is_available()}")

    detect_config = config_load.load_configuration(_CFG['CFG_PATH']['DETECT'])
    detect_infer.build_detect_model_pool(detect_config)

    seg_pipeline = seg_infer.TIGERSegmentationPipeline(_CFG['CFG_PATH']['SEG'])

    return seg_pipeline


def _read_wsi_at_seg_infer_mpp(
    wsi_mri: mir.MultiResolutionImage,
    fast_read_level: int = 3,
) -> np.ndarray:
    wsi_w_at_fast_read_level, wsi_h_at_fast_read_level = wsi_mri.getLevelDimensions(fast_read_level)
    wsi_at_fast_read_level = wsi_mri.getUCharPatch(
        startX=0,
        startY=0,
        width=wsi_w_at_fast_read_level,
        height=wsi_h_at_fast_read_level,
        level=fast_read_level,
    )

    # resize to shape at seg infer mpp
    wsi_w_at_l0_mpp, wsi_h_at_l0_mpp = wsi_mri.getDimensions()
    wsi_w_at_seg_infer_mpp, wsi_h_at_seg_infer_mpp = (
        _rescale_l0_to_seg_infer(wsi_w_at_l0_mpp),
        _rescale_l0_to_seg_infer(wsi_h_at_l0_mpp)
    )
    wsi_at_seg_infer_mpp = cv2.resize(
        src=wsi_at_fast_read_level,
        dsize=(
            wsi_w_at_seg_infer_mpp,
            wsi_h_at_seg_infer_mpp
        ),
        interpolation=cv2.INTER_CUBIC)

    return wsi_at_seg_infer_mpp


def _rescale_l0_to_seg_infer(
    num: int,
) -> int:
    return round(num * (_CFG['MPP']['L0'] / _CFG['MPP']['SEG_INFER']))


def _rescale_seg_infer_to_l0(
    num: int,
) -> int:
    return round(num * (_CFG['MPP']['SEG_INFER'] / _CFG['MPP']['L0']))

@timer.timing
def _run_detection(
    image_tile: np.ndarray,
    seg_mask: np.ndarray,
) -> np.ndarray:
    # run detection
    x_coords, y_coords, probs = detect_infer.detect_cells(image=image_tile, mask=seg_mask)

    # format detection as tuple
    detections = list(zip(x_coords, y_coords, probs))

    return detections
