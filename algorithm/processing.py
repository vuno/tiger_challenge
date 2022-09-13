from numbers import Number
from typing import List, Tuple

import numpy as np

import algorithm.gc_io as gc_io
import algorithm.rw as rw
import pipeline.tils_pipeline as tils_pipeline


def process(
) -> None:
    gc_io.initialize_output_folders()

    # open wsi
    wsi_filepath = gc_io.get_image_path_from_input_folder()
    wsi_mri = rw.open_multiresolutionimage_image(wsi_filepath)

    print(f"Input WSI: {wsi_filepath}")

    # get image info
    tile_size = rw.WRITING_TILE_SIZE
    dimensions = wsi_mri.getDimensions()
    spacing = wsi_mri.getSpacing()

    # create writers
    seg_writer = rw.SegmentationWriter(
        output_path=gc_io.TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detect_writer = rw.DetectionWriter(gc_io.TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = rw.TilsScoreWriter(gc_io.TMP_TILS_SCORE_PATH)

    # save dummy results for segmentation and detection
    empty_seg_mask, empty_detection = _create_empty_seg_and_detect_outputs(tile_size)
    seg_writer.write_segmentation(tile=empty_seg_mask, x=0, y=0)
    detect_writer.write_detections(detections=empty_detection, spacing=spacing, x_offset=0, y_offset=0)

    tils_score = tils_pipeline.run_tils_pipeline(wsi_mri, seg_writer, detect_writer)

    # write tils score
    tils_score_writer.set_tils_score(tils_score=tils_score)

    # save result
    seg_writer.save()
    detect_writer.save()
    tils_score_writer.save()

    gc_io.copy_data_to_output_folders()


def _create_empty_seg_and_detect_outputs(
    tile_size: int,
) -> Tuple[np.ndarray, List[List[Number]]]:
    empty_seg_mask = np.zeros((tile_size, tile_size))

    empty_detections = list(zip([], [], []))

    return empty_seg_mask, empty_detections
