""" GrandChallenge Input/Output (gcio)

In this file settings concerning folders and paths for reading and writing on GrandChallenge are defined.
Note that these settings are moslty specific to the GrandChallenge Tiger Challenge.

"""

from pathlib import Path
from shutil import copy


# Grand Challenge paths
_GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH = Path("/output/images/breast-cancer-segmentation-for-tils/segmentation.tif")
_GRAND_CHALLENGE_DETECTION_OUTPUT_PATH = Path("/output/detected-lymphocytes.json")
_GRAND_CHALLENGE_TILS_SCORE_PATH = Path("/output/til-score.json")

# Temporary paths
_TMP_FOLDER = Path("/home/user/tmp")
TMP_SEGMENTATION_OUTPUT_PATH = (_TMP_FOLDER / _GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH.name)
TMP_DETECTION_OUTPUT_PATH = _TMP_FOLDER / _GRAND_CHALLENGE_DETECTION_OUTPUT_PATH.name
TMP_TILS_SCORE_PATH = _TMP_FOLDER / _GRAND_CHALLENGE_TILS_SCORE_PATH.name

# Grand Challenge folders were input files can be found
_GRAND_CHALLENGE_IMAGE_FOLDER = Path("/input/")
_GRAND_CHALLENGE_MASK_FOLDER = Path("/input/images/")

# Grand Challenge suffixes for required files
_GRAND_CHALLENGE_IMAGE_SUFFIX = ".tif"
_GRAND_CHALLENGE_MASK_SUFFIX = ".tif"


def initialize_output_folders(
) -> None:
    """This function initialize all output folders for grandchallgenge output as well as tempory folder"""
    _GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GRAND_CHALLENGE_DETECTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GRAND_CHALLENGE_TILS_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TMP_FOLDER.mkdir(parents=True, exist_ok=True)


def _get_file_from_folder(
    folder: Path,
    suffix: str,
) -> Path:
    """Gets this first file in a folder with the specified suffix

    Args:
        folder (Path): folder to search for files
        suffix (str): suffix for file  to search for

    Returns:
        Path: path to file
    """

    return list(Path(folder).glob("*" + suffix))[0]


def get_image_path_from_input_folder(
) -> Path:
    """Gets a test image which needs to be processed for the Tiger Challenge

    Returns:
        Path: path to multiresolution image from the test set
    """

    return _get_file_from_folder(_GRAND_CHALLENGE_IMAGE_FOLDER, _GRAND_CHALLENGE_IMAGE_SUFFIX)


def get_tissue_mask_path_from_input_folder(
) -> Path:
    """Gets the tissue mask for the corresponding test image that needs to be processed

    Returns:
        Path: path to tissue tissue mask
    """

    return _get_file_from_folder(_GRAND_CHALLENGE_MASK_FOLDER, _GRAND_CHALLENGE_MASK_SUFFIX)


def copy_data_to_output_folders(
) -> None:
    """Copies all temporary files to the (mandatory) output files/folders"""

    # copy segmentation tif to grand challenge
    copy(TMP_SEGMENTATION_OUTPUT_PATH, _GRAND_CHALLENGE_SEGMENTATION_OUTPUT_PATH)
    # copy detections json to grand challenge
    copy(TMP_DETECTION_OUTPUT_PATH, _GRAND_CHALLENGE_DETECTION_OUTPUT_PATH)
    # copy tils score json to grand challenge
    copy(TMP_TILS_SCORE_PATH, _GRAND_CHALLENGE_TILS_SCORE_PATH)
