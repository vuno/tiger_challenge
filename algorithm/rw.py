"""Reading and Writing

This file contains utilities for reading and writing.

It contains a helper function to easily open a multiresolution image.
Furthermore, it contains Writer classes that ease the process of writing prediction masks, detection output, and a tils score.

"""

from pathlib import Path
from typing import List, Protocol, Union
import json

import multiresolutionimageinterface as mir
import numpy as np

WRITING_TILE_SIZE = 4096  # should be a power of 2


def open_multiresolutionimage_image(
    path: Path,
) -> mir.MultiResolutionImage:
    """Opens a multiresolution image with ASAP python bindings

    Args:
        path (Path): path to image

    Raises:
        IOError: raises when opened image is None

    Returns:
        MultiResolutionImage: opened multiresolution image
    """

    reader = mir.MultiResolutionImageReader()
    image = reader.open(str(path))
    if image is None:
        raise IOError(f"Error opening image: {path}, image is None")
    
    return image


class Writer(Protocol):
    def save(
        self,
    ) -> None:
        ...


class SegmentationWriter:
    def __init__(
        self,
        output_path: Path,
        tile_size: int,
        dimensions: tuple,
        spacing: tuple,
    ) -> None:
        """Writer for writing and saving multiresolution mask/prediction images

        Args:
            output_path (Path): path to output file
            tile_size (int): tile size used for writing image tiles
            dimensions (tuple): dimensions of the output image
            spacing (tuple): base spacing of the output image
        """

        if output_path.suffix != '.tif':
            output_path = output_path / '.tif'

        self._writer = mir.MultiResolutionImageWriter()
        self._writer.openFile(str(output_path))
        self._writer.setTileSize(tile_size)
        self._writer.setCompression(mir.LZW)
        self._writer.setDataType(mir.UChar)
        self._writer.setInterpolation(mir.NearestNeighbor)
        self._writer.setColorType(mir.Monochrome)
        self._writer.writeImageInformation(dimensions[0], dimensions[1])
        pixel_size_vec = mir.vector_double()
        pixel_size_vec.push_back(spacing[0])
        pixel_size_vec.push_back(spacing[1])
        self._writer.setSpacing(pixel_size_vec)

    def write_segmentation(
        self,
        tile: np.ndarray,
        x: Union[int, float],
        y: Union[int, float],
    ) -> None:
        if tile.shape[0] != WRITING_TILE_SIZE or tile.shape[1] != WRITING_TILE_SIZE:
            raise ValueError(
                f"Dimensions of tile {tile.shape} is incompatible with writing tile size {WRITING_TILE_SIZE}.")
        self._writer.writeBaseImagePartToLocation(tile.flatten().astype('uint8'), x=int(x), y=int(y))

    def save(
        self,
    ) -> None:
        self._writer.finishImage()


class DetectionWriter:
    """Writes detection to json format that can be handled by grand challenge"""

    def __init__(
        self,
        output_path: Path,
    ) -> None:
        """init

        Args:
            output_path (Path): path to json output file
        """

        if output_path.suffix != ".json":
            output_path = output_path / ".json"

        self._output_path = output_path
        self._data = {
            'type': "Multiple points",
            'points': [],
            'version': {
                'major': 1,
                'minor': 0,
            },
        }

    @property
    def detections(
        self,
    ) -> None:
        detections = []
        for point in self._data['points']:
            x, y = point['point'][:2]
            probability = point['probability']
            detections.append((x, y, probability))

        return detections

    def write_detections(
        self,
        detections: List[tuple],
        spacing: tuple,
        x_offset: Union[int, float],
        y_offset: Union[int, float],
    ) -> None:
        for detection in detections:
            x, y, probalility = detection
            x += x_offset
            y += y_offset
            # world coordinates in grand challenge
            x = x * spacing[0] / 1000
            y = y * spacing[1] / 1000
            self.add_point(x=x, y=y, probalility=probalility)

    def add_point(
        self,
        x: Union[int, float],
        y: Union[int, float],
        probalility: float,
    ) -> None:
        _3d_space_value = 0.5009999871253967
        point = {
            'point': [float(x), float(y), _3d_space_value],
            'probability': probalility,
        }
        self._data['points'].append(point)

    def save(
        self
    ) -> None:
        with open(self._output_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=4)


class TilsScoreWriter:
    def __init__(
        self,
        output_path: Path,
    ) -> None:
        """Writer for keeping track and saving a tils score.

        Args:
            output_path (Path): json file output path
        """

        if output_path.suffix != ".json":
            output_path = output_path / ".json"

        self._output_path = output_path
        self._tils_score = None

    def set_tils_score(
        self,
        tils_score: Union[int, float],
    ) -> None:
        self._tils_score = tils_score

    def save(
        self,
    ) -> None:
        with open(self._output_path, 'w') as file1:
            file1.write(str(self._tils_score))
