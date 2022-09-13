from abc import abstractmethod
from argparse import Namespace
import os
from pathlib import Path
from typing import Callable, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from tqdm import tqdm
import ttach

import pipeline.timer as timer
import segmentation.utils as utils

_RAW_CLASS_INFO = {
    'stroma': 0,
    'tumor': 1,
    'rest': 2,
}

FINAL_CLASS_INFO = {
    'ignore': 0,
    'tumor': 1,
    'stroma': 2,
    'rest': 3,
}

_FINAL_CLASS_REMAP = {
    0: 2,  # stroma
    1: 1,  # tumor
    2: 3,  # rest
}


class ComposeModel(torch.nn.Module):
    def __init__(self,
                 model_collection: dict, ensemble_weights: list, activation_fx: Callable, normalize_ensemble_weights: bool = True) -> None:
        assert len(model_collection) > 0
        assert len(model_collection) == len(ensemble_weights)
        assert set(model_collection.keys()) == set(ensemble_weights.keys())

        super(ComposeModel, self).__init__()

        self.model_collection = model_collection
        self.ensemble_weights = ensemble_weights
        self.activation_fx = activation_fx
        self.normalize_ensemble_weights = normalize_ensemble_weights

        self.model_name_list = sorted(list(self.model_collection.keys()))
        self.ensemble_weights_sum = torch.tensor(list(self.ensemble_weights.values())).sum()

    def cuda(self):
        self.model_collection = {name: model.cuda() for name, model in self.model_collection.items()}
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ret = 0
        with torch.no_grad():
            for model_name in self.model_name_list:
                pred = self.activation_fx(self.model_collection.get(model_name)(x))
                ret += (pred * self.ensemble_weights.get(model_name))
        if self.normalize_ensemble_weights:
            ret /= self.ensemble_weights_sum
        return ret


class PatchifiedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img: np.ndarray, wsize: int, wstride: int, transform: Callable, min_foreground_mask_ratio_to_infer: float, foreground_mask=None) -> None:
        self.img = img
        self.wsize = wsize
        self.wstride = wstride
        self.transform = transform
        self.foreground_mask = foreground_mask
        self.min_foreground_mask_ratio_to_infer = min_foreground_mask_ratio_to_infer

        self.yx_list = [
            (y, x) for y in range(0, img.shape[0] - self.wsize + 1, self.wstride) for x in range(0, img.shape[1] - self.wsize + 1, self.wstride)]
        if self.foreground_mask is not None:
            assert list(self.foreground_mask.shape[:2]) == list(self.img.shape[:2])
            assert 0. <= self.min_foreground_mask_ratio_to_infer <= 1.
            crit_px = round(self.wsize ** 2 * self.min_foreground_mask_ratio_to_infer)
            self.yx_list = [
                (y, x) for y, x in self.yx_list if np.count_nonzero(self.foreground_mask[y:y + self.wsize, x:x + self.wsize]) >= crit_px]

    def __len__(self):
        return len(self.yx_list)

    def __getitem__(self, idx):
        ymin, xmin = self.yx_list[idx]
        img = self.img[ymin:ymin + self.wsize, xmin:xmin + self.wsize]
        return {
            'img': self.transform(image=img)['image'],
            'y': ymin,
            'x': xmin,
        }


def _model_ckpt_init(initial_model: torch.nn.Module, ckpt_fname: Union[Path, str]) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_fname))
    initial_model.load_state_dict(ckpt)
    return initial_model


def _get_activation_fx_softmax_with_temperature(temperature: float) -> Callable:
    def _softmax_with_temperature(x):
        return torch.nn.functional.softmax(x/temperature, dim=1)
    return _softmax_with_temperature


class TIGERSegmentationPipelineBase(Namespace):
    def __init__(self, model_config_fname: Union[Path, str], gpu: str = '0', src_mpp: float = 4.):
        self.model_config_fname = Path(model_config_fname).resolve()
        self.gpu = str(gpu)
        self.src_mpp = src_mpp
        self._config = utils.load_yaml_config(self.model_config_fname)

        super().__init__(**self._config)
        self.build_config()

    @abstractmethod
    def build_config(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, img: np.ndarray, foreground_mask: np.ndarray) -> dict:
        raise NotImplementedError

    def __call__(self, img: np.ndarray, foreground_mask: np.ndarray = None) -> dict:
        return self.predict(img=img, foreground_mask=foreground_mask)


class TIGERSegmentationPipeline(TIGERSegmentationPipelineBase):
    def build_config(self):
        assert self.wsize and self.wstride
        assert self.wsize >= self.wstride

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu

        self.final_class_lut = utils.classmap_dict2lut(
            class_map=_FINAL_CLASS_REMAP, default_value=FINAL_CLASS_INFO['ignore'])
        self.model  # property method, call once in advance

    @property
    def model(self) -> torch.nn.Module:  # Avoid unnecessarily re-loading model
        if hasattr(self, '_model'):
            return self._model
        self._model = utils.resolve_module_instantiation(self.ensemble_model_conf)
        self._model = self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        self._model = ttach.SegmentationTTAWrapper(
            model=self._model, transforms=ttach.Compose([ttach.Rotate90(angles=[0, 90, 180, 270])]), merge_mode="mean")
        self.transform = A.Compose([A.Normalize(), ToTensorV2()])
        return self._model

    def predict(self, img: np.ndarray, foreground_mask: np.ndarray = None) -> np.ndarray:
        # <Pre-processing>
        img = utils.make_lowsatval_img(img)

        # <Inference>
        # NOTE: Now, prediction is multichannel pre-argmax probability map
        pred = self._run_model_on_image(img=img, foreground_mask=foreground_mask)

        # <Post-processing>: probability map
        pred = utils.boost_suboptimal_class(
            prob=pred, competing_idx_list=[_RAW_CLASS_INFO['tumor'], _RAW_CLASS_INFO['rest']],
            boosting_idx=_RAW_CLASS_INFO['rest'], confidence_lowerbound=205)

        # Get single class prediction for each pixel
        # NOTE: Now, prediction is single-channel argmax prediction map
        pred = np.argmax(pred, -1)

        # <Post-processing>: argmax prediction map
        pred, _ = utils.replace_small_object(
            pred=pred,
            src_mpp=self.src_mpp,
            src_idx=_RAW_CLASS_INFO['tumor'],
            dst_idx=_RAW_CLASS_INFO['stroma'],
        )

        # Remap class prediction to final challenge evaluation class
        pred = self.final_class_lut[pred]

        return pred

    def _run_model_on_image(self, img: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
        def _make_linear_kernel(ksize: int, edge_min: float = 0.1, canvas_nch: int = 3) -> np.ndarray:
            lhs_size = ksize // 2
            rhs_size = lhs_size
            if ksize % 2 != 0:
                rhs_size += 1
            kern_1d = np.concatenate(
                [np.linspace(edge_min, 1.0, num=lhs_size, endpoint=True), np.linspace(1.0, edge_min, num=rhs_size, endpoint=True)], axis=0)
            kern_2d = np.outer(kern_1d, kern_1d)
            if canvas_nch > 0:
                kern_2d = np.repeat(kern_2d[..., np.newaxis], canvas_nch, axis=-1)
            return kern_2d

        # Basic setup
        # Prepare prediction canvas
        if max(img.shape[:2]) > self.wsize:
            padsize_h = padsize_w = self.wstride
            img = np.pad(img, ((padsize_h, padsize_h), (padsize_w, padsize_w), (0, 0)), mode='reflect')
            if foreground_mask is not None:
                foreground_mask = np.pad(foreground_mask, ((padsize_h, padsize_h),
                                         (padsize_w, padsize_w)), mode='reflect')
        else:
            padsize_h = self.wsize - img.shape[0]
            padsize_w = self.wsize - img.shape[1]
            img = np.pad(img, ((padsize_h, padsize_h), (padsize_w, padsize_w), (0, 0)), mode='reflect')
            if foreground_mask is not None:
                foreground_mask = np.pad(foreground_mask, ((padsize_h, padsize_h),
                                         (padsize_w, padsize_w)), mode='reflect')

        canvas_shape = list(img.shape[:2]) + [self.out_channels]
        prediction = np.zeros(canvas_shape, dtype=np.float16)

        # Prepare linear_kernel for slow_overlap
        kernel = _make_linear_kernel(ksize=self.wsize, edge_min=0.1, canvas_nch=self.out_channels)
        kernel_canvas = np.full(canvas_shape, 1e-4, dtype=np.float16)

        # Patchify input image
        input_ds = PatchifiedDataset(
            img=img, wsize=self.wsize, wstride=self.wstride, transform=self.transform,
            min_foreground_mask_ratio_to_infer=self.min_foreground_mask_ratio_to_infer, foreground_mask=foreground_mask)
        input_dl = torch.utils.data.DataLoader(
            input_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=False)

        # Setup inplace splicer
        def _splice_single_patch_smoothed(patch_pred, y, x):
            prediction[y:y + self.wsize, x:x + self.wsize, ...] += np.multiply(patch_pred, kernel).astype(np.float16)
            kernel_canvas[y:y + self.wsize, x:x + self.wsize, ...] += kernel

        # Predict
        use_cuda = torch.cuda.is_available()
        with torch.no_grad():
            for batch in tqdm(input_dl, desc="Segmentation"):
                ys, xs = batch['y'], batch['x']
                inputs = batch['img']
                if use_cuda:
                    inputs = inputs.cuda()
                outputs = self.model.forward(inputs).cpu().numpy()
                outputs = outputs.transpose(0, 2, 3, 1)  # NCHW -> NHWC
                _ = [_splice_single_patch_smoothed(output, y, x) for output, y, x in zip(outputs, ys, xs)]

        # Convert outputs to single prediction
        prediction = prediction.squeeze()
        if padsize_h:
            prediction = prediction[padsize_h:-padsize_h, ...]
            kernel_canvas = kernel_canvas[padsize_h:-padsize_h, ...]
        if padsize_w:
            prediction = prediction[:, padsize_w:-padsize_w, ...]
            kernel_canvas = kernel_canvas[:, padsize_w:-padsize_w, ...]

        prediction *= 255
        prediction /= kernel_canvas
        prediction = prediction.clip(max=255.).astype(np.uint8)

        return prediction

    @timer.timing
    def analyze_tumor_stroma_area(
        self,
        seg_mask: np.ndarray,
        seg_mask_mpp: float,
        dist_from_tumor_in_um: float,  # um
    ) -> Tuple[np.ndarray, np.ndarray, bool]:

        stroma_mask = (seg_mask == FINAL_CLASS_INFO['stroma']).astype(np.uint8)

        # remove small object in tumor mask (focus on relatively big blobs)
        seg_mask, tumor_mask_with_small_objects = utils.replace_small_object(
            pred=seg_mask,
            src_mpp=seg_mask_mpp,
            src_idx=FINAL_CLASS_INFO['tumor'],
            dst_idx=FINAL_CLASS_INFO['rest'],
        )

        # check tumor and stroma area
        tumor_mask = (seg_mask == FINAL_CLASS_INFO['tumor']).astype(np.uint8)
        stroma_mask = (seg_mask == FINAL_CLASS_INFO['stroma']).astype(np.uint8)

        peritumor_stroma_band_mask = None
        has_valid_tils_region = False
        if np.any(tumor_mask) and np.any(stroma_mask):
            # check peritumoral stroma area
            disk_selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dist_from_tumor_in_px = int(dist_from_tumor_in_um / seg_mask_mpp)
            tumor_mask_dilated = cv2.dilate(
                src=tumor_mask,
                kernel=disk_selem,
                iterations=dist_from_tumor_in_px,
            )

            peritumor_stroma_band_mask = tumor_mask_dilated * stroma_mask

            has_valid_tils_region = np.any(peritumor_stroma_band_mask)

        return tumor_mask_with_small_objects, peritumor_stroma_band_mask, has_valid_tils_region
