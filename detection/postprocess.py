from typing import List
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ensemble_boxes import nms

import sahi


def sahi_result_postprocess(result: sahi.prediction.PredictionResult) -> sahi.prediction.PredictionResult:
    for detect_object in result.object_prediction_list:
        w = detect_object.bbox.maxx - detect_object.bbox.minx
        h = detect_object.bbox.maxy - detect_object.bbox.miny
        if ((w < 9) and (h < 9)) or \
            ((w > 20) and (h > 20)) or \
                (w / h < 4 / 7 and w / h > 7 / 4):
            result.object_prediction_list.remove(detect_object)

    return result


class TNBCPostprocess:
    def __init__(
        self,
        cfg,
        bboxes_list: List,
        scores_list: List,
        labels_list: List = None,
        image: np.ndarray = None,
        mask: np.ndarray = None,
    ):
        '''
            Initialize postprocess hparameters and additional parameters.
            :param cfg: configuration instance for postprocessing.
            :param bboxes_list: collections of bboxes of each detector. len(bboxes_list) == number of detectors
            :param scores_list: collections of scores(confidence) of each detector. len(scores_list) == number of detectors
            :param labels_list: dummy list for normal nms.
            :param image: inference raw image.
            :param mask: semantic segmentation model inference result.
            :return: None
        '''
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        if mask is not None:
            self.mask = mask
        self.bboxes_list = bboxes_list
        self.scores_list = scores_list
        self.labels_list = labels_list

        self._EMPTY_DETECTION_RESULT = [], [], []

        self.adjust_list = cfg._CONF_ADJUST_HPARAM
        self.boosting_dict = cfg._ENSEMBLE_POST_HPARAM
        self.class_th = cfg._SEG_POSTPROCESS_TH
        self.eccentricity_threshold = cfg._ECC_TH
        self.ecc_conf_multiplier = cfg._ECC_CONF_MULT
        self._IOU_THR = 0.3

    def conf_adjust(
        self,
    ):
        '''
            Adjust confidence scores between detecting results
            :return: None
        '''
        new_scores_list = []
        for i, scores in enumerate(self.scores_list):
            scores += self.adjust_list[i]
            new_scores_list.append(np.clip(scores, 0, 1))
        self.scores_list = new_scores_list
        del new_scores_list

    def normal_nms(
        self
    ):
        '''
            Ordinary NMS for object detection with IOU
            self.bbox_list -> self.x_coords, self. y_coords
            self.scores_list -> self.scores
        '''

        for bboxes in self.bbox_list:
            bboxes[:, 2] += bboxes[:, 0]  # x2 = x1 + width
            bboxes[:, 3] += bboxes[:, 1]  # y2 = y1 + height
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / self.width  # Normalize x1, x2
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / self.height  # Normalize y1, y2

        if len(self.boxes_list) > 1:
            if len(self.boxes_list) == 0:
                return self._EMPTY_DETECTION_RESULT
            # Perform NMS
            bboxes, scores, _ = nms(self.boxes_list, self.scores_list, self.labels_list,
                                    weights=None, iou_thr=self._IOU_THR)
        else:
            if len(self.boxes_list) > 0:
                bboxes, scores = self.boxes_list[0], self.scores_list[0]
            else:
                return self._EMPTY_DETECTION_RESULT

        f_bboxes = bboxes.copy()
        f_bboxes[:, [0, 2]] = f_bboxes[:, [0, 2]] * self.width
        f_bboxes[:, [1, 3]] = f_bboxes[:, [1, 3]] * self.height
        f_bboxes = np.around(f_bboxes)
        f_bboxes, self.scores = self.bbox_postprocess(f_bboxes, scores)

        if len(f_bboxes) == 0:
            return self._EMPTY_DETECTION_RESULT

        # Compute center point coordinates
        hw_bbox = (f_bboxes[:, 2:] - f_bboxes[:, :2]) // 2
        cordi = f_bboxes[:, :2] + hw_bbox
        self.x_coords = cordi[..., 0].tolist()
        self.y_coords = cordi[..., 1].tolist()

    def bbox_postprocess(
        self,
        bboxes: List,
        scores: List
    ) -> List:
        '''
            Remove too large or too small bboxes. Only use with normal NMS
            :return: bboxes -> bboxes
                     scores -> scores
        '''
        bboxes = bboxes.tolist()
        scores = scores.tolist()
        for bbox in bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            # remove bbox if too small or too big or not close to sqaure (where height is close width)
            if ((w < 11) and (h < 11)) or \
                ((w >= 20) and (h >= 20)) or \
                    (w / h < 2 / 3 and w / h > 3 / 2):
                remove_index = bboxes.index(bbox)
                del bboxes[remove_index]
                del scores[remove_index]

        return np.array(bboxes), np.array(scores)

    def center_nms(
        self,
        radius: float = 8,
    ) -> List:
        '''
            Conduct NMS with centerpoint
            :param rdius: distance of each centerpoint to conduct NMS.
            :hparam boosting_dict: The more bboxes detected by multiple detectors, the higher the score is adjusted.
            :return: self.bbox_list -> self.x_coords, self. y_coords
                     self.scores_list -> self.scores
        '''
        for bboxes in self.bboxes_list:
            bboxes[:, 0] += bboxes[:, 2] / 2
            bboxes[:, 1] += bboxes[:, 3] / 2

        if len(self.bboxes_list) == 0:
            self.x_coords = np.array([])
            self.y_coords = np.array([])
            self.scores = np.array([])

            return
        else:
            mixed_centers = np.concatenate([bboxes[:, :2] for bboxes in self.bboxes_list], axis=0)
            mixed_scores = np.concatenate([scores for scores in self.scores_list], axis=0)
        mixed_centers = np.around(mixed_centers)
        sort_ind = np.argsort(mixed_scores)
        sorted_centers = mixed_centers[sort_ind][::-1]
        sorted_scores = mixed_scores[sort_ind][::-1]

        remove_idx = set([])
        for i, tg_center in enumerate(sorted_centers):
            # skip nms for centerpoints to be removed
            if i in remove_idx:
                continue
            # compute euclidian distance
            tg_distance = np.sqrt(np.sum((tg_center - sorted_centers) ** 2, axis=1))
            tg_distance[i] = radius + 1
            idx = np.where(tg_distance < radius)[0]
            # remove centerpoint from single model with low confidence
            if len(self.bboxes_list) > 1:
                if len(idx) == 1:
                    sorted_scores[i] *= self.boosting_dict['decrease']  # -= ensemble_decrease
                elif len(idx) == 2:
                    sorted_scores[i] *= self.boosting_dict['boost_2']  # += ensemble_boost
                elif len(idx) >= 3:
                    sorted_scores[i] *= self.boosting_dict['boost_3+']  # += ensemble_boost
                remove_idx.update(idx)

        if len(remove_idx) != 0:
            sorted_centers = np.delete(sorted_centers, np.array(list(remove_idx)), axis=0)
            sorted_scores = np.delete(sorted_scores, np.array(list(remove_idx)), axis=0)

        sorted_scores = np.clip(sorted_scores, 0, 1)

        self.x_coords = sorted_centers[:, 0]
        self.y_coords = sorted_centers[:, 1]
        self.scores = sorted_scores

    def seg_postprocess(
        self,
    ) -> List:
        '''
            Remove and adjust confidence with segmentation result
            :hparam class_th: bbox scores' threshold. Depends on segmentation reuslt's class.
            :return: self.x_coords, self. y_coords -> self.x_coords, self. y_coords
                     self.scores -> self.scores
        '''
        self._dilate_seg_mask()

        del_ind_list = []
        for i, (x, y, score) in enumerate(zip(self.x_coords, self.y_coords, self.scores)):
            try:
                point_class = self.mask[round(y), round(x)]
            except:
                del_ind_list.append(i)

            if score < self.class_th[point_class]:
                del_ind_list.append(i)
                continue
            elif point_class == 3:
                score *= .4

            if self._remove_edge(x, y):
                score *= .45

        self.x_coords = np.delete(np.array(self.x_coords), del_ind_list).tolist()
        self.y_coords = np.delete(np.array(self.y_coords), del_ind_list).tolist()
        self.scores = np.delete(np.array(self.scores), del_ind_list).tolist()

        del del_ind_list

    def eccentricity_filtering(
        self,
        hematoxylin_threshold: float = 0.4,
    ) -> List:
        """
            Filter a lymphocyte based on the eccentricity of the largest component in 16x16 RGB image from the center point.
            :param: img (np.array): RGB tissue image.
            :param: center_x (List): a center_x list of detected lympocytes.
            :param: center_y (List): a center_y list of detected lympocytes.
            :param: scores (List): a confidence score list of detected lympocytes.

            :return:Tuple[List, List, List]: (center_x, center_y, scores) of filtered lymphocytes.
        """
        patches = [self._crop_patch_from_centerpoints(self.image, x, y) for (x, y) in zip(self.x_coords, self.y_coords)]
        patches_hema = [self._get_binary_hematoxylin(x, hematoxylin_threshold) for x in patches]
        patches_mask = [self._get_largest_component(x) for x in patches_hema]
        ecc = np.array([self._get_eccentricity(x) for x in patches_mask])
        # logger.debug(f"before ecc: {len(ecc)}")
        idx = np.where(ecc > self.eccentricity_threshold)
        # logger.debug(f"adj ecc: {len(idx[0])}")

        scores = np.array(self.scores)
        scores[idx] *= self.ecc_conf_multiplier

        self.scores = scores.tolist()

    def _crop_patch_from_centerpoints(self, img, x, y):
        # crop patch image from img as centerpoint of (y,x)
        _PS = 16  # patch_size #must be even number

        lt_y = int(np.round(y)) - int(np.round(_PS/2))
        lt_x = int(np.round(x)) - int(np.round(_PS/2))
        pad_ty = 0 if lt_y > 0 else abs(lt_y)
        pad_lx = 0 if lt_x > 0 else abs(lt_x)
        patch = img[pad_ty+lt_y:lt_y+_PS, pad_lx+lt_x:lt_x+_PS, :]
        pad_by = 0 if patch.shape[0] == _PS else (_PS - pad_ty - patch.shape[0])
        pad_rx = 0 if patch.shape[1] == _PS else (_PS - pad_lx - patch.shape[1])

        return np.pad(patch, ((pad_ty, pad_by), (pad_lx, pad_rx), (0, 0)), 'constant', constant_values=((255, 255), (255, 255), (255, 255)))

    def _get_binary_hematoxylin(self, img, threshold):
        try:
            _, hema, _ = normalize_staining(img)
        except np.linalg.LinAlgError as ex:
            hema = img

        hema = (255 - cv2.cvtColor(hema, cv2.COLOR_RGB2GRAY)) / 255.0
        return np.expand_dims(hema > threshold, -1)

    def _get_largest_component(self, mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            areas = []
            out = np.zeros_like(mask, dtype=np.uint8)
            for cnt in contours:
                if len(cnt) < 5:
                    areas.append(0)
                    continue
                _area = cv2.contourArea(cnt)
                areas.append(_area)
            idx = np.argmax(areas)
            out = cv2.fillPoly(out, [contours[idx]], 1)
            return out
        else:
            return mask

    def _get_eccentricity(self, mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        ret = [[0.0, 2.0]]  # [area, eccentricity]
        for cnt in contours:
            if len(cnt) < 5:
                ret.append([0.0, 2.0])
                continue
            (x, y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(cnt)
            a = majorAxisLength / 2
            b = minorAxisLength / 2

            eccentricity = np.round(np.sqrt(np.power(a, 2) - np.power(b, 2))/a, 2)
            ret.append([cv2.contourArea(cnt), eccentricity])

        ret = np.array(ret)
        return ret[ret[:, 0].argsort()][::-1][0][1]

    def _dilate_seg_mask(
        self,
        target_label: int = 1,
        kernel: list = [5, 5],
        iteration: int = 1,
        append_channel: bool = True
    ):

        if target_label in np.unique(self.mask):
            # dilate tumor mask
            tumor_img = (self.mask == target_label).copy().astype(np.uint8)
            d_tumor_img = cv2.dilate(tumor_img, np.ones(kernel, np.uint8), iteration)
            d_tumor_img -= tumor_img
            if append_channel == True:
                self.mask = np.where(d_tumor_img == 1, 4, self.mask)
            else:
                self.mask = np.where(d_tumor_img == 1, d_tumor_img, self.mask)

    def _remove_edge(
        self,
        x: int,
        y: int,
        edge_thick: int = 6
    ):
        if (x < edge_thick) or \
            (x > (self.width - edge_thick)) or \
            (y < edge_thick) or \
                (y > (self.height - edge_thick)):
            return True

        return False

    def visualize(
        self,
        visualize_fname: str
    ):
        img = self.image[..., ::-1].copy()
        rcolor = (0, 255, 0)
        n = 2
        for xi, yi in zip(self.x_coords, self.y_coords):
            xi, yi = map(round, [xi, yi])
            img = cv2.rectangle(img, (xi - n, yi - n), (xi + n, yi + n), rcolor, 1)
        cv2.imwrite(Path('./') / visualize_fname, img)


def normalize_staining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float64)+1)/Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile)  # +'.png')
        # Image.fromarray(H).save(saveFile+'_H.png')
        # Image.fromarray(E).save(saveFile+'_E.png')

    return Inorm, H, E
