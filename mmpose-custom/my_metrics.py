# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS


def _calc_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray,
                    norm_factor: np.ndarray) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T



def keypoint_epe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """Calculate the end-point error.

    Note:
        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        float: Average end-point error.
    """

    distances = _calc_distances(
        pred, gt, mask,
        np.ones((pred.shape[0], pred.shape[2]), dtype=np.float32))
    distance_valid = distances[distances != -1]
    
    epe_mean = distance_valid.sum() / max(1, len(distance_valid))
    epe_std = np.std(distance_valid)
    
    return epe_mean, epe_std





@METRICS.register_module()
class EPE_my(BaseMetric):
    """EPE evaluation metric.

    Calculate the end-point error (EPE) of keypoints.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        logger.info(f'Evaluating {self.__class__.__name__}...')
        
        """
        N, K, _ = pred_coords.shape
        # set mask=0 when norm_factor==0
        _mask = mask.copy()
        norm_factor = np.ones((pred_coords.shape[0], pred_coords.shape[2]), dtype=np.float32)
        _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

        distances = np.full((N, K), -1, dtype=np.float32)
        # handle invalid values
        norm_factor[np.where(norm_factor <= 0)] = 1e6
        distances[_mask] = np.linalg.norm(
            ((pred_coords - gt_coords) / norm_factor[:, None, :])[_mask], axis=-1)
        
        distances = distances.T
        
        distance_valid = distances[distances != -1]
        epe_mean = distance_valid.sum() / max(1, len(distance_valid))
        epe_std = np.std(distance_valid)
        
        
        
        """
        epe_mean, epe_std = keypoint_epe(pred_coords, gt_coords, mask)
        
        
        logger.info(f'EPE: {epe_mean:.4f}, STD: {epe_std:.4f}')
        metrics = dict()
        metrics['EPE'] = epe_mean
        metrics['EPE_std'] = epe_std
        return metrics
    
    
    
    