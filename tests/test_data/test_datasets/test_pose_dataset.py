import numpy as np  # isort: skip
import pytest  # isort: skip

from mmaction.datasets import PoseDataset  # isort: skip
from .base import BaseTestDataset  # isort: skip


class TestPoseDataset(BaseTestDataset):

    def test_pose_dataset(self):
        ann_file = self.pose_ann_file
        data_prefix = 'root'
        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            box_thre='0.5',
            data_prefix=data_prefix)
        assert len(dataset) == 100
        item = dataset[0]
        assert item['filename'].startswith(data_prefix)

        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            valid_ratio=0.2,
            box_thre='0.9',
            data_prefix=data_prefix)
        assert len(dataset) == 84
        for item in dataset:
            assert item['filename'].startswith(data_prefix)
            assert np.all(item['box_score'][item['anno_inds']] >= 0.9)
            assert item['valid@0.9'] / item['total_frames'] >= 0.2

        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            valid_ratio=0.3,
            box_thre='0.7',
            data_prefix=data_prefix)
        assert len(dataset) == 87
        for item in dataset:
            assert item['filename'].startswith(data_prefix)
            assert np.all(item['box_score'][item['anno_inds']] >= 0.7)
            assert item['valid@0.7'] / item['total_frames'] >= 0.3

        with pytest.raises(AssertionError):
            dataset = PoseDataset(
                ann_file=ann_file,
                pipeline=[],
                valid_ratio=0.2,
                box_thre='0.55',
                data_prefix=data_prefix)
