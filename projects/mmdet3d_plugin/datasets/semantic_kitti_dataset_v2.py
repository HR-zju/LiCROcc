# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
from os import path as osp
from PIL import Image
import glob
import random
import copy
import mmcv
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.ssc_rs.utils.ssc_metric import SSCMetrics
from numba import njit, prange
import skimage
import skimage.io
import yaml
from collections import defaultdict, OrderedDict


def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask

def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def augmentation_random_flip(data, flip_type, is_scan=False):
    if flip_type==1:
        if is_scan:
            data[:, 0] = 51.2 - data[:, 0]
        else:
            data = np.flip(data, axis=0).copy()
    elif flip_type==2:
        if is_scan:
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(data, axis=1).copy()
    elif flip_type==3:
        if is_scan:
            data[:, 0] = 51.2 - data[:, 0]
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(np.flip(data, axis=0), axis=1).copy()
    return data


@DATASETS.register_module()
class SemanticKittiDatasetV2(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        lims,
        sizes,
        labels_tag = 'labels',
        augmentation=False,
        shuffle_index=False,
        temporal = [],
        frames = [],
        eval_range = 51.2,
        occupancy_tag = 'sequences'
    ):
        super().__init__()
        
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, labels_tag)
        self.occupancy_tag = occupancy_tag
        self.frames = frames

        self.lims = lims
        self.sizes = sizes
        self.augmentation = augmentation
        self.shuffle_index = shuffle_index

        self.eval_range = eval_range
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split 
        self.sequences = splits[split]
        self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", 
                            "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", 
                            "vegetation", "trunk", "terrain", "pole", "traffic-sign",]

        self.poses=self.load_poses()
        self.target_frames = temporal
        self.load_scans()
        
        self.test_mode = test_mode
        self.set_group_flag()
        

    def __getitem__(self, index):
        if not self.test_mode:
            flip_type = np.random.randint(0, 4) if self.augmentation else 0
            data_queue = self.prepare_train(index, flip_type)

            points = [each['points'][:, :4].float() for each in data_queue.values()]

            meta_dict = data_queue[0]['img_metas']
            lidar2global = []
            occupancy = []
            for i, each in data_queue.items():
                lidar2global.append(each['img_metas']['lidar2global'])
                occupancy.append(each['img_metas']['occupancy'])
            meta_dict['lidar2global'] = lidar2global
            meta_dict['occupancy'] = occupancy

            data_queue[0]['points'] = DC(points, cpu_only=False, stack=False)
            data_queue[0]['img_metas'] = DC(meta_dict, cpu_only=True)

            return data_queue[0]
        else:
            flip_type = 0
            example = self.get_data_info(index, flip_type)
            points = example['points'][:, :4].float()
            example['points'] = DC(points, cpu_only=False, stack=False)
            example['img_metas'] = DC(example['img_metas'], cpu_only=True)

            return example

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )

            for voxel_path in sorted(glob.glob(glob_path)):

                self.scans.append(
                    {
                        "sequence": sequence,
                        "pose": self.poses[sequence],
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path
                    }
                )

    def set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_train(self, index, flip_type):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = OrderedDict()
        example = self.get_data_info(index, flip_type)
        data_queue[0] = example
        cur_squence = self.scans[index]["sequence"]

        for frame_idx in self.frames:
            chosen_idx = index + frame_idx
            if frame_idx ==0 or chosen_idx <0 or chosen_idx >= len(self.scans) or self.scans[chosen_idx]["sequence"] != cur_squence:
                chosen_idx = index

            example = self.get_data_info(chosen_idx, flip_type)
            data_queue[frame_idx] = example

        data_queue = OrderedDict(sorted(data_queue.items()))

        return data_queue


    def get_data_info(self, index, flip_type):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        scan = self.scans[index]
        voxel_path = scan["voxel_path"]

        sequence = scan["sequence"]
        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        meta_dict = self.get_meta_info(scan, sequence, frame_id, flip_type)
        points = self.get_input_info(sequence, frame_id, flip_type)
        target = self.get_gt_info(sequence, frame_id, flip_type)

        data_info = dict(
            img_metas = meta_dict,
            points = points,
            target = target
        )

        return data_info

    def get_meta_info(self, scan, sequence, frame_id, flip_type):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id,
            flip_type (str): aug.

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        points_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "velodyne", frame_id + ".bin"
        )
        pose_list = self.poses[sequence]

        lidar2global = pose_list[int(frame_id)]

        occupancy_path = os.path.join(
            self.data_root, "dataset", self.occupancy_tag, sequence, "voxels", frame_id + ".bin"
        )
        occupancy = unpack(np.fromfile(occupancy_path, dtype=np.uint8))
        occupancy = occupancy.reshape(256, 256, 32).astype(np.float32)
        if self.augmentation:
            occupancy = augmentation_random_flip(occupancy, flip_type)

        if self.split == 'train' or self.split == 'val':
            target_1_2_path = os.path.join(self.label_root, sequence, frame_id + "_1_2.npy")
            target_1_2 = np.load(target_1_2_path)
            target_1_2 = target_1_2.reshape(128, 128, 16)
            target_1_2 = target_1_2.astype(np.float32)

            target_1_4_path = os.path.join(self.label_root, sequence, frame_id + "_1_4.npy")
            target_1_4 = np.load(target_1_4_path)
            target_1_4 = target_1_4.reshape(64, 64, 8)
            target_1_4 = target_1_4.astype(np.float32)

            target_1_8_path = os.path.join(self.label_root, sequence, frame_id + "_1_8.npy")
            target_1_8 = np.load(target_1_8_path)
            target_1_8 = target_1_8.reshape(32, 32, 4)
            target_1_8 = target_1_8.astype(np.float32)

            invalid_1_2_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "voxels", frame_id + ".invalid_1_2"
            )
            invalid_1_2 = unpack(np.fromfile(invalid_1_2_path, dtype=np.uint8))
            invalid_1_2 = invalid_1_2.reshape(128, 128, 16)
            invalid_1_2 = invalid_1_2.astype(np.float32)

            invalid_1_4_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "voxels", frame_id + ".invalid_1_4"
            )
            invalid_1_4 = unpack(np.fromfile(invalid_1_4_path, dtype=np.uint8))
            invalid_1_4 = invalid_1_4.reshape(64, 64, 8)
            invalid_1_4 = invalid_1_4.astype(np.float32)

            invalid_1_8_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "voxels", frame_id + ".invalid_1_8"
            )
            invalid_1_8 = unpack(np.fromfile(invalid_1_8_path, dtype=np.uint8))
            invalid_1_8 = invalid_1_8.reshape(32, 32, 4)
            invalid_1_8 = invalid_1_8.astype(np.float32)

            target_1_2[invalid_1_2==1] = 255
            target_1_4[invalid_1_4==1] = 255
            target_1_8[invalid_1_8==1] = 255

            if self.augmentation:
                target_1_2 = augmentation_random_flip(target_1_2, flip_type)
                target_1_4 = augmentation_random_flip(target_1_4, flip_type)
                target_1_8 = augmentation_random_flip(target_1_8, flip_type)
        else:
            target_1_2 = None
            target_1_4 = None
            target_1_8 = None
        
        post_trans = np.eye(4, dtype=np.float32)
        if self.augmentation:
            if flip_type == 1:
                post_trans[0, 0] = -post_trans[0, 0]
                post_trans[0, 3] = 51.2
            if flip_type == 2:
                post_trans[1, 1] = -post_trans[1, 1]
            if flip_type == 3:
                post_trans[0, 0] = -post_trans[0, 0]
                post_trans[0, 3] = 51.2
                post_trans[1, 1] = -post_trans[1, 1]

        meta_dict = dict(
            sequence_id = sequence,
            frame_id = frame_id,
            points_paths = [points_path],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            lidar2global = np.matmul(lidar2global, post_trans),
            occupancy=occupancy,
            target_1_2=target_1_2,
            target_1_4=target_1_4,
            target_1_8=target_1_8
        )

        return meta_dict

    def get_input_info(self, sequence, frame_id, flip_type):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Point.
        """
        seq_len = len(self.poses[sequence])
        point_list = []

        points_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "velodyne", frame_id + ".bin"
        )
        points = np.fromfile(points_path, dtype=np.float32)
        points = points.reshape((-1, 4))

        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
            points = points[pt_idx]

        if self.lims:
            filter_mask = get_mask(points, self.lims)
            points = points[filter_mask]

        if self.augmentation:
            points = augmentation_random_flip(points, flip_type, is_scan=True)
        
        point_list.append(torch.from_numpy(points).float())

        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            points_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "velodyne", target_id + ".bin"
            )
            points = np.fromfile(points_path, dtype=np.float32)
            points = points.reshape((-1, 4))

            pose_list = self.poses[sequence]
            ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
            target = pose_list[int(target_id)]
            points[:, :3] = np.matmul(np.matmul(inv(ref), target), np.concatenate([points[:, :3], np.ones_like(points[:, :1])], axis=-1).T).T[:, :3]

            if self.shuffle_index:
                pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
                points = points[pt_idx]

            if self.lims:
                filter_mask = get_mask(points, self.lims)
                points = points[filter_mask]

            if self.augmentation:
                points = augmentation_random_flip(points, flip_type, is_scan=True)

            point_list.append(torch.from_numpy(points).float())

        point_tensor = torch.cat(point_list, dim=0) #[N, 4]

        return point_tensor

    def get_gt_info(self, sequence, frame_id, flip_type):
        """Get the ground truth.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            array: target. 
        """
        if self.split == "train" or self.split == "val":
            # load full-range groundtruth
            target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
            target = np.load(target_1_path)
            invalid_1_1_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "voxels", frame_id + ".invalid"
            )
            invalid_1_1 = unpack(np.fromfile(invalid_1_1_path, dtype=np.uint8))
            invalid_1_1 = invalid_1_1.reshape(256, 256, 32)
            invalid_1_1 = invalid_1_1.astype(np.float32)
            target[invalid_1_1 == 1] = 255
            # short-range groundtruth
            if self.eval_range == 25.6:
                target[128:, :, :] = 255
                target[:, :64, :] = 255
                target[:, 192:, :] = 255

            elif self.eval_range == 12.8:
                target[64:, :, :] = 255
                target[:, :96, :] = 255
                target[:, 160:, :] = 255

            if self.augmentation:
                target = augmentation_random_flip(target, flip_type)
        else:
            target = np.ones((256,256,32))

        return target

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        if results is None:
            logger.info('Skip Evaluation')
        
        if 'ssc_scores' in results:
            # for single-GPU inference
            ssc_scores = results['ssc_scores']
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            # for multi-GPU inference
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])
            
            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])
            
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            
            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }
        
        for name, iou in zip(self.class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)
        
        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']
        
        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)
        
        return eval_results

