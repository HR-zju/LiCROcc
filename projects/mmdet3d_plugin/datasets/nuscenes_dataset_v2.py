# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
from os import path as osp
import pickle
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
import numba as nb
from collections import defaultdict, OrderedDict
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask

def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask

def augmentation_random_flip(data, flip_type, is_scan=False):
    if flip_type==1:
        if is_scan:
            data[:, 0] = -data[:, 0]
        else:
            data = np.flip(data, axis=0).copy()
    elif flip_type==2:
        if is_scan:
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(data, axis=1).copy()
    elif flip_type==3:
        if is_scan:
            data[:, 0] = -data[:, 0]
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(np.flip(data, axis=0), axis=1).copy()
    return data

@nb.jit('u1[:,:,:](u1[:,:,:],i4[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


@DATASETS.register_module()
class nuScenesDatasetV2(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        occ_root,
        lims,
        sizes,
        frames = [],
        sweeps_num = -1,
        temporal = [],
        augmentation=False,
        shuffle_index=False
    ):
        super().__init__()
        
        self.data_root = data_root
        self.occ_root = occ_root
        self.sweeps_num = sweeps_num

        self.frames = frames

        if split == 'train':
            data_path = os.path.join(data_root, 'nuscenes_occ_infos_train.pkl')
        elif split == 'val':
            data_path = os.path.join(data_root, 'nuscenes_occ_infos_val.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']

        self.lims = lims
        self.sizes = sizes
        self.augmentation = augmentation
        self.shuffle_index = shuffle_index

        self.split = split 
        self.class_names =  ['noise',
               'barrier', 'bicycle', 'bus', 'car', 'construction',
               'motorcycle', 'pedestrian', 'trafficcone', 'trailer',
               'truck', 'driveable_surface', 'other', 'sidewalk', 'terrain',
               'mannade', 'vegetation']
        
        self.test_mode = test_mode
        self.set_group_flag()

    def __getitem__(self, index):
        if not self.test_mode:
            flip_type = np.random.randint(0, 4) if self.augmentation else 0
            data_queue = self.prepare_train(index, flip_type)

            points = [torch.from_numpy(each['points'][:, :4]).float() for each in data_queue.values()]

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
            example = self.prepare_example(index, flip_type)
            points = torch.from_numpy(example['points'][:, :4]).float()
            example['points'] = DC(points, cpu_only=False, stack=False)
            example['img_metas'] = DC(example['img_metas'], cpu_only=True)

            return example

    def __len__(self):
        return len(self.nusc_infos)

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
        example = self.prepare_example(index, flip_type)
        data_queue[0] = example
        cur_scene_token = self.nusc_infos[index]['scene_token']

        for frame_idx in self.frames:
            chosen_idx = index + frame_idx
            if frame_idx ==0 or chosen_idx <0 or chosen_idx >= len(self.nusc_infos) or self.nusc_infos[chosen_idx]['scene_token'] != cur_scene_token:
                chosen_idx = index

            example = self.prepare_example(chosen_idx, flip_type)
            data_queue[frame_idx] = example

        data_queue = OrderedDict(sorted(data_queue.items()))

        return data_queue

    def prepare_example(self, index, flip_type):
        info = self.nusc_infos[index]
        points_path = info['lidar_path']        
        points = np.fromfile(points_path, dtype=np.float32, count=-1).reshape([-1, 5])

        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = info['lidar2ego_translation']

        egocurr2global = np.eye(4, dtype=np.float32)
        egocurr2global[:3,:3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        egocurr2global[:3,3] = info['ego2global_translation']
        lidarcurr2global = egocurr2global @ lidar2ego
        
        # surround
        if self.sweeps_num > 0:
            sweep_points_list = [points]
            ts = info['timestamp']
            if len(info['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(info['sweeps']))
            else:
                choices = np.random.choice(len(info['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = info['sweeps'][idx]
                points_sweep = np.fromfile(sweep['data_path'], dtype=np.float32, count=-1).reshape([-1, 5])
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                sweep_points_list.append(points_sweep)
            points = np.concatenate(sweep_points_list, axis=0)

        # history frame
        history_points = [points]
        cur_scene_token = info['scene_token']
        for frame_idx in self.frames:
            chosen_idx = index + frame_idx
            if frame_idx ==0 or chosen_idx <0 or chosen_idx >= len(self.nusc_infos):
                continue
            ref_info = self.nusc_infos[chosen_idx]
            
            if ref_info['scene_token'] == cur_scene_token:     
                ref_points = np.fromfile(ref_info['lidar_path'], dtype=np.float32, count=-1).reshape([-1, 5])
                ref_sweep_points_list = [ref_points]
                ts = ref_info['timestamp']
                if len(ref_info['sweeps']) <= self.sweeps_num:
                    choices = np.arange(len(ref_info['sweeps']))
                else:
                    choices = np.random.choice(len(ref_info['sweeps']), self.sweeps_num, replace=False)
                for idx in choices:
                    sweep = ref_info['sweeps'][idx]
                    points_sweep = np.fromfile(sweep['data_path'], dtype=np.float32, count=-1).reshape([-1, 5])
                    sweep_ts = sweep['timestamp'] / 1e6
                    points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                    points_sweep[:, :3] += sweep['sensor2lidar_translation']
                    points_sweep[:, 4] = ts - sweep_ts
                    ref_sweep_points_list.append(points_sweep)
                ref_points = np.concatenate(ref_sweep_points_list, axis=0)

                ego2global = np.eye(4, dtype=np.float32)
                ego2global[:3,:3] = Quaternion(ref_info['ego2global_rotation']).rotation_matrix
                ego2global[:3,3] = ref_info['ego2global_translation']
                lidar2global = ego2global @ lidar2ego
                ref2curr = np.matmul(inv(lidarcurr2global), lidar2global)
                ref_points[:, :3] = np.matmul(ref2curr, np.concatenate([ref_points[:, :3], np.ones_like(ref_points[:, :1])], axis=-1).T).T[:, :3]
                history_points.append(ref_points)
        points = np.concatenate(history_points, axis=0)

        if self.lims:
            filter_mask = get_mask(points, self.lims)
            points = points[filter_mask]
        
        max_bound = np.array([self.lims[0][0], self.lims[1][0], self.lims[2][0]])
        min_bound = np.array([self.lims[0][1], self.lims[1][1], self.lims[2][1]])
        intervals = (max_bound - min_bound) / np.array(self.sizes)
        xyz_grid_ind = (np.floor((points[:, :3] - min_bound) / intervals)).astype(np.int32)
        xyz_occ_label = np.ones((xyz_grid_ind.shape[0], 1), dtype=np.uint8)
        label_voxel_pair = np.concatenate([xyz_grid_ind, xyz_occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((xyz_grid_ind[:, 0], xyz_grid_ind[:, 1], xyz_grid_ind[:, 2])), :].astype(np.int32)
        occupancy = np.zeros(self.sizes, dtype=np.uint8)
        occupancy = nb_process_label(occupancy, label_voxel_pair)

        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(info['scene_token'], info['lidar_token'])
        #  [z y x cls]
        pcd = np.load(os.path.join(self.occ_root, rel_path))
        occ_label = pcd[..., -1:]
        occ_label[occ_label==0] = 255
        occ_xyz_grid = pcd[..., [2,1,0]]
        
        label_voxel_pair = np.concatenate([occ_xyz_grid, occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid[:, 0], occ_xyz_grid[:, 1], occ_xyz_grid[:, 2])), :].astype(np.int32)
        target = np.zeros([self.sizes[0], self.sizes[1], self.sizes[2]], dtype=np.uint8)
        target = nb_process_label(target, label_voxel_pair)

        occ_xyz_grid_1_2 = occ_xyz_grid//2
        label_voxel_pair = np.concatenate([occ_xyz_grid_1_2, occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid_1_2[:, 0], occ_xyz_grid_1_2[:, 1], occ_xyz_grid_1_2[:, 2])), :].astype(np.int32)
        target_1_2 = np.zeros([self.sizes[0]//2, self.sizes[1]//2, self.sizes[2]//2], dtype=np.uint8)
        target_1_2 = nb_process_label(target_1_2, label_voxel_pair)

        occ_xyz_grid_1_4 = occ_xyz_grid//4
        label_voxel_pair = np.concatenate([occ_xyz_grid_1_4, occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid_1_4[:, 0], occ_xyz_grid_1_4[:, 1], occ_xyz_grid_1_4[:, 2])), :].astype(np.int32)
        target_1_4 = np.zeros([self.sizes[0]//4, self.sizes[1]//4, self.sizes[2]//4], dtype=np.uint8)
        target_1_4 = nb_process_label(target_1_4, label_voxel_pair)

        occ_xyz_grid_1_8 = occ_xyz_grid//8
        label_voxel_pair = np.concatenate([occ_xyz_grid_1_8, occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid_1_8[:, 0], occ_xyz_grid_1_8[:, 1], occ_xyz_grid_1_8[:, 2])), :].astype(np.int32)
        target_1_8 = np.zeros([self.sizes[0]//8, self.sizes[1]//8, self.sizes[2]//8], dtype=np.uint8)
        target_1_8 = nb_process_label(target_1_8, label_voxel_pair)

        if self.augmentation:
            points = augmentation_random_flip(points, flip_type, is_scan=True)
            occupancy = augmentation_random_flip(occupancy, flip_type)
            target = augmentation_random_flip(target, flip_type)
            target_1_2 = augmentation_random_flip(target_1_2, flip_type)
            target_1_4 = augmentation_random_flip(target_1_4, flip_type)
            target_1_8 = augmentation_random_flip(target_1_8, flip_type)
        
        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
            points = points[pt_idx]

        post_trans = np.eye(4, dtype=np.float32)
        if self.augmentation:
            if flip_type == 1:
                post_trans[0, 0] = -post_trans[0, 0]
            if flip_type == 2:
                post_trans[1, 1] = -post_trans[1, 1]
            if flip_type == 3:
                post_trans[0, 0] = -post_trans[0, 0]
                post_trans[1, 1] = -post_trans[1, 1]

        meta_dict = dict(
            points_paths = [points_path],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            occupancy=occupancy.astype(np.float32),
            target_1_2=target_1_2,
            target_1_4=target_1_4,
            target_1_8=target_1_8,
            lidar2global=np.matmul(lidarcurr2global, post_trans),
            sequence_id=info['scene_token']
        )

        data_info = dict(
            img_metas = meta_dict,
            points = points,
            target = target
        )

        return data_info

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

        eval_results = {}
        if 'ssc_scores' in results:
            ssc_scores = results['ssc_scores']
            
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
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
        
        for key, val in res_dic.items():
            eval_results['nuScenes_{}'.format(key)] = round(val * 100, 2)
        
        # add two main metrics to serve as the sort metric
        eval_results['nuScenes_combined_IoU'] = eval_results['nuScenes_SC_IoU'] + eval_results['nuScenes_SSC_mIoU']
        
        if logger is not None:
            logger.info('NuScenes SSC Evaluation')
            logger.info(eval_results)

        return eval_results

