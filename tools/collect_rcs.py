import os
import pickle
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import seaborn as sns
import math

data_root = 'data/nuscenes'
occ_root = './data/nuScenes-Occupancy'
sweeps_num = 10 
data_path = os.path.join(data_root, 'nuscenes_occ_infos_val.pkl')
sizes = [512,512,40]
lims = [[-51.2, 51.2], [-51.2, 51.2], [-5, 3.0]]







def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask

def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask


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

def gen_data():
    file_names = []
    rcs_dit_org = []
    for i in range(17):
        file_names.append(f'{str(i)}.txt')
        rcs_dit_org.append([])

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    nusc_infos = data['infos']
    for id, info in enumerate(nusc_infos):
    
    
        print(id)
        rcs_dit = rcs_dit_org
        points_path = info['lidar_path'].replace('samples/LIDAR_TOP', 'radar_bev_filter')      
        points = np.fromfile(points_path, dtype=np.float32, count=-1).reshape([-1, 7])
        if lims:
            filter_mask = get_mask(points, lims)
            points = points[filter_mask]
        # get lidar indices
        min_bound = np.array([lims[0][0], lims[1][0], lims[2][0]])
        max_bound = np.array([lims[0][1], lims[1][1], lims[2][1]])
        intervals = (max_bound - min_bound) / np.array(sizes)
        xyz_grid_ind = (np.floor((points[:, :3] - min_bound) / intervals)).astype(np.int32)

        # load gt
        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(info['scene_token'], info['lidar_token'])
        pcd = np.load(os.path.join(occ_root, rel_path))
        occ_label = pcd[..., -1:]
        occ_label[occ_label==0] = 255
        occ_xyz_grid = pcd[..., [2,1,0]]
        label_voxel_pair = np.concatenate([occ_xyz_grid, occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid[:, 0], occ_xyz_grid[:, 1], occ_xyz_grid[:, 2])), :].astype(np.int32)
        target = np.zeros([sizes[0], sizes[1], sizes[2]], dtype=np.uint8)
        target = nb_process_label(target, label_voxel_pair)
        for i in range(len(points)):
            pc_class = target[xyz_grid_ind[i][0]][xyz_grid_ind[i][1]][xyz_grid_ind[i][2]]
            # breakpoint()
            if pc_class == 0 or pc_class == 255:
                continue

            # rcs_dit[pc_class].append(points[i][4])
            rcs_dit[pc_class].append(math.sqrt(points[i][5]**2+points[i][6]**2))

       

        final_arrays = [np.array(l) for l in rcs_dit]

        for single_array, file_name in zip(final_arrays, file_names):
            np.save('collect_data/'+file_name, single_array)

def draw_data():
    file_names = []
    rcs_dit_org = []
    for i in range(17):
        file_names.append(f'{str(i)}.txt')
        file_name = f'collect_data/{str(i)}.txt.npy'
        my_array = np.load(file_name)
        rcs_dit_org.append(my_array)
    plt.figure(figsize=(10, 6))

    # 为列表中的每个Numpy数组画箱形图
    sns.boxplot(data=rcs_dit_org)

    # 显示图表
    # plt.show()
    plt.savefig('collect_data/boxplot.png')


    # breakpoint()


if __name__ == "__main__":
    
#    draw_data()
    gen_data()