import os
import pickle
import numpy as np
data_root = 'data/nuscenes'
sweeps_num = 10 
data_path = os.path.join(data_root, 'nuscenes_occ_infos_val.pkl')
with open(data_path, 'rb') as f:
    data = pickle.load(f)
nusc_infos = data['infos']
for id, info in enumerate(nusc_infos):
    points_path = info['lidar_path']        
    points = np.fromfile(points_path, dtype=np.float32, count=-1).reshape([-1, 5])
    if sweeps_num > 0:
        sweep_points_list = [points]
        ts = info['timestamp']
        breakpoint()
        if len(info['sweeps']) <= sweeps_num:
            choices = np.arange(len(info['sweeps']))
        else:
            choices = np.random.choice(len(info['sweeps']), sweeps_num, replace=False)
        for idx in choices:
            sweep = info['sweeps'][idx]
            points_sweep = np.fromfile(sweep['data_path'], dtype=np.float32, count=-1).reshape([-1, 5])
            sweep_ts = sweep['timestamp'] / 1e6
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)
        points = np.concatenate(sweep_points_list, axis=0)
        processed_points = points[:, :3]  
        processed_points.tofile(points_path.replace('nuscenes/samples/','raw_lidar_for_vis/'))  
        # breakpoint()
        print(id)
