from nuscenes.nuscenes import NuScenes
# import open3d as o3d  
import numpy as np  
import os
import shutil  
  
sweeps_num =10 # 十帧拼接
# breakpoint()
nusc = NuScenes(version='v1.0-trainval', dataroot = 'data/nuscenes', verbose=False)
my_sample = nusc.get('sample', '0acd18a285544d33a6bbb11fe37af5bd')
lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
# breakpoint()
shutil.copy(('data/nuscenes/'+lidar_data['filename']).replace('nuscenes/samples/','raw_lidar_for_vis/'), 'raw_data_vis/rainy_night/')  #.replace('nuscenes/samples/','raw_lidar_for_vis/')
shutil.copy(('data/nuscenes/'+lidar_data['filename']).replace('samples/LIDAR_TOP', 'radar_bev_filter'), 'raw_data_vis/rainy_night/radar')


# lidar_data['filename']
# breakpoint()
cam_view = ['LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
for view in cam_view:
    image_data = nusc.get('sample_data', my_sample['data'][view])
    image_path = image_data['filename']
    shutil.copy('data/nuscenes/'+image_path, 'raw_data_vis/rainy_night/')  


