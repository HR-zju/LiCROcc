from nuscenes.nuscenes import NuScenes
# import open3d as o3d  
import numpy as np  
import os
import shutil  
import pickle
import mmcv
from PIL import Image, ImageDraw, ImageFont  
# from PIL import ImageFont  
import matplotlib
font_path = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')  
chosen_font = [font for font in font_path if "LiberationSans" in font][0]  
font = ImageFont.truetype(chosen_font, 50)  
# sweeps_num =10 # 十帧拼接
# # breakpoint()
# nusc = NuScenes(version='v1.0-trainval', dataroot = 'data/nuscenes', verbose=False)
# my_sample = nusc.get('sample', '0acd18a285544d33a6bbb11fe37af5bd')
# lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
# # breakpoint()
# shutil.copy(('data/nuscenes/'+lidar_data['filename']).replace('nuscenes/samples/','raw_lidar_for_vis/'), 'raw_data_vis/rainy_night/')  #.replace('nuscenes/samples/','raw_lidar_for_vis/')
# shutil.copy(('data/nuscenes/'+lidar_data['filename']).replace('samples/LIDAR_TOP', 'radar_bev_filter'), 'raw_data_vis/rainy_night/radar')


# # lidar_data['filename']
# # breakpoint()
# cam_view = ['LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
# for view in cam_view:
#     image_data = nusc.get('sample_data', my_sample['data'][view])
#     image_path = image_data['filename']
#     shutil.copy('data/nuscenes/'+image_path, 'raw_data_vis/rainy_night/')  


# 

data_root = 'data/nuscenes'
show_dir = 'data/nuscenes/raw_data_for_vis'

data_path = os.path.join(data_root, 'nuscenes_occ_infos_val.pkl')
with open(data_path, 'rb') as f:
    data = pickle.load(f)
nusc_infos = data['infos']
for id, info in enumerate(nusc_infos):
    # breakpoint()
    print(id)
    points_path = info['lidar_path'].replace("samples", 'raw_lidar_for_vis')  
    radar_path = info['lidar_path'].replace('samples/LIDAR_TOP', 'radar_bev_filter')  
    scene_name = 'scene_{}'.format(info['scene_token'])
    sample_token = info['token']
    save_path = os.path.join(show_dir, scene_name, sample_token)
    mmcv.mkdir_or_exist(save_path)
    # lidar
    # shutil.copy(points_path, os.path.join(save_path, 'lidar.bin'))  
    # shutil.copy(radar_path, os.path.join(save_path, 'radar.bin'))  
    # cam
    cam_view = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
    width = 2400
    height = 900
    big_image = Image.new('RGB', (width, height))  
    for i,view in enumerate(cam_view):
        image_data = info['cams'][view]/mnt/workspace/code/ssc-rs-open-mmlab-master/visualization
        image_path = image_data['data_path']
        # shutil.copy('data/nuscenes/'+image_path, 'raw_data_vis/rainy_night/')
        # 打开图片  
        # 创建一个新的大图  
        
        
        # 计算每个小图的宽度和高度  
        small_width = big_image.width // 3  
        small_height = big_image.height // 2 
        small_image = Image.open(image_path)  
        
        # 计算小图在大图中的位置  
        x = (i % 3) * small_width  
        y = (i // 3) * small_height  
        
        # 将小图粘贴到大图中  
        big_image.paste(small_image.resize((small_width, small_height)), (x, y)) 
        draw = ImageDraw.Draw(big_image)  
        draw.text((x + 10, y + 10), view, fill="yellow", font = font)   
  
# 保存大图  
    big_image.save(os.path.join(save_path, 'cam.jpg'))    

    # breakpoint()


 
    