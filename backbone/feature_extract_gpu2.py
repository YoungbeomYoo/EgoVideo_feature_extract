import torch

from model.setup_model import *
import argparse
import pandas as pd
model, _ = build_model(ckpt_path='ckpt_4frames.pth',num_frames=4)
model = model.eval().to('cuda').to(torch.float16)

from tqdm import tqdm
import math
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
from torch import nn
import torchvision.transforms._transforms_video as transforms_video
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop
import torchvision

def get_frame_ids(start_frame, end_frame, num_segments=4):
    # 각 세그먼트의 길이를 구함
    seg_size = (end_frame - start_frame) / num_segments
    
    # 각 세그먼트의 프레임을 구함
    frame_ids = [start_frame + seg_size * (i + 1) - 1 for i in range(num_segments)]
    
    return frame_ids

def get_frame_ids_r(start_frame, end_frame, num_segments=4):
    # 각 세그먼트의 길이를 구함
    seg_size = (end_frame - start_frame) / num_segments
    
    # 각 세그먼트의 프레임을 구함
    frame_ids = [end_frame - seg_size * (3 - i) - 1 for i in range(num_segments)]
    
    return frame_ids

def EK100_list(folder_path):
    video_files = []

    # 폴더를 순회하며 .MP4 파일을 찾음
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".MP4"):
                video_files.append(os.path.join(root, file))
    
    # 파일을 정렬
    sorted_files = sorted(video_files)
    # print(len(sorted_files))

    # 505번째 파일을 찾음
    return sorted_files

def feature_processing(vr, frame_ids, val_transform, encoder):
    frames = vr.get_batch(frame_ids).asnumpy()                
    return_frame = torch.from_numpy(frames.astype(np.float32))
    transformed_frame = val_transform(return_frame)
    batched_tensor = transformed_frame.unsqueeze(0)
    image_features = encoder(batched_tensor.to('cuda'))
    image_feat = F.normalize(image_features, dim=-1)
    return image_feat

# 사용할 폴더 경로를 지정하세요
folder_path = "/mnt/NAS3/CIPLAB/dataset/EPICKITCHEN/videos/EK100_original_videos"
# EK100_path = EK100_list(folder_path)[:44]
# EK100_path = EK100_list(folder_path)[44:88]
# EK100_path = EK100_list(folder_path)[98:99]
# EK100_path = EK100_list(folder_path)[132:175]
# EK100_path = EK100_list(folder_path)[175:219]
# EK100_path = EK100_list(folder_path)[219:263]
# EK100_path = EK100_list(folder_path)[263:307]
EK100_path = EK100_list(folder_path)[307:350]
# EK100_path = EK100_list(folder_path)[350:394]
# EK100_path = EK100_list(folder_path)[394:438]
# EK100_path = EK100_list(folder_path)[438:482]
# EK100_path = EK100_list(folder_path)[482:525]
# EK100_path = EK100_list(folder_path)[525:569]
# EK100_path = EK100_list(folder_path)[569:613]
# EK100_path = EK100_list(folder_path)[613:657]
# EK100_path = EK100_list(folder_path)[657:700]
error_video_list = []

#pre-define
save_path = "/mnt/NAS3/CIPLAB/dataset/EPICKITCHEN/EK100/rgb_egovideo_feature/"
rrc_params = (224,)
fps = -1
second = 0
end_second = 1
video_chunk_len = -1
num_frames = 4
clip_length = 4
clip_stride = 16
num_clips = 1
num_crops = 1
sparse_sample = False
decode_threads = 1
fused_decode_crop = True
crop_size = 224
mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]  
base_val_transform_ls = [
    Permute([3, 0, 1, 2]),
    torchvision.transforms.Resize(crop_size),
    torchvision.transforms.CenterCrop(crop_size),
    transforms_video.NormalizeVideo(mean=mean, std=std),
]
val_transform = torchvision.transforms.Compose(base_val_transform_ls)

class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)

for video_path in tqdm(EK100_path, desc="Processing videos", unit="video"):
    save_name = video_path.split('/')[-1].split('.')[0]+'.npy'
    
    if os.path.isfile(save_path + save_name):
        print("is file is true!")
        continue
    else:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1, width=rrc_params[0], height=rrc_params[0])
        fps = round(vr.get_avg_fps()) if fps == -1 else fps

        all_features = []
        second = 0
        end_second = 1

        # 비디오가 끝날 때까지 second와 end_second를 0.25씩 증가시키면서 프레임 추출
        while second < len(vr) / fps:
            end_second = min(second + 1, len(vr) / fps)  # end_second가 비디오의 끝을 넘지 않도록 설정
            frame_offset = second * fps            
            total_duration = max((end_second - second) * fps, clip_length)
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length)
            
            if second == 0:
                # -0.75 ~ 0.25
                frame_ids_1 = [0, 0, 0, frame_ids[0]]
                image_feat = feature_processing(vr, frame_ids_1, val_transform, model.encode_image)
                all_features.append(image_feat.detach().cpu())
                # -0.50 ~ 0.50
                frame_ids_2 = [0, 0, frame_ids[0], frame_ids[1]]
                image_feat = feature_processing(vr, frame_ids_2, val_transform, model.encode_image)
                all_features.append(image_feat.detach().cpu())
                # -0.25 ~ 0.75
                frame_ids_3 = [0, frame_ids[0], frame_ids[1], frame_ids[2]]

                all_features.append(image_feat.detach().cpu())
                image_feat = feature_processing(vr, frame_ids_3, val_transform, model.encode_image)
                # 0.00 ~ 1.00
                image_feat = feature_processing(vr, frame_ids, val_transform, model.encode_image)
                all_features.append(image_feat.detach().cpu())
            
            elif end_second == len(vr) / fps:
                second = end_second - 1              
                frame_offset = second * fps                
                total_duration = max((end_second - second) * fps, clip_length)
                frame_ids = get_frame_ids_r(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length)
                image_feat = feature_processing(vr, frame_ids, val_transform, model.encode_image)
                all_features.append(image_feat.detach().cpu())
                break

            else:
                image_feat = feature_processing(vr, frame_ids, val_transform, model.encode_image)
                all_features.append(image_feat.detach().cpu())

            # second와 end_second를 0.25씩 증가시킴
            second += 0.25


        # 리스트의 모든 텐서를 결합하여 하나의 텐서로 만듦
        all_features_tensor = torch.cat(all_features, dim=0)  # [num_clips, 512]

        # check num_clips is correct! expacted vs stacked
        if (len(vr) / fps) % 0.25 > 0:
            if not len(all_features_tensor) == (len(vr) / fps) // 0.25 + 1:
                print("wrong!")
                error_video_list.append(video_path)
        elif (len(vr) / fps) % 0.25 == 0:
            if not len(all_features_tensor) == (len(vr) / fps) // 0.25:
                print("wrong!")
                error_video_list.append(video_path)

        # Feature 저장 (torch의 .pt 파일로 저장)
        
        # torch.save(all_features_tensor, 'video_features.pt')

        # 또는 numpy로 저장하고 싶다면:
        np.save(save_path + save_name, all_features_tensor.numpy())
        
    print(error_video_list)
        

print(error_video_list)