import os
import json
import random
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class ResizeByLongSide:
    def __init__(self, target_long, interpolation=transforms.InterpolationMode.BILINEAR):
        self.target_long = target_long
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        long_side = max(w, h)
        scale = self.target_long / long_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return transforms.functional.resize(img, (new_h, new_w), interpolation=self.interpolation, antialias=True)

class CustomTrainDataset(Dataset):
    def __init__(
        self,
        json_index_path,           # 必需：主索引文件路径
        followbench_root,          # 必需：原始数据根目录 (e.g. .../FollowBench/train)
        warped_video_root,         # 必需：Warp 数据根目录 (e.g. .../WarpedVideo/train)
        height=512,
        width=512,
        resize_long=1505,
        sample_n_frames=49,
        stride=2,
        is_one2three=True,
        training_len=-1,
        **kwargs # 吸收多余参数
    ):  
        self.stride = stride
        self.training_len = training_len
        self.is_one2three = is_one2three
        
        self.followbench_root = followbench_root
        self.warped_video_root = warped_video_root
        
        self.height = height
        self.width = width
        self.resize_long = resize_long
        self.sample_n_frames = sample_n_frames
        
        # [核心修复] 加载 JSON 并转换为列表
        print(f"[Dataset] Loading index: {json_index_path}")
        with open(json_index_path, 'r') as f:
            data_raw = json.load(f)
            
        # 将 Dict 转换为 List，并排序以确保多卡训练时顺序一致
        if isinstance(data_raw, dict):
            # 按照 Case ID (Key) 排序
            self.data_list = [data_raw[k] for k in sorted(data_raw.keys())]
        elif isinstance(data_raw, list):
            self.data_list = data_raw
        else:
            raise ValueError(f"Unsupported JSON format: {type(data_raw)}")
        
        print(f"[Dataset] Found {len(self.data_list)} samples.")

        # 定义 Transforms
        self.video_transforms = transforms.Compose([
            ResizeByLongSide(self.resize_long), 
            transforms.CenterCrop([self.height, self.width]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if self.is_one2three:
            self.image_transforms = transforms.Compose([   
                transforms.Resize([self.height, self.width]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        # 如果指定了 training_len (用于 epoch 长度控制)，则返回该长度
        if self.training_len != -1:
            return self.training_len
        return len(self.data_list)

    def __getitem__(self, index):
        # 循环取数据，防止 index 越界
        item = self.data_list[index % len(self.data_list)]
        
        # [核心修复] 1. 使用您 JSON 中的正确键名解析路径
        ego_rel = item.get('ego video path')
        exo_rel = item.get('exo video path')
        ref_rel = item.get('reference image path')
        
        # 拼接完整路径
        ego_path = os.path.join(self.followbench_root, ego_rel)
        exo_path = os.path.join(self.followbench_root, exo_rel)
        
        ref_path = None
        if self.is_one2three and ref_rel:
            ref_path = os.path.join(self.followbench_root, ref_rel)

        # 3. 读取视频 (带异常捕获和重试)
        try:
            ego_vr = VideoReader(ego_path)
            exo_vr = VideoReader(exo_path)
        except Exception as e:
            print(f"[Dataset Error] Failed to load {item}: {e}")
            # 随机换一个样本重试
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        # 4. 采样索引 (确保时间对齐)
        len_ego = len(ego_vr)
        len_exo = len(exo_vr)
        
        min_len = min(len_ego, len_exo)
        
        # 确保有足够的帧
        if min_len < self.sample_n_frames:
             # 如果视频太短，重试其他样本
             return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        # 随机起始点
        max_start = max(min_len - (self.sample_n_frames - 1) * self.stride, 0)
        start_idx = np.random.randint(0, max_start + 1)
        indices = start_idx + np.arange(self.sample_n_frames) * self.stride
        indices = np.clip(indices, 0, min_len - 1)

        # 5. 获取 Tensor
        def load_frames(vr, indices):
            frames = vr.get_batch(indices).asnumpy()
            return torch.stack([self.video_transforms(Image.fromarray(f)) for f in frames])

        ego_pixel = load_frames(ego_vr, indices).permute(1, 0, 2, 3)    # [C, F, H, W]
        exo_pixel = load_frames(exo_vr, indices).permute(1, 0, 2, 3)    # [C, F, H, W]

        ref_pixel = []
        if self.is_one2three and ref_path:
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                ref_pixel = self.image_transforms(ref_img) # [C, H, W]
            except Exception:
                ref_pixel = torch.zeros((3, self.height, self.width))
        elif self.is_one2three:
             ref_pixel = torch.zeros((3, self.height, self.width))

        # Prompt 处理 (如果没有 prompt 字段则使用默认)
        prompt = item.get('prompt', 'Transform it into the third-person perspective.')

        return {
            'ref_pixel_values': ref_pixel,       # [C, H, W]
            'first_pixel_values': ego_pixel,     # [C, F, H, W]
            'third_pixel_values': exo_pixel,     # [C, F, H, W]
            'prompts': prompt,
        }
        
class CustomTestDataset(Dataset):
    def __init__(
        self,
        json_path,                 
        video_root,                
        height=512,
        width=512,
        resize_long=1280,
        sample_n_frames=49,
        stride=2,
        is_one2three=True,
        training_len=-1,
        **kwargs
    ):  
        self.json_path = json_path
        self.video_root = video_root
        self.height = height
        self.width = width
        self.resize_long = resize_long
        self.sample_n_frames = sample_n_frames
        self.stride = stride
        self.is_one2three = is_one2three
        self.training_len = training_len

        print(f"[CustomTestDataset] Loading index from: {json_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Index file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            self.samples = [data[k] for k in sorted(data.keys())]
        else:
            self.samples = data
            
        print(f"[CustomTestDataset] Loaded {len(self.samples)} samples.")

        self.video_transforms = transforms.Compose([
            ResizeByLongSide(self.resize_long), 
            transforms.CenterCrop([self.height, self.width]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if self.is_one2three:
            self.image_transforms = transforms.Compose([
                transforms.Resize([self.height, self.width]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return self.training_len if self.training_len > 0 else len(self.samples)

    def _load_frames(self, vr, indices):
        frames = vr.get_batch(indices).asnumpy()
        return torch.stack([self.video_transforms(Image.fromarray(f)) for f in frames])

    def __getitem__(self, index):
        sample_idx = index % len(self.samples)
        item = self.samples[sample_idx]
        
        ego_rel = item.get('ego video path')
        exo_rel = item.get('exo video path')
        
        # 确定 Input Video 路径
        if self.is_one2three:
            input_rel = ego_rel
        else:
            input_rel = exo_rel
            
        if not input_rel:
             raise ValueError(f"Missing input video path in sample: {item}")
             
        input_path = os.path.join(self.video_root, input_rel)
        ref_rel = item.get('reference image path')
        ref_path = os.path.join(self.video_root, ref_rel) if (self.is_one2three and ref_rel) else None

        if not os.path.exists(input_path): raise FileNotFoundError(f"Input video not found: {input_path}")
        vr_input = VideoReader(input_path, ctx=cpu(0))

        # 确定性采样：从第0帧开始
        start_idx = 0
        indices = [start_idx + i * self.stride for i in range(self.sample_n_frames)]
        
        input_pixels = self._load_frames(vr_input, indices).permute(1, 0, 2, 3) # [C, F, H, W]
        
        data_dict = {
            'input_pixel_values': input_pixels,
            'prompts': 'Transform it into the third-person perspective.' if self.is_one2three else 'Transform it into the first-person perspective.',
            'path': item.get('case id')
        }
        
        if self.is_one2three:
            if not ref_path or not os.path.exists(ref_path):
                raise FileNotFoundError(f"Ref image not found: {ref_path}")
            ref_img = Image.open(ref_path).convert("RGB")
            data_dict['ref_pixel_values'] = self.image_transforms(ref_img)
        else:
            data_dict['ref_pixel_values'] = []

        return data_dict
