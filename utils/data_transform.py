import numpy as np
import os
import torch
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn
import cv2


from utils.data_augumentation import Enhance, ToAbsoluteCoords, RandomMirror, ToPercentCoords, Resize, Normalize_Tensor

class DataTransform():
    
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
            	Enhance(alpha = [0.75, 1.25], beta = 0.0),
                ToAbsoluteCoords(),  # アノテーションデータの規格化を戻す
                RandomMirror(),  # 画像を反転させる
                ToPercentCoords(),  # アノテーションデータを0-1に規格化
                Resize(input_size),  # 画像サイズをinput_size×input_sizeに変形
                Normalize_Tensor(color_mean, color_std),
                
            ]),
            'val': Compose([
                Resize(input_size), # 画像サイズをinput_size×input_sizeに変形
                Normalize_Tensor(color_mean, color_std)
                
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, boxes, labels)