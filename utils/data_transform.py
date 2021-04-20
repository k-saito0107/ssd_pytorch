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


def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なります。
    画像内の物体数が2個であれば(2, 5)というサイズですが、3個であれば（3, 5）など変化します。
    この変化に対応したDataLoaderを作成するために、
    カスタイマイズした、collate_fnを作成します。
    collate_fnは、PyTorchでリストからmini-batchを作成する関数です。
    ミニバッチ分の画像が並んでいるリスト変数batchに、
    ミニバッチ番号を指定する次元を先頭に1つ追加して、リストの形を変形します。
    """

    targets = []
    imgs = []
    batch = filter(None, batch)
    
    for sample in batch:
        if not(sample[0] is None):
            imgs.append(sample[0])  # sample[0] は画像imgです
            targets.append(torch.FloatTensor(sample[1]))  # sample[1] はアノテーションgtです

    # imgsはミニバッチサイズのリストになっています
    # リストの要素はtorch.Size([3, 300, 300])です。
    # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換します
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリストです。
    # リストのサイズはミニバッチサイズです。
    # リストtargetsの要素は [n, 5] となっています。
    # nは画像ごとに異なり、画像内にある物体の数となります。
    # 5は [xmin, ymin, xmax, ymax, class_index] です

    return imgs, targets