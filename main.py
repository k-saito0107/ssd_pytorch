# パッケージのimport
import os
import os.path as osp
import random
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import glob

from utils.ssd_model import SSD
from utils.make_dataset import MakeDataset
from utils.data_transform import DataTransform, od_collate_fn
from utils.multiboxloss import MultiBoxLoss
import json


# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model_path = "/kw_resources/distortion_detection/weights/ssd_resnet.pth"
    #model_path = '../weights/ssd_resnet_'+str(num_epochs)+'.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)

    # ネットワークをGPUへ
    
    net.to(device)
    #net = nn.DataParallel(net)

    # ネットワークがある程度固定であれば、高速化させる
    #torch.backends.cudnn.benchmark = True

    # イテレーションカウンタをセット
    iteration = 1
    epoch_train_loss = 0.0  # epochの損失和
    epoch_val_loss = 0.0  # epochの損失和
    logs = []

    # epochのループ
    for epoch in range(num_epochs+1):
        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和
        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()
        #epoch += 307

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                print('train')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()   # モデルを検証モードに
                    print('-------------')
                    print('val')
                else:
                    # 検証は10回に1回だけ行う
                    continue

            # データローダーからminibatchずつ取り出すループ
            for images, targets in dataloaders_dict[phase]:

                # GPUが使えるならGPUにデータを送る
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]  # リストの各要素のテンソルをGPUへ

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    # 順伝搬（forward）計算
                    outputs = net(images)
                    
                    

                    # 損失の計算
                    loss_l, loss_c = criterion(outputs, targets)
                    #if loss_l == float('inf') or loss_c == float('inf'):
                    #    continue
                    loss = loss_l + loss_c
                    

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()  # 勾配の計算

                        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
                        nn.utils.clip_grad_value_(
                            net.parameters(), clip_value=2.0)

                        optimizer.step()  # パラメータ更新
                        
                        epoch_train_loss += loss.item()
                        
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss += loss.item()

        
        epoch_train_loss = epoch_train_loss/iteration
        epoch_val_loss = epoch_val_loss/iteration
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch, epoch_train_loss, epoch_val_loss))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        #df.to_csv("log_output.csv")
        df.to_csv("/kw_resources/distortion_detection/log_output.csv")

        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        # ネットワークを保存する
        if ((epoch+1) % 10 == 0):
            #torch.save(net.state_dict(), 'weights/ssd300_' +str(epoch+1) + '.pth')
            torch.save(net.state_dict(), model_path)
        
        '''
        if ((epoch +1) % 50 == 0):
            #torch.save(net.state_dict(), '/kw_resources/moji_classification/weights/ssd_resnet_ver1_'+str(epoch + 1)+'.pth')
            torch.save(net.state_dict(), './weights/ssd_resnet18_'+str(epoch + 1)+'.pth')
        '''
        


def main():
    #root_path = '../data'
    root_path = '/kw_resources/distortion_detection/data'
    img_path = glob.glob(root_path+'/img/*.png')
    anno_path = glob.glob(root_path + '/anno/*.json')
    #get-train-img-path
    theta = int(0.8 * len(img_path))
    random.shuffle(img_path)
    train_img_path = img_path[:theta]
    val_img_path = img_path[theta:]

    #class
    classes = ['distortion']
    print(len(img_path))


    anno_dic = {}
    for json_path in anno_path:
        json_open = open(json_path, 'r', encoding='utf-8')
        json_load = json.load(json_open)
        img_name = json_load['imagePath']
        width = json_load['imageWidth']
        height = json_load['imageHeight']
        ret = []
        for region in json_load['shapes']:
            bndbox = []
            tags = region['label']
            box = region['points']
            xmin = box[0][0]/width
            ymin = box[0][1]/height
            xmax = box[1][0]/width
            ymax = box[1][1]/height
            label = classes.index(tags)
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)
            bndbox.append(label)
            ret += [bndbox]
        anno_dic[img_name] = np.array(ret)
    
    


    
    
    color_mean = (0.5, 0.5, 0.5)
    color_std = (0.5, 0.5, 0.5)
    
    input_size = 512
    train_dataset = MakeDataset(train_img_path, anno_dic, phase="train", transform=DataTransform(
                                input_size, color_mean, color_std))
    
    val_dataset = MakeDataset(val_img_path, anno_dic, phase="val", transform=DataTransform(
                                input_size, color_mean, color_std))
    
    #make dataloader
    batch_size = 12

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    
    # SSD300の設定
    
    ssd_cfg = {
        'num_classes': 2,  # 背景クラスを含めた合計クラス数
        'input_size': input_size,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        'feature_maps': [128, 32, 15, 8, 6, 4],  # 各sourceの画像サイズ
        'steps': [12, 24, 48, 96, 160, 512],  # DBOXの大きさを決める
        'min_sizes': [52, 104, 208, 288, 382, 480],  # DBOXの大きさを決める
        'max_sizes': [104, 208, 288, 382, 480, 540],  # DBOXの大きさを決める
        'aspect_rations': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'in_ch' : 3,
        'out_ch' : 64
    }
    
    # SSDネットワークモデル
    net = SSD(phase="train", cfg=ssd_cfg)
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



    #setting-loss-function
    
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
    optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adam(net.parameters(), lr = 0.00001)

    #trainning
    num_epoch = 200
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epoch)





if __name__ == '__main__':
    main()
    print('finish')
