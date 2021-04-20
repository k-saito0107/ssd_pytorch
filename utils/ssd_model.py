import numpy as np
import os
import torch
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn


def conv3x3(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


def conv1x1(in_channels, out_channels, kernel_size = 1, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)


class BasickBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = None):
        super(BasickBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity_x = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity_x = self.downsample(x)
        
        out += identity_x
        return self.relu(out)


class ResidualLayer(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, block = BasickBlock):
        super(ResidualLayer, self).__init__()
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels)
            )
        self.first_block = block(in_channels, out_channels, downsample=downsample)
        self.blocks = nn.ModuleList(block(out_channels, out_channels) for _ in range(num_blocks))
    
    def forward(self, x):
        out = self.first_block(x)
        for block in self.blocks:
            out = block(out)
        
        return out


class Feature_extractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Feature_extractor, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.layer1 = ResidualLayer(2, in_channels=out_ch, out_channels=out_ch)
        self.layer2 = ResidualLayer(2, in_channels=out_ch, out_channels=out_ch*2)
        self.layer3 = ResidualLayer(2, in_channels=out_ch*2, out_channels=out_ch*4)
        self.layer4 = ResidualLayer(2, in_channels=out_ch*4, out_channels=out_ch*8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.maxpool(out)

        out = self.layer2(out)
        source1 = out
        out = self.maxpool(out)

        out = self.layer3(out)
        out = self.maxpool(out)

        source2 = self.layer4(out)

        return source1, source2


class Extras(nn.Module):
    def __init__(self, in_ch):
        super(Extras, self).__init__()
        self.conv1_1 = nn.Conv2d(in_ch, int(in_ch/4), kernel_size=(1))
        self.conv1_2 = nn.Conv2d(int(in_ch/4), int(in_ch/2), kernel_size=(3), stride=3, padding=7)

        self.conv2_1 = nn.Conv2d(int(in_ch/2), int(in_ch/8), kernel_size=(1))
        self.conv2_2 = nn.Conv2d(int(in_ch/8), int(in_ch/4), kernel_size=(3), stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(int(in_ch/4), int(in_ch/8), kernel_size=(1))
        self.conv3_2 = nn.Conv2d(int(in_ch/8), int(in_ch/4), kernel_size=(3))

        self.conv4_1 = nn.Conv2d(int(in_ch/4), int(in_ch/8), kernel_size=(1))
        self.conv4_2 = nn.Conv2d(int(in_ch/8), int(in_ch/4), kernel_size=(3))

    def forward(self, x):
        source3 = self.conv1_2(self.conv1_1(x))
        source4 = self.conv2_2(self.conv2_1(source3))
        source5 = self.conv3_2(self.conv3_1(source4))
        source6 = self.conv4_2(self.conv4_1(source5))

        return source3, source4, source5, source6

class L2Norm(nn.Module):
    def __init__(self, in_ch, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ch))
        self.scale = scale
        self.reset_parameter()
        self.eps = 1e-10
    
    def reset_parameter(self):
        init.constant_(self.weight, self.scale)
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        out =weights * x
        return out


class DBox():
    def __init__(self, cfg):
        super(DBox, self).__init__()
        self.image_size = cfg['input_size']
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg['feature_maps'])
        self.steps = cfg['steps']
        self.min_size = cfg['min_sizes']
        self.max_size = cfg['max_sizes']
        self.aspect_rations = cfg['aspect_rations']

    def make_box_list(self):
        mean = []

        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size/self.steps[k]

                #DBox normalize
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #Small DBox with aspect ratio of 1[cx, cy, width, height]
                s_k = self.min_size[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                #Big DBox with aspect ratio of 1[cx, cy, width, height]
                s_k_prime = sqrt(s_k * (self.max_size[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                #Other aspect ratio DBox
                for ar in self.aspect_rations[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        
        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max = 1, min = 0)

        return output




# オフセット情報を使い、DBoxをBBoxに変換する関数
def decode(loc, dbox_list):
    """
    オフセット情報を使い、DBoxをBBoxに変換する。

    Parameters
    ----------
    loc:  [8732,4]
        SSDモデルで推論するオフセット情報。
    dbox_list: [8732,4]
        DBoxの情報

    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBoxの情報
    """

    # DBoxは[cx, cy, width, height]で格納されている
    # locも[Δcx, Δcy, Δwidth, Δheight]で格納されている

    # オフセット情報からBBoxを求める
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
    # boxesのサイズはtorch.Size([8732, 4])となります

    # BBoxの座標情報を[cx, cy, width, height]から[xmin, ymin, xmax, ymax] に
    boxes[:, :2] -= boxes[:, 2:] / 2  # 座標(xmin,ymin)へ変換
    boxes[:, 2:] += boxes[:, :2]  # 座標(xmax,ymax)へ変換

    return boxes

# Non-Maximum Suppressionを行う関数


def nm_suppression(boxes, scores, overlap=0.45, top_k=500):
    """
    Non-Maximum Suppressionを行う関数。
    boxesのうち被り過ぎ（overlap以上）のBBoxを削除する。

    Parameters
    ----------
    boxes : [確信度閾値（0.01）を超えたBBox数,4]
        BBox情報。
    scores :[確信度閾値（0.01）を超えたBBox数]
        confの情報

    Returns
    -------
    keep : リスト
        confの降順にnmsを通過したindexが格納
    count：int
        nmsを通過したBBoxの数
    """

    # returnのひな形を作成
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep：torch.Size([確信度閾値を超えたBBox数])、要素は全部0

    # 各BBoxの面積areaを計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # boxesをコピーする。後で、BBoxの被り度合いIOUの計算に使用する際のひな形として用意
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # socreを昇順に並び変える
    v, idx = scores.sort(0)

    # 上位top_k個（200個）のBBoxのindexを取り出す（200個存在しない場合もある）
    idx = idx[-top_k:]

    # idxの要素数が0でない限りループする
    while idx.numel() > 0:
        i = idx[-1]  # 現在のconf最大のindexをiに

        # keepの現在の最後にconf最大のindexを格納する
        # このindexのBBoxと被りが大きいBBoxをこれから消去する
        keep[count] = i
        count += 1

        # 最後のBBoxになった場合は、ループを抜ける
        if idx.size(0) == 1:
            break

        # 現在のconf最大のindexをkeepに格納したので、idxをひとつ減らす
        idx = idx[:-1]

        # -------------------
        # これからkeepに格納したBBoxと被りの大きいBBoxを抽出して除去する
        # -------------------
        # ひとつ減らしたidxまでのBBoxを、outに指定した変数として作成する
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # すべてのBBoxに対して、現在のBBox=indexがiと被っている値までに設定(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # wとhのテンソルサイズをindexを1つ減らしたものにする
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clampした状態でのBBoxの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 幅や高さが負になっているものは0にする
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clampされた状態での面積を求める
        inter = tmp_w*tmp_h

        # IoU = intersect部分 / (area(a) + area(b) - intersect部分)の計算
        rem_areas = torch.index_select(area, 0, idx)  # 各BBoxの元の面積
        union = (rem_areas - inter) + area[i]  # 2つのエリアのANDの面積
        IoU = inter/union

        # IoUがoverlapより小さいidxのみを残す
        idx = idx[IoU.le(overlap)]  # leはLess than or Equal toの処理をする演算です
        # IoUがoverlapより大きいidxは、最初に選んでkeepに格納したidxと同じ物体に対してBBoxを囲んでいるため消去

    # whileのループが抜けたら終了

    return keep, count


# SSDの推論時にconfとlocの出力から、被りを除去したBBoxを出力する


class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=500, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)  # confをソフトマックス関数で正規化するために用意
        self.conf_thresh = conf_thresh  # confがconf_thresh=0.01より高いDBoxのみを扱う
        self.top_k = top_k  # nm_supressionでconfの高いtop_k個を計算に使用する, top_k = 200
        self.nms_thresh = nms_thresh  # nm_supressionでIOUがnms_thresh=0.45より大きいと、同一物体へのBBoxとみなす

    def forward(self, loc_data, conf_data, dbox_list):
        """
        順伝搬の計算を実行する。

        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            オフセット情報。
        conf_data: [batch_num, 8732,num_classes]
            検出の確信度。
        dbox_list: [8732,4]
            DBoxの情報

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            （batch_num、クラス、confのtop200、BBoxの情報）
        """

        # 各サイズを取得
        num_batch = loc_data.size(0)  # ミニバッチのサイズ
        num_dbox = loc_data.size(1)  # DBoxの数 = 8732
        num_classes = conf_data.size(2)  # クラス数 = 21

        # confはソフトマックスを適用して正規化する
        conf_data = self.softmax(conf_data)

        # 出力の型を作成する。テンソルサイズは[minibatch数, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # cof_dataを[batch_num,8732,num_classes]から[batch_num, num_classes,8732]に順番変更
        conf_preds = conf_data.transpose(2, 1)

        # ミニバッチごとのループ
        for i in range(num_batch):

            # 1. locとDBoxから修正したBBox [xmin, ymin, xmax, ymax] を求める
            decoded_boxes = decode(loc_data[i], dbox_list)

            # confのコピーを作成
            conf_scores = conf_preds[i].clone()

            # 画像クラスごとのループ（背景クラスのindexである0は計算せず、index=1から）
            for cl in range(1, num_classes):

                # 2.confの閾値を超えたBBoxを取り出す
                # confの閾値を超えているかのマスクを作成し、
                # 閾値を超えたconfのインデックスをc_maskとして取得
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # gtはGreater thanのこと。gtにより閾値を超えたものが1に、以下が0になる
                # conf_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])

                # scoresはtorch.Size([閾値を超えたBBox数])
                scores = conf_scores[cl][c_mask]

                # 閾値を超えたconfがない場合、つまりscores=[]のときは、何もしない
                if scores.nelement() == 0:  # nelementで要素数の合計を求める
                    continue

                # c_maskを、decoded_boxesに適用できるようにサイズを変更します
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                # l_maskをdecoded_boxesに適応します
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask]で1次元になってしまうので、
                # viewで（閾値を超えたBBox数, 4）サイズに変形しなおす

                # 3. Non-Maximum Suppressionを実施し、被っているBBoxを取り除く
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)
                # ids：confの降順にNon-Maximum Suppressionを通過したindexが格納
                # count：Non-Maximum Suppressionを通過したBBoxの数

                # outputにNon-Maximum Suppressionを抜けた結果を格納
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        return output


class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase
        self.num_classes = cfg['num_classes']

        bbox_aspect_num = cfg['bbox_aspect_num']
        
        self.resnet = Feature_extractor(in_ch=cfg['in_ch'], out_ch=cfg['out_ch'])
        self.extras = Extras(in_ch=cfg['out_ch']*8)

        s1 = self.resnet.layer2.first_block.conv1.out_channels
        s2 = self.resnet.layer4.first_block.conv1.out_channels
        s3 = self.extras.conv1_2.out_channels
        s4 = self.extras.conv2_2.out_channels
        s5 = self.extras.conv3_2.out_channels
        s6 = self.extras.conv4_2.out_channels

        self.conv_loc1 = nn.Conv2d(s1, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)
        self.conv_conf1 = nn.Conv2d(s1, bbox_aspect_num[0] * self.num_classes, kernel_size=3, padding=1)

        self.conv_loc2 = nn.Conv2d(s2, bbox_aspect_num[1]*4, kernel_size=3, padding=1)
        self.conv_conf2 = nn.Conv2d(s2, bbox_aspect_num[1]*self.num_classes, kernel_size=3, padding=1)

        self.conv_loc3 = nn.Conv2d(s3, bbox_aspect_num[2]*4, kernel_size=3, padding=1)
        self.conv_conf3 = nn.Conv2d(s3, bbox_aspect_num[2]*self.num_classes, kernel_size=3, padding=1)

        self.conv_loc4 = nn.Conv2d(s4, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)
        self.conv_conf4 = nn.Conv2d(s4, bbox_aspect_num[3] * self.num_classes, kernel_size=3, padding=1)

        self.conv_loc5 = nn.Conv2d(s5, bbox_aspect_num[4]*4, kernel_size=3, padding=1)
        self.conv_conf5 = nn.Conv2d(s5, bbox_aspect_num[4]*self.num_classes, kernel_size=3, padding=1)

        self.conv_loc6 = nn.Conv2d(s6, bbox_aspect_num[5]*4, kernel_size=1)
        self.conv_conf6 = nn.Conv2d(s6, bbox_aspect_num[5]*self.num_classes, kernel_size=1)

        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        self.L2Norm = L2Norm(s1, self.num_classes)

        if phase == 'inference':
            self.detect = Detect()
    

    def forward(self, x):
        loc = list()  # locの出力を格納
        conf = list()  # confの出力を格納
        source1, source2 = self.resnet(x)
        source3, source4, source5, source6 = self.extras(source2)
        source1 = self.L2Norm(source1)

        l1 = self.conv_loc1(source1)
        c1 = self.conv_conf1(source1)
        l2 = self.conv_loc2(source2)
        c2 = self.conv_conf2(source2)
        l3 = self.conv_loc3(source3)
        c3 = self.conv_conf3(source3)
        l4 = self.conv_loc4(source4)
        c4 = self.conv_conf4(source4)
        l5 = self.conv_loc5(source5)
        c5 = self.conv_conf5(source5)
        l6 = self.conv_loc6(source6)
        c6 = self.conv_conf6(source6)

        loc += [l1.permute(0, 2, 3, 1).contiguous(), l2.permute(0, 2, 3, 1).contiguous(),
                l3.permute(0, 2, 3, 1).contiguous(), l4.permute(0, 2, 3, 1).contiguous(),
                l5.permute(0, 2, 3, 1).contiguous(), l6.permute(0, 2, 3, 1).contiguous()]
        
        conf += [c1.permute(0, 2, 3, 1).contiguous(), c2.permute(0, 2, 3, 1).contiguous(),
                c3.permute(0, 2, 3, 1).contiguous(), c4.permute(0, 2, 3, 1).contiguous(),
                c5.permute(0, 2, 3, 1).contiguous(), c6.permute(0, 2, 3, 1).contiguous()]
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # 最後に出力する
       
        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":  # 推論時
            with torch.no_grad():
                return self.detect(output[0], output[1], output[2])

        else:  # 学習時
            return output

                

