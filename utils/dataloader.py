from random import sample, shuffle

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from utils.utils import preprocess_input


def cvtColor_cv2(image):
    """Convert image to RGB using OpenCV (faster than PIL)"""
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            return image
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    # 灰度图转RGB
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)

        self.bbox_attrs         = 5 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)

            if self.mixup and self.rand() < self.mixup_prob:
                lines           = sample(self.annotation_lines, 1)
                image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)

        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)

        #---------------------------------------------------#
        #   对真实框进行预处理
        #---------------------------------------------------#
        nL          = len(box)
        labels_out  = np.zeros((nL, 6))
        if nL:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            #---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            #---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return image, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """使用 OpenCV 优化的数据加载"""
        line    = annotation_line.split()
        #------------------------------#
        #   使用 OpenCV 读取图像 (比 PIL 快 3-5 倍)
        #------------------------------#
        image   = cv2.imread(line[0])
        image   = cvtColor_cv2(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        ih, iw  = image.shape[:2]
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   使用 OpenCV 缩放 (比 PIL 快)
            #---------------------------------#
            nw = max(1, nw)
            nh = max(1, nh)
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.full((h, w, 3), 128, dtype=np.uint8)
            new_image[dy:dy+nh, dx:dx+nw] = image
            image_data = new_image.astype(np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)

        # 确保 nw, nh 至少为 1
        nw = max(1, nw)
        nh = max(1, nh)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        # 处理边界情况
        dx = int(self.rand(0, max(1, w-nw)))
        dy = int(self.rand(0, max(1, h-nh)))

        # 计算实际需要复制的区域
        src_x1 = max(0, -dx)
        src_y1 = max(0, -dy)
        src_x2 = nw - max(0, dx + nw - w)
        src_y2 = nh - max(0, dy + nh - h)

        dst_x1 = max(0, dx)
        dst_y1 = max(0, dy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        if src_x2 > src_x1 and src_y2 > src_y1:
            new_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip:
            image = cv2.flip(image, 1)

        #---------------------------------#
        #   对图像进行色域变换
        #   使用 OpenCV 实现 (比 PIL 快)
        #---------------------------------#
        hue_r, sat_r, val_r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        # 转换到 HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

        # 应用变换
        hsv[:,:,0] = (hsv[:,:,0] * hue_r) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_r, 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * val_r, 0, 255)

        image_data = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip:
                box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]

        return image_data, box
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        """使用 OpenCV 优化的 Mosaic 数据增强"""
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   使用 OpenCV 读取图片
            #---------------------------------#
            image = cv2.imread(line_content[0])
            image = cvtColor_cv2(image)

            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            ih, iw = image.shape[:2]
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])

            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = cv2.flip(image, 1)
                box[:, [0,2]] = iw - box[:, [2,0]]

            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)

            # 确保 nw, nh 至少为 1
            nw = max(1, nw)
            nh = max(1, nh)
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh

            # 处理边界情况：确保图片在画布范围内
            src_x1 = max(0, -dx)
            src_y1 = max(0, -dy)
            src_x2 = nw - max(0, dx + nw - w)
            src_y2 = nh - max(0, dy + nh - h)

            dst_x1 = max(0, dx)
            dst_y1 = max(0, dy)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            new_image = np.full((h, w, 3), 128, dtype=np.uint8)
            if src_x2 > src_x1 and src_y2 > src_y1:
                new_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
            image_data = new_image

            index = index + 1
            box_data = []
            #---------------------------------#
            #   对box进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3], dtype=np.uint8)
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        #---------------------------------#
        #   对图像进行色域变换 (OpenCV 优化)
        #---------------------------------#
        hue_r, sat_r, val_r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hsv = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] * hue_r) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_r, 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * val_r, 0, 255)
        new_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
    
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes

# # DataLoader中collate_fn使用
# def yolo_dataset_collate(batch):
#     images      = []
#     n_max_boxes = 0
#     bs          = len(batch)
#     for i, (img, box) in enumerate(batch):
#         images.append(img)
#         n_max_boxes = max(n_max_boxes, len(box))
    
#     bboxes  = torch.zeros((bs, n_max_boxes, 4))
#     labels  = torch.zeros((bs, n_max_boxes, 1))
#     masks   = torch.zeros((bs, n_max_boxes, 1))
    
#     for i, (img, box) in enumerate(batch):
#         _sub_length = len(box)
#         bboxes[i, :_sub_length] = box[:, :4]
#         labels[i, :_sub_length] = box[:, 4]
#         masks[i, :_sub_length]  = 1
    
#     images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
#     return images, bboxes, labels, masks
