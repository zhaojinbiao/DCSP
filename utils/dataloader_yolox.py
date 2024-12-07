from random import sample, shuffle
# from turtle import clear

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, clearimage_lines, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.clearimage_lines   = clearimage_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image, box, clearimg, img_id      = self.get_random_data(self.annotation_lines[index], self.clearimage_lines[index], self.input_shape, random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        clearimg    = np.transpose(preprocess_input(np.array(clearimg, dtype=np.float32)), (2, 0, 1))
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box, clearimg, img_id

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, clearimage_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):

        
        line    = annotation_line.split()


        clearline = clearimage_line.split()
        image_path = line[0] # voc-fog/test/VOC2007/VOCtest-FOG/2007_003831.jpg

        # import pdb
        # pdb.set_trace()

        img_name = image_path.split('/')[-1] # 2007_003831.jpg
        image_name = img_name.split('.')[0] # 2007_003831
        # image_name_index = img_name.split('.')[1] # jpg
        img_id = image_path

        # resize: 640-640 & cvtColor:check 通道数
        image   = Image.open(line[0])
        image   = cvtColor(image)

        clearimg = Image.open(clearline[0])
        clearimg = cvtColor(clearimg)

        iw, ih  = image.size
        h, w    = input_shape

        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # val - test : 非训练状态
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image       = image.resize((nw, nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            '''clear'''
            clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
            new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
            new_clearimg.paste(clearimg, (dx, dy))
            clear_image_data = np.array(new_clearimg, np.float32)

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


            return image_data, box, clear_image_data, img_id

        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        clearimg = clearimg.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        '''clear'''
        new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
        new_clearimg.paste(clearimg, (dx, dy))
        clearimg = new_clearimg

        # flip & hsv
        flip = self.rand()<.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            clearimg = clearimg.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        clear_image_data = np.array(clearimg, np.uint8)

        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype

        hue1, sat1, val1 = cv2.split(cv2.cvtColor(clear_image_data, cv2.COLOR_RGB2HSV))
        dtype1 = clear_image_data.dtype

        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        x1 = np.arange(0, 256, dtype=r.dtype)
        lut_hue1 = ((x1 * r[0]) % 180).astype(dtype)
        lut_sat1 = np.clip(x1 * r[1], 0, 255).astype(dtype)
        lut_val1 = np.clip(x1 * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        clear_image_data = cv2.merge((cv2.LUT(hue1, lut_hue1), cv2.LUT(sat1, lut_sat1), cv2.LUT(val1, lut_val1)))
        clear_image_data = cv2.cvtColor(clear_image_data, cv2.COLOR_HSV2RGB)

        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box, clear_image_data, img_id



def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    clearimg = []
    img_ides = []
    for img, box, clear, img_id in batch:
        images.append(img)
        bboxes.append(box)
        clearimg.append(clear)
        img_ides.append(img_id)
    images = np.array(images)
    clearimg = np.array(clearimg)
    return  clearimg, bboxes, images, img_ides


