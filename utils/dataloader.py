import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import image_preporcess1, image_preporcess


class MyDataset(Dataset):
    def __init__(self, opt, train):
                #  \,annotation_lines, input_shape, num_classes, epoch_length, \
                #         mosaic, mixup, mosaic_prob, mixup_prob, is_train, special_aug_ratio = 0.7):
                # train_lines, opt, num_classes, train = True
                # MyDataset(train_lines, opt, train = True) 
        """
        Args:
            annotation_lines: train_lines:train_annotation_path:干净的图标注文件
            input_shape:      输入到模型的图像尺寸
            num_classes:      类别数量
            epoch_length:
            mosaic:           是否使用马赛克数据增强
            mixup:            是否使用mix_up数据增强
            mosaic_prob:      当mosaic=True时，图片进行马赛克数据增强的概率
            mixup_prob:       当mixup=True时，图片进行mixup数据增强的概率
            train:            bool类型,true或者false

            train_dataset = YoloDataset(train_lines, opt, num_classes, train = True)  
            val_dataset   = YoloDataset(val_lines, opt, num_classes, train = False)

        """
        super(MyDataset, self).__init__()
        # self.annotation_lines   = annotation_lines
        self.input_shape        = opt.input_shape# 图像尺寸
        # self.num_classes        = num_classes# 类别数
        # self.epoch_length       = opt.opt.UnFreeze_Epoch
        self.train              = train
        self.epoch_now          = -1  # 用来对读取了多少张图片进行计数
        self.vocfog_traindata_dir = opt.vocfog_traindata_dir
        self.vocfog_valdata_dir = opt.vocfog_valdata_dir # 没有验证集用不到
        if self.train:
            self.annotations = self.load_annotations(opt.train_clean_annotation_path)
        else:
            self.annotations = self.load_annotations(opt.val_clean_annotation_path)
        # self.annotations = self.load_annotations(opt.train_clean_annotation_path if self.train else opt.val_clean_annotation_path)
        self.num_samples = len(self.annotations) # 图片数量
        self.length      = self.num_samples# 注释文件的行数，也就是图片的数量
        self.num_batchs = int(np.ceil(self.num_samples / opt.batch_size)) # 一轮的batch数量
        self.batch_count = 0
        # import pdb
        # pdb.set_trace()


    def load_annotations(self, train_annotation_path):
        with open(train_annotation_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        # np.random.shuffle(annotations)
        print('###################the total image:', len(annotations))
        return annotations

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = index % self.length#将索引调整到0-self.length，防止索引越界
        # print(self.annotation_lines[index]) #  用于打印读取的标注信息
        image_data, box, foggy_image  = self.get_random_data(self.annotations[index],self.input_shape)
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2] # 将右下角坐标减去左上角坐标，以计算出边界框的宽度和高度
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2 # 将左上角坐标加上宽度和高度的一半，以计算出边界框的中心点坐标
        return image_data, box, foggy_image

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def get_random_data(self, annotation_line, input_shape):
        #------------------------------#
        #   将图片和标注信息分割
        #------------------------------#       
        line  = annotation_line.split()     

        # print(line) 
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image = cv2.imread(line[0])
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        image_path = line[0]
        img_name = image_path.split('/')[-1]
        image_name = img_name.split('.')[0]
        image_name_index = img_name.split('.')[1]

        if not self.train:
     

            # image_data, box = image_preporcess(np.copy(image),input_shape,np.copy(box))
            clean_image_data, box = image_preporcess(np.copy(image),input_shape, np.copy(box))
            img_name = self.vocfog_valdata_dir + image_name  + '.' + image_name_index
            foggy_image = cv2.imread(img_name)
            foggy_image_data = image_preporcess(np.copy(foggy_image),input_shape)

            # image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # box = np.copy(box)
            return clean_image_data, box, foggy_image_data
        
        else:
            # 训练状态时为 数据混合训练
            # image_path = line[0]
            # img_name = image_path.split('/')[-1]
            # image_name = img_name.split('.')[0]
            # image_name_index = img_name.split('.')[1]
            img_name = self.vocfog_traindata_dir + image_name  + '.' + image_name_index
            foggy_image = cv2.imread(img_name)
            # # 2/3的概率载入fog作为训练数据
            # if random.randint(0, 2) > 0:
                # foggy_image = cv2.imread(img_name)
                # 以 50% 的概率来决定是否进行水平翻转
            if random.random() < 0.5:
                # 随机产生 [0.0, 1.0) 
                _, w, _ = image.shape
                #  :表示一个轴， ::-1表示 表示逆序选择，即反转该轴上的顺序
                image = image[:, ::-1, :] # clean , 水平方向翻转
                foggy_image = foggy_image[:, ::-1, :] # fog ， 水平方向翻转
                # image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # foggy_image = foggy_image.transpose(Image.FLIP_LEFT_RIGHT)
                # bboxes 数组中选择所有行（冒号表示选择所有行），但是只选择列索引为 2 和 0 的元素。\
                # 换句话说，它选取了每个边界框的第三个和第一个元素，即 x_max 和 x_min 的值
                box[:, [0, 2]] = w - box[:, [2, 0]] # 对边界框坐标的调整，使其适应于水平翻转后的图像

        
            # 以 50% 的概率确定是否执行裁剪操作
            if random.random() < 0.5:
                h, w, _ = image.shape
                # 计算所有边界框的最小 x、y 坐标和最大 x、y 坐标，得到一个包围所有边界框的最大边界框
                max_bbox = np.concatenate([np.min(box[:, 0:2], axis=0), np.max(box[:, 2:4], axis=0)],
                                            axis=-1)

                max_l_trans = max_bbox[0] # 计算裁剪边界的上边界
                max_u_trans = max_bbox[1] # 计算裁剪边界的下边界
                max_r_trans = w - max_bbox[2] # 计算裁剪边界的左边界
                max_d_trans = h - max_bbox[3] # 计算裁剪边界的右边界
                
                # 裁剪区域的边界坐标
                crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
                crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
                crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
                crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

                # 根据计算得到的裁剪区域边界，对图像及fog图像进行裁剪
                image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
                foggy_image = foggy_image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

                # 更新边界框的坐标以适应裁剪后的图像
                box[:, [0, 2]] = box[:, [0, 2]] - crop_xmin
                box[:, [1, 3]] = box[:, [1, 3]] - crop_ymin

            # 以 50% 的概率确定是否执行平移操作  
            # if random.random() < 0.5:
            #     h, w, _ = image.shape
            #     max_bbox = np.concatenate([np.min(box[:, 0:2], axis=0), np.max(box[:, 2:4], axis=0)],axis=-1)

            #     max_l_trans = max_bbox[0]
            #     max_u_trans = max_bbox[1]
            #     max_r_trans = w - max_bbox[2]
            #     max_d_trans = h - max_bbox[3]

            #     # 随机生成平移量 tx 和 ty
            #     tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            #     ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            #     M = np.array([[1, 0, tx], [0, 1, ty]])
            #     image = cv2.warpAffine(image, M, (w, h)) # 对原始图像进行仿射变换，实现平移操作
            #     foggy_image = cv2.warpAffine(foggy_image, M, (w, h)) # 对fog图像进行相同的仿射变换

            #     box[:, [0, 2]] = box[:, [0, 2]] + tx
            #     box[:, [1, 3]] = box[:, [1, 3]] + ty
            clean_image_data, box = image_preporcess(np.copy(image),input_shape, np.copy(box))
            # img_name = self.vocfog_traindata_dir + image_name  + '.' + image_name_index
            foggy_image_data = image_preporcess(np.copy(foggy_image),input_shape)
            return clean_image_data, box, foggy_image_data

# DataLoader中collate_fn使用：一次性导入多张图片及其标签（即一个batch的data和targets）
def yolo_dataset_collate(batch):
    # image_data, box, foggy_image
    images = []
    bboxes = []
    # add
    foggy_images = []
    for img, box, foggy_imag in batch:
        images.append(img)
        bboxes.append(box)
        foggy_images.append(foggy_imag)

    images = np.array(images)
    # type(foggy_images) <class 'list'>
    foggy_images= np.array(foggy_images)

    # bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes, foggy_images