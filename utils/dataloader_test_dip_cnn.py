import torch
import colorsys
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader import MyDataset, yolo_dataset_collate
import cv2
from dip import Dip_Cnn, Dip_Filters, cfg

def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_annotation_path', type=str, default='datasets/VOC/train/voc_norm_train.txt', help='干净的train图标注文件')
    parser.add_argument('--val_annotation_path', type=str, default='voc_norm_test.txt', help='干净的test图标注文件')
    parser.add_argument('--vocfog_traindata_dir', dest='vocfog_traindata_dir', default='datasets/VOC-fog/train/JPEGImages/',help='the dir contains ten levels synthetic foggy images')
    parser.add_argument('--vocfog_valdata_dir', dest='vocfog_valdata_dir', default='datasets/VOC-fog/val/JPEGImages/',help='the dir contains ten levels synthetic foggy images')
    parser.add_argument('--input_shape', type=list, default=[640,640], help='输入图片resize大小')
    parser.add_argument('--save_dir', type=str, default="logs", help='默认为logs')
    parser.add_argument('--num_workers', type=int, default=4, help='默认为4')
    parser.add_argument('--batch_size', type=int, default= 8, help='根据是否freeze动态调整batch_size')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):


    """设置种子"""
    np.random.seed(0)

    input_shape = [640, 640]
    num_classes = 4

    opt = parse_opt()

    with open(opt.train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()

    """建立数据集类对象"""
    train_dataset = MyDataset(train_lines, opt, train = False)  


    num_workers = 4

    """建立导入器对象"""
    gen = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=num_workers, pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
    print(len(gen))

    for iteration, batch in enumerate(gen):
        images, bboxes, foggy_images = batch[0], batch[1], batch[2]
        print("images type", type(images))
        print("images shape", images.shape)
        print("bboxes type", type(bboxes))
        print("bboxes len", len(bboxes))
        print(bboxes)
        # print(bboxes)
        print("foggy_images type", type(foggy_images))
        print("foggy_images shape", foggy_images.shape)        
        print('-'*50)

        DIP_Cnn = Dip_Cnn() # DIP & pp
        # 转换数据类型
        foggy_images_np = foggy_images.astype(np.float32)
        # NumPy的表示转换为PyTorch的表示
        foggy_images_tensor = torch.from_numpy(foggy_images_np).to('cpu')
        # 首先，调整张量的顺序，从 [批次大小, 高度, 宽度, 通道数] 到 [批次大小, 通道数, 高度, 宽度]
        foggy_images_tensor_permuted = foggy_images_tensor.permute(0, 3, 1, 2) # torch.Size([4, 640, 640, 3])--torch.Size([4, 3, 640, 640])
        resized_images_tensor = F.interpolate(foggy_images_tensor_permuted, size=(256, 256), mode='bilinear', align_corners=False) # torch.Size([4, 3, 256, 256])
        feature_para = DIP_Cnn(resized_images_tensor)
        filtered_img = Dip_Filters(feature_para, cfg, foggy_images_tensor_permuted)# [b, 3, 640, 640]

        filtered_img = filtered_img.permute(0,2,3,1)
        filtered_img = filtered_img.to("cpu").detach().numpy()# [b, 640, 640, 3]
        # filtered_img =  np.clip(filtered_img, 0.0, 1.0).astype(np.uint8)
        # filtered_img = np.clip(filtered_img*255, 0, 255)# (b, 1330, 13303, 3)
        
        # 创建一个包含8个子图的画布（4个用于显示 images，4个用于显示 foggy_images）
        fig, axs = plt.subplots(3, 8, figsize=(64, 12))
        color = (0, 255, 0)  # 框的颜色，这里使用绿色
        thickness = 2  # 框的线条粗细
        # 循环显示每张原始图像
        # 循环显示每张原始图像和模糊图像
        for i in range(opt.batch_size):
            # 绘制原始图像上的边界框
            for bbox in bboxes[i]:
                x1, y1, x2, y2 = int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())
                cv2.rectangle(images[i], (x1, y1), (x2, y2), color, thickness)
            
            axs[0, i].imshow(images[i])
            axs[0, i].set_title(f"Original Image {i+1}")
            axs[0, i].axis('off')  # 关闭坐标轴

            # 绘制模糊图像上的边界框
            for bbox in bboxes[i]:
                x1, y1, x2, y2 = int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())
                cv2.rectangle(foggy_images[i], (x1, y1), (x2, y2), color, thickness)
            
            axs[1, i].imshow(foggy_images[i])
            axs[1, i].set_title(f"Foggy Image {i+1}")
            axs[1, i].axis('off')  # 关闭坐标轴

            # 绘制原始图像上的边界框
            for bbox in bboxes[i]:
                x1, y1, x2, y2 = int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())
                cv2.rectangle(filtered_img[i], (x1, y1), (x2, y2), color, thickness)
            
            axs[2, i].imshow(filtered_img[i])
            axs[2, i].set_title(f"Restored Image {i+1}")
            axs[2, i].axis('off')  # 关闭坐标轴
        # 添加延迟以便能够看到绘制的结果
        plt.pause(10)  # 延迟1秒
        plt.show()


        if iteration == 0:
            break

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

