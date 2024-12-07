
import colorsys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.dataloader import MyDataset, yolo_dataset_collate
import cv2

def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_annotation_path', type=str, default='datasets/VOC/train/voc_norm_train.txt', help='干净的train图标注文件')
    parser.add_argument('--val_annotation_path', type=str, default='voc_norm_test.txt', help='干净的test图标注文件')
    parser.add_argument('--vocfog_traindata_dir', dest='vocfog_traindata_dir', default='datasets/VOC-fog/train/JPEGImages/',help='the dir contains ten levels synthetic foggy images')
    parser.add_argument('--vocfog_valdata_dir', dest='vocfog_valdata_dir', default='datasets/VOC-fog/val/JPEGImages/',help='the dir contains ten levels synthetic foggy images')
    parser.add_argument('--input_shape', type=list, default=[640,640], help='输入图片resize大小')
    parser.add_argument('--save_dir', type=str, default="logs", help='默认为logs')
    parser.add_argument('--num_workers', type=int, default=4, help='默认为4')
    parser.add_argument('--batch_size', type=int, default= 4, help='根据是否freeze动态调整batch_size')

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
    train_dataset = MyDataset(train_lines, opt, train = True)  


    batch_size = 4
    num_workers = 4

    """建立导入器对象"""
    gen = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,drop_last=True, collate_fn=yolo_dataset_collate)
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
        print("foggy_images type", foggy_images.shape)        
        print('-'*50)

        
        # 创建一个包含8个子图的画布（4个用于显示 images，4个用于显示 foggy_images）
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        color = (0, 255, 0)  # 框的颜色，这里使用绿色
        thickness = 2  # 框的线条粗细
        # 循环显示每张原始图像
        # 循环显示每张原始图像和模糊图像
        for i in range(4):
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

        # 添加延迟以便能够看到绘制的结果
        plt.pause(10)  # 延迟1秒
        plt.show()

        if iteration == 0:
            break

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

