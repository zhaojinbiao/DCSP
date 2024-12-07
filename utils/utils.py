
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
from utils.utils_bbox import decode_outputs, non_max_suppression
import shutil

# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    """image是PIL.Image.open的返回值，该函数的意义在于将图像转化成RGB三个通道"""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:   #检查image是否为3个通道
        return image 
    else:
        image = image.convert('RGB')
        return image

def preprocess_input(image):
    """
        这些操作有助于将输入图像调整到模型在训练时所期望的范围，
        并有助于提高模型的性能。这些数值（均值和标准差）通常是
        在训练集上计算得到的
    """
    image /= 255.0                              # 将图像的像素值归一化到 [0, 1] 的范围，原始像素值通常是 [0, 255]
    image -= np.array([0.485, 0.456, 0.406])    # 对图像的每个通道减去均值。这里使用了一个预定义的均值向量 [0.485, 0.456, 0.406]
    image /= np.array([0.229, 0.224, 0.225])    # 对图像的每个通道除以标准差。同样，这里使用了一个预定义的标准差向量 [0.229, 0.224, 0.225]
    return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)   

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
import random
#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def seed_everything(seed=11, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

#----------------------#
#   记录Loss
#----------------------#
from torch.utils.tensorboard import SummaryWriter
import scipy.signal
from matplotlib import pyplot as plt
class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
 
def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

import cv2

def image_preporcess1(image, target_size, gt_boxes=None):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    ih, iw    = target_size
    h,  w, _    = image.shape # (w, h, c)
    nw, nh  = int(ih), int(iw)
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = image_resized / 255.
    if gt_boxes is None:
        return image_paded
    else:
    # 将box进行调整
        box_resize = []
    for boxx in gt_boxes:
        boxx[0] = str(int(int(boxx[0]) * (nw / iw)))
        boxx[1] = str(int(int(boxx[1]) * (nh / ih)))
        boxx[2] = str(int(int(boxx[2]) * (nw / iw)))
        boxx[3] = str(int(int(boxx[3]) * (nh / ih)))
        box_resize.append(boxx)   
    return image_paded, box_resize

def image_preporcess(image, target_size, gt_boxes=None):
    '''
        image_preporcess(np.copy(image),input_shape,np.copy(box))
        image_preporcess 函数用于图像的预处理，主要包括以下几个步骤：
        将图像从 BGR 格式转换为 RGB 格式，并将像素值转换为浮点数类型。
        根据目标尺寸调整输入图像的大小，使得图像可以完全包含在指定尺寸的矩形框内，同时保持图像的原始宽高比。
        调整后的图像会被嵌入到一个新的画布中，未覆盖区域用零填充，然后对图像进行归一化处理（除以255）。
        如果提供了边界框信息 gt_boxes，则同样需要对边界框的坐标进行相应的缩放和平移，以适应调整后的图像
    
    '''

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _    = image.shape # (w, h, c)
    # # 按比例resize
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128, dtype=np.uint8)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.
    if gt_boxes is None:
        return image_paded
    else:
        np.random.shuffle(gt_boxes)
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        gt_boxes[:, 0:2][gt_boxes[:, 0:2]<0] = 0
        gt_boxes[:, 2][gt_boxes[:, 2]>iw] = iw
        gt_boxes[:, 3][gt_boxes[:, 3]>ih] = ih
        box_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        box_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        box_2 = gt_boxes[np.logical_and(box_w>1, box_h>1)]
        return image_paded, box_2
    
    # 不按比例resize
    # scale = min(iw/w, ih/h)
    # image_resized = cv2.resize(image, (iw, ih))

    # image_resized = image_resized / 255.
    # if gt_boxes is None:
    #     return image_resized
    # else:
    #     np.random.shuffle(gt_boxes)
    #     gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
    #     gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
    #     gt_boxes[:, 0:2][gt_boxes[:, 0:2]<0] = 0
    #     gt_boxes[:, 2][gt_boxes[:, 2]>iw] = iw
    #     gt_boxes[:, 3][gt_boxes[:, 3]>ih] = ih
    #     box_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    #     box_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    #     box_2 = gt_boxes[np.logical_and(box_w>1, box_h>1)]
    #     return image_paded, box_2 


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
#---------------------------------------------------#
#   检测图片
#---------------------------------------------------#
def detect_image(self, image, crop = False, count = False):
    #---------------------------------------------------#
    #   获得输入图片的高和宽
    #---------------------------------------------------#
    image_shape = np.array(np.shape(image)[0:2])
    #---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #---------------------------------------------------------#
    image       = cvtColor(image)
    #---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    #---------------------------------------------------------#
    image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
    #---------------------------------------------------------#
    #   添加上batch_size维度
    #---------------------------------------------------------#
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        outputs = self.net(images)
        outputs = decode_outputs(outputs, self.input_shape)
        #---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        #---------------------------------------------------------#
        results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                
        if results[0] is None: 
            return image

        top_label   = np.array(results[0][:, 6], dtype = 'int32')
        top_conf    = results[0][:, 4] * results[0][:, 5]
        top_boxes   = results[0][:, :4]
    #---------------------------------------------------------#
    #   设置字体与边框厚度
    #---------------------------------------------------------#
    font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
    #---------------------------------------------------------#
    #   计数
    #---------------------------------------------------------#
    if count:
        print("top_label:", top_label)
        classes_nums    = np.zeros([self.num_classes])
        for i in range(self.num_classes):
            num = np.sum(top_label == i)
            if num > 0:
                print(self.class_names[i], " : ", num)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)
    #---------------------------------------------------------#
    #   是否进行目标的裁剪
    #---------------------------------------------------------#
    if crop:
        for i, c in list(enumerate(top_label)):
            top, left, bottom, right = top_boxes[i]
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))
            
            dir_save_path = "img_crop"
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            crop_image = image.crop([left, top, right, bottom])
            crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
            print("save crop_" + str(i) + ".png to " + dir_save_path)

    #---------------------------------------------------------#
    #   图像绘制
    #---------------------------------------------------------#
    for i, c in list(enumerate(top_label)):
        predicted_class = self.class_names[int(c)]
        box             = top_boxes[i]
        score           = top_conf[i]

        top, left, bottom, right = box

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, top, left, bottom, right)
        
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    return image

def clean_dir(path):
    '''
    if delete is True: if path exist, then delete it's files and folders under it, if not, make it;
    if delete is False: if path not exist, make it.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
def get_metrics(tensor_image1, tensor_image2, psnr_only=True, reduction=False):
    
    '''
    function: given a batch tensor image pair, get the mean or sum psnr and ssim value.
    input:  range:[0,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: two python value, e.g., psnr_value, ssim_value
    '''
    
    if len(tensor_image1.shape) != 4 or len(tensor_image2.shape) != 4:
        raise Exception('a batch tensor image pair should be given!')
        
    numpy_imgs = tensor2img(tensor_image1)
    numpy_gts = tensor2img(tensor_image2)
    psnr_value, ssim_value = 0., 0.
    batch_size = numpy_imgs.shape[0]
    for i in range(batch_size):
        if not psnr_only:
            ssim_value += structural_similarity(numpy_imgs[i],numpy_gts[i], multichannel=True, gaussian_weights=True, use_sample_covariance=False)
        psnr_value += peak_signal_noise_ratio(numpy_imgs[i],numpy_gts[i])
        
    if reduction:
        psnr_value = psnr_value/batch_size
        ssim_value = ssim_value/batch_size
    
    if not psnr_only:  
        return psnr_value, ssim_value
    else:
        return psnr_value   


def tensor2img(tensor_image):
    
    '''
    function: transform a tensor image to a numpy image
    input:  range:[0,1]     type:tensor.FloatTensor  format:[b,c,h,w]  RGB
    output: range:[0,255]    type:numpy.uint8         format:[b,h,w,c]  RGB
    '''
    
    tensor_image = tensor_image*255
    tensor_image = tensor_image.permute([0, 2, 3, 1])
    if tensor_image.device != 'cpu':
        tensor_image = tensor_image.cpu()
    numpy_image = np.uint8(tensor_image.numpy())
    return numpy_image

def split_img(x, h_chunk, w_chunk):
    x = torch.cat(x.chunk(h_chunk, dim=2), dim=0)
    x = torch.cat(x.chunk(w_chunk, dim=3), dim=0)
    return x

def cat_img(x, h_chunk, w_chunk):
    x = torch.cat(x.chunk(w_chunk, dim=0), dim=3)
    x = torch.cat(x.chunk(h_chunk, dim=0), dim=2)
    return x


class AverageMeter(object):
    
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        
    def average(self, auto_reset=False):
        avg = self.sum / self.count
        
        if auto_reset:
            self.reset()
            
        return avg
