import os
import gc 
import random
import cv2
import torch
import torch.distributed as dist
import argparse
from pathlib import Path
import torch.nn as nn
import datetime
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from functools import partial
import torch.nn.functional as F


from tensorboardX import SummaryWriter
# 初始化SummaryWriter
writer = SummaryWriter('runs/multi_losses')

from utils.general import  increment_path, print_args
from net.DIPYOLO import DIPYOLO
from net.AODnet import AODnet
from net.EndToEndFramework import EndToEndFramework
from net.darknet2 import CSPDarknet_Encode_Focus, CSPDarknet_Encoder_Focus_next
from net.NAFNet_arch import NAFNet, Encooder_Res_L
from net.AGPLPLS_training import YOLOLoss, MMDLoss, select_device, weights_init, set_optimizer_lr, get_lr_scheduler, ModelEMA
from utils.utils import get_classes,show_config, seed_everything, LossHistory, worker_init_fn
from utils.dataloader_yolox import YoloDataset, yolo_dataset_collate
from torch.utils.data import DataLoader
from utils.utils_fit import fit_one_epoch
from tqdm import tqdm
from utils.utils import get_lr, LossHistory
from utils.psnr_ssim import batch_PSNR
from ptflops import get_model_complexity_info
# from utils.utils_bbox import decode_outputs, non_max_suppression, draw_boxes
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import colorsys


import warnings
warnings.filterwarnings('ignore')
# # 初始化wandbe
# import wandb
# wandb.init(project="IAYOLOX3quanbujiego", entity="2495264314", name = 'IAYOLOX3-Train-voc-fog-9578')

# def print_para_num(model):
    
#     '''
#     function: print the number of total parameters and trainable parameters
#     '''
    
#     total_params = sum(p.numel() for p in model.parameters())
#     total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     inp_shape = (3, 256, 256)
#     macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)
#     print('GMACs with shape ' + str(inp_shape) + ':', macs)
#     print('total parameters: %d' % total_params)
#     print('trainable parameters: %d' % total_trainable_params)
seed_value = 3407   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    
    
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=3407, help='Global training seed')
    parser.add_argument('--classes_path', type=str, default='datasets/classes/voc_fog_classes.txt', help='分类标签路径')
    parser.add_argument('--model_path', type=str, default='', help='加载模型路径') # 'pth/yolox_s.pth'
    # parser.add_argument('--model_path1', type=str, default='pth/dehazer.pth', help='加载模型路径') # 

    parser.add_argument('--NAFNet_model_path1', type=str, default='pth/NAFNet-GoPro-width32.pth', help='加载模型路径') # 
    parser.add_argument('--Encooder_Res_L_model_path2', type=str, default='pth/NAFNet-GoPro-width32.pth', help='加载模型路径') # 

    parser.add_argument('--DIPYOLO_model_path3', type=str, default='pth/yolox_s.pth', help='加载模型路径') # 'pth/yolox_s.pth'
    parser.add_argument('--CSPDarknet_Encode_Focus_model_path4', type=str, default='pth/yolox_s.pth', help='加载模型路径') # 'pth/yolox_s.pth'
    parser.add_argument('--CSPDarknet_Encoder_Focus_next_model_path5', type=str, default='pth/yolox_s.pth', help='加载模型路径') # pth/yolox_s.pth



    parser.add_argument('--train_fog_annotation_path', type=str, default='voc-fog/2007_train_fog.txt', help='雾图训练集标注')
    parser.add_argument('--val_fog_annotation_path', type=str, default='voc-fog/2007_val_fog.txt', help='雾图验证集标注')
    parser.add_argument('--train_clean_annotation_path', type=str, default='voc-fog/2007_train.txt', help='干净图训练集标注')
    parser.add_argument('--val_clean_annotation_path', type=str, default='voc-fog/2007_val.txt', help='干净图验证集标注')

    # parser.add_argument('--train_annotation_path', type=str, default='datasets/VOC/train/voc_norm_test.txt', help='干净的train图标注文件')# datasets/VOC/train/voc_norm_train.txt
    # parser.add_argument('--val_annotation_path', type=str, default='datasets/VOC/test/voc_norm_test.txt', help='干净的test图标注文件')
    parser.add_argument('--vocfog_traindata_dir', dest='vocfog_traindata_dir', default='voc-fog/train/VOC2007/VOC2007-FOG/',help='train the dir contains ten levels synthetic foggy images')
    parser.add_argument('--vocfog_valdata_dir', dest='vocfog_valdata_dir', default='voc-fog/train/VOC2007/VOC2007-FOG/',help='the dir contains ten levels synthetic foggy images')
    # 验证的--val_annotation_path --val_clear_annotation_path 因为我没有使用验证集，所以没有这两个路径
    # parser.add_argument('--clear_annotation_path', type=str, default='2007_train.txt', help='2007_train.txt')
    # parser.add_argument('--val_clear_annotation_path', type=str, default='2007_val.txt', help='2007_val.txt')
    parser.add_argument('--input_shape', type=list, default=[640,640], help='输入图片resize大小')
    parser.add_argument('--mosaic', type=bool, default=False, help='是否mosaic')
    parser.add_argument('--Init_Epoch', type=int, default=0, help='默认为解冻训练')
    parser.add_argument('--Freeze_Epoch', type=int, default=99, help='默认为解冻训练')
    parser.add_argument('--Freeze_batch_size', type=int, default=4, help='冻结时batch_size=16')
    parser.add_argument('--UnFreeze_Epoch', type=int, default=99, help='解冻训练Epoch_num设置')
    parser.add_argument('--Unfreeze_batch_size', type=int, default=2, help='解冻时batch_size=16')
    parser.add_argument('--Freeze_Train', type=bool, default= True, help='是否进行冻结训练')
    parser.add_argument('--batch_size', type=int, default= None, help='根据是否freeze动态调整batch_size')
    parser.add_argument('--Init_lr', type=float, default=1e-2, help='默认为1e-2')
    parser.add_argument('--Min_lr', type=float, default=1e-4, help='默认为1e-6,Init_lr * 0.01')
    parser.add_argument('--optimizer_type', type=str, default="sgd", help='默认为sgd')
    parser.add_argument('--momentum', type=float, default=0.937, help='默认为0.937')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='默认为5e-4')
    parser.add_argument('--lr_decay_type', type=str, default="cos", help='默认为cos')
    parser.add_argument('--save_period', type=int, default=1, help='默认为10')
    parser.add_argument('--save_dir', type=str, default="logs", help='默认为logs')
    parser.add_argument('--num_workers', type=int, default=4, help='默认为4')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--local-rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--distributed', type=bool, default=True, help='')
    parser.add_argument('--sync_bn', type=bool, default=True,help='')
    parser.add_argument('--eval_flag',type=bool,default=True,help='')
    parser.add_argument('--eval_period',type=int,default=1,help='')
    parser.add_argument('--UnFreeze_flag', type=bool, default=False, help='')
    parser.add_argument('--phi',type=str,default='s',help='指定model类型S-X')
    parser.add_argument('--font_path', type=str, default='datasets/simhei.ttf', help='字体文件')
    # parser.add_arfument('--duandian',type=bool,default=False,help='')
    # parser.add_arfument('--duandian_epoch',type=int,default=90,help='')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):

    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_git_status()
        # check_requirements()
    # --------------------------------------------defin model---------------------------------------------------
    class_names, num_classes = get_classes(opt.classes_path)

    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]


    # enc_blks = [2, 2, 2, 20]
    # middle_blk_num = 2
    # dec_blks = [2, 2, 2, 2]


    enc_blks = [2, 2]
    middle_blk_num = 1
    dec_blks = [1, 1, 1]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num =  1
    # dec_blks = [1, 1, 1, 1]

    model1 = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    enc_blks_res = [2, 2]
    EncooderRes = Encooder_Res_L(img_channel=img_channel, width=width, enc_blk_nums=enc_blks_res)

    model3 = DIPYOLO(num_classes, opt.phi)
    depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    depth, width    = depth_dict[opt.phi], width_dict[opt.phi]
    EncoderFocus = CSPDarknet_Encode_Focus(depth, width, depthwise = False, act="silu")
    EncoderFocus_next = CSPDarknet_Encoder_Focus_next(depth, width, depthwise = False, act="silu")
    
    model = EndToEndFramework('s', model1,  EncooderRes, model3, EncoderFocus, EncoderFocus_next)

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a valid --batch-size'
        # assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        ngpus_per_node  = torch.cuda.device_count()
        # dist.init_process_group(backend='nccl', world_size = WORLD_SIZE, rank=LOCAL_RANK)
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")


    # model = torch.nn.parallel.DistributedDataParallel(model.cuda(LOCAL_RANK), device_ids=[LOCAL_RANK])

    # --------------------------------------------load pre model weight---------------------------------------------------
    if opt.NAFNet_model_path1 != '':
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        # model1.load_state_dict(torch.load(opt.model_path1, map_location=device))
        model_dict1     = model1.state_dict()
        pretrained_dict1 = torch.load(opt.NAFNet_model_path1, map_location = device)
        # print(pretrained_dict1.keys())# ['params']
        load_key1, no_load_key1, temp_dict1 = [], [], {}
        for k, v in pretrained_dict1['params'].items():
            if k in model_dict1.keys() and np.shape(model_dict1[k]) == np.shape(v):
                temp_dict1[k] = v
                load_key1.append(k)
            else:
                no_load_key1.append(k)
        print('load_key1:{}'.format(len(load_key1)))
        print('no_load_key1:{}'.format(len(no_load_key1)))
        print('-'*50)
        model_dict1.update(temp_dict1)
        model1.load_state_dict(model_dict1)


    if opt.Encooder_Res_L_model_path2 != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        # if LOCAL_RANK == 0:
        #     print('Load weights {}.'.format(opt.model_path))      
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict2     = EncooderRes.state_dict()
        # print(model_dict)
        pretrained_dict2 = torch.load(opt.Encooder_Res_L_model_path2, map_location = device)
        load_key2, no_load_key2, temp_dict2 = [], [], {}
        for k, v in pretrained_dict2['params'].items():
            if k in model_dict2.keys() and np.shape(model_dict2[k]) == np.shape(v):
                temp_dict2[k] = v
                load_key2.append(k)
            else:
                no_load_key2.append(k)
        print('load_key2:{}'.format(len(load_key2)))
        print('no_load_key2:{}'.format(len(no_load_key2)))
        print('-'*50)
        model_dict2.update(temp_dict2)
        EncooderRes.load_state_dict(model_dict2)

    if opt.DIPYOLO_model_path3 != '':
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        # model1.load_state_dict(torch.load(opt.model_path1, map_location=device))
        model_dict3     = model3.state_dict()
        pretrained_dict3 = torch.load(opt.DIPYOLO_model_path3, map_location = device)
        load_key3, no_load_key3, temp_dict3 = [], [], {}
        for k, v in pretrained_dict3.items():
            if k in model_dict3.keys() and np.shape(model_dict3[k]) == np.shape(v):
                temp_dict3[k] = v
                load_key3.append(k)
            else:
                no_load_key3.append(k)
        print('load_key3:{}'.format(len(load_key3)))
        print('no_load_key3:{}'.format(len(no_load_key3)))
        print('-'*50)
        model_dict3.update(temp_dict3)
        model3.load_state_dict(model_dict3)

    if opt.CSPDarknet_Encode_Focus_model_path4 != '':
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        # model1.load_state_dict(torch.load(opt.model_path1, map_location=device))
        model_dict4     = EncoderFocus.state_dict() # print(len(model_dict4.keys())) 42个key()
        pretrained_dict4 = torch.load(opt.CSPDarknet_Encode_Focus_model_path4, map_location = device)
        prefix_to_remove = "backbone.backbone."
        load_key4, no_load_key4, temp_dict4 = [], [], {}
        for k, v in pretrained_dict4.items():
            if k.startswith(prefix_to_remove):
                new_key = k[len(prefix_to_remove):]  # 去除指定前缀
                if new_key in model_dict4 and v.shape == model_dict4[new_key].shape:
                    temp_dict4[new_key] = v
                    load_key4.append(new_key)
                else:
                    no_load_key4.append(new_key)
        print('load_key4:{}'.format(len(load_key4))) # load_key: 42
        print('no_load_key4:{}'.format(len(no_load_key4))) # no_load_key4:168
        model_dict4.update(temp_dict4)
        EncoderFocus.load_state_dict(model_dict4)


    if opt.CSPDarknet_Encoder_Focus_next_model_path5 != '':
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        # model1.load_state_dict(torch.load(opt.model_path1, map_location=device))
        model_dict5      = EncoderFocus_next.state_dict()
        # print(len(model_dict5.keys())) # 206个key()
        pretrained_dict5 = torch.load(opt.CSPDarknet_Encoder_Focus_next_model_path5, map_location = device)
        prefix_to_remove = "backbone.backbone."
        load_key5, no_load_key5, temp_dict5 = [], [], {}
        for k, v in pretrained_dict5.items():
            if k.startswith(prefix_to_remove):
                new_key = k[len(prefix_to_remove):]  # 去除指定前缀
                if new_key in model_dict5.keys() and np.shape(model_dict5[new_key]) == np.shape(v):
                    temp_dict5[new_key] = v
                    load_key5.append(new_key)
                else:
                    no_load_key5.append(new_key)
        print('load_key5:{}'.format(len(load_key5))) # load_key5:168
        print('no_load_key5:{}'.format(len(no_load_key5))) # no_load_key5:42
        print('-'*50)
        model_dict5.update(temp_dict5)
        EncoderFocus_next.load_state_dict(model_dict5)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        # if LOCAL_RANK == 0:
        #     print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        #     print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        #     print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")


    
    # weights_init(model)

    # print_para_num(model)
    # -----------------------------------------------------------------------defin loss---------------------------------------------------
    #----------------------#
    #   获得损失函数
    #----------------------#   
    loss_AOD = torch.nn.MSELoss()
    # loss_feature = MMDLoss()
    loss_feature = torch.nn.MSELoss()
    loss_detection = YOLOLoss(num_classes)
    #----------------------#
    #   记录Loss
    #----------------------#
    if RANK in {-1, 0}:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(opt.save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=opt.input_shape)
    else:
        loss_history    = None
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    fp16   = False    
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if opt.sync_bn and ngpus_per_node > 1 and opt.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif opt.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")
    if opt.distributed:
        #----------------------------#
        #   多卡平行运行
        #----------------------------#
        model_train = model_train.cuda(LOCAL_RANK)
        model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[LOCAL_RANK], find_unused_parameters=True)
    else:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #----------------------------#
    #   权值平滑
    #----------------------------#
    ema = ModelEMA(model_train) if RANK in {-1, 0} else None
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#                     
    with open(opt.train_fog_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(opt.val_fog_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    with open(opt.train_clean_annotation_path, encoding='utf-8') as f:
        clear_lines = f.readlines()
    with open(opt.val_clean_annotation_path, encoding='utf-8') as f:
        val_clear_lines = f.readlines()

    num_train   = len(train_lines)
    num_val     = len(val_lines)
    #---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数 
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    #----------------------------------------------------------#
    wanted_step = 5e4 if opt.optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // opt.Unfreeze_batch_size * opt.UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // opt.Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // opt.Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(opt.optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, opt.Unfreeze_batch_size, opt.UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    

    if True:
        opt.UnFreeze_flag = False
        batch_size = opt.Freeze_batch_size  if opt.Freeze_Train else opt.UnFreeze_batch_size
        if opt.distributed:
            # batch_size = batch_size * ngpus_per_node
            batch_size = batch_size * ngpus_per_node
  

        if opt.Freeze_Train:

            for name, param in model1.named_parameters():
                if 'decoders' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # Freeze EncooderRes
            for param in EncooderRes.parameters():
                param.requires_grad = False

            # Free not is 'head'
            for name, param in model3.named_parameters():
                if 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False 
                    
            # Freeze
            for param in EncoderFocus.parameters():
                param.requires_grad = False 

            # Freeze
            for param in EncoderFocus_next.parameters():
                param.requires_grad = False

            # for name, param in model.named_parameters():
            #     if 'head' in name:
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False 
                    
        nbs  = 64
        Init_lr_fit = max(batch_size / nbs * opt.Init_lr, 1e-4)
        Min_lr_fit  = max(batch_size / nbs * opt.Min_lr, 1e-6) 
        # Optimizer
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (opt.momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = opt.momentum, nesterov=True)
        }[opt.optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": opt.weight_decay})
        optimizer.add_param_group({"params": pg2})
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        # scheduler
        lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, Init_lr_fit, Min_lr_fit, opt.UnFreeze_Epoch)
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        # if epoch_step == 0 or epoch_step_val == 0:
        #     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # EMA
        if ema:
            ema.updates = epoch_step * opt.Init_Epoch
        
        # 加载数据 

        train_dataset   = YoloDataset(train_lines, clear_lines, opt.input_shape, num_classes, train = True)
        val_dataset     = YoloDataset(val_lines, val_clear_lines, opt.input_shape, num_classes, train = False)

        if opt.distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen           = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=RANK, seed=opt.seed))
        gen_val       = DataLoader(val_dataset, shuffle = False, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate,sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=RANK, seed=opt.seed))


    # opt.UnFreeze_flag = False
    # epoch_29 = False
    # epoch_39 = False
    # epoch_139 = False
    # epoch_239 = False


    for epoch in range(opt.Init_Epoch, opt.UnFreeze_Epoch):
        # gc.collect()  # 清理内存

        # import pdb
        # pdb.set_trace()
        if epoch >= opt.Freeze_Epoch and not opt.UnFreeze_flag and opt.Freeze_Train:
            batch_size = opt.Unfreeze_batch_size  if opt.Freeze_Train else opt.Unfreeze_batch_size
            if opt.distributed:
                batch_size = batch_size * ngpus_per_node

            for name, param in model1.named_parameters():
                if 'decoders' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # Freeze EncooderRes
            for param in EncooderRes.parameters():
                param.requires_grad = False

            # Free not is 'head'
            for name, param in model3.named_parameters():
                if 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False 
                    
            # Freeze
            for param in EncoderFocus.parameters():
                param.requires_grad = False 

            # Freeze
            for param in EncoderFocus_next.parameters():
                param.requires_grad = False
                        
            nbs  = 64
            Init_lr_fit = max(batch_size / nbs * opt.Init_lr, 1e-4)
            Min_lr_fit  = max(batch_size / nbs * opt.Min_lr, 1e-6) 
            # Optimizer
            pg0, pg1, pg2 = [], [], []  
            for k, v in model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)    
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)    
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  
            optimizer = {
                'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (opt.momentum, 0.999)),
                'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = opt.momentum, nesterov=True)
            }[opt.optimizer_type]
            optimizer.add_param_group({"params": pg1, "weight_decay": opt.weight_decay})
            optimizer.add_param_group({"params": pg2})
            #---------------------------------------#
            #   获得学习率下降的公式
            #---------------------------------------#
            # scheduler
            lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, Init_lr_fit, Min_lr_fit, opt.UnFreeze_Epoch)
            #---------------------------------------#
            #   判断每一个世代的长度
            #---------------------------------------#
            epoch_step      = num_train // batch_size
            epoch_step_val  = num_val // batch_size
            # if epoch_step == 0 or epoch_step_val == 0:
            #     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            # EMA
            if ema:
                ema.updates = epoch_step * opt.Init_Epoch
            
            # 加载数据 

            train_dataset   = YoloDataset(train_lines, clear_lines, opt.input_shape, num_classes, train = True)
            val_dataset     = YoloDataset(val_lines, val_clear_lines, opt.input_shape, num_classes, train = False)

            if opt.distributed:
                train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
                val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
                batch_size      = batch_size // ngpus_per_node
                shuffle         = False
            else:
                train_sampler   = None
                val_sampler     = None
                shuffle         = True

            gen           = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                        worker_init_fn=partial(worker_init_fn, rank=RANK, seed=opt.seed))
            gen_val       = DataLoader(val_dataset, shuffle = False, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                        drop_last=True, collate_fn=yolo_dataset_collate,sampler=val_sampler, 
                                        worker_init_fn=partial(worker_init_fn, rank=RANK, seed=opt.seed))
            opt.UnFreeze_flag = True


        gen.dataset.epoch_now       = epoch
        gen_val.dataset.epoch_now   = epoch 
        if opt.distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        loss = 0  

        loss_AOD_value = 0
        loss_feature_value = 0
        losss_detection_value = 0


        val_loss = 0  

        val_loss_AOD_value = 0
        val_loss_feature_value = 0
        val_losss_detection_value = 0


        if RANK in {-1, 0}:
            print('Start Train...')
            pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{opt.UnFreeze_Epoch}',postfix=dict,mininterval=0.3)

        model_train.train()
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            # batch = batch.cuda(opt.local_rank)
            # images, targets, clearimgs = batch[0], batch[1], batch[2]
            images, bboxes, foggy_images = batch[0], batch[1], batch[2]
            with torch.no_grad():
                images  = torch.from_numpy(images).type(torch.FloatTensor).cuda(opt.local_rank)

                # bboxes = [torch.from_numpy(ann.numpy()).type(torch.FloatTensor).cpu() for ann in bboxes]
                bboxes = [torch.from_numpy(ann).type(torch.FloatTensor).cuda(opt.local_rank) for ann in bboxes]
                foggy_images = torch.from_numpy(foggy_images).type(torch.FloatTensor).cuda(opt.local_rank)
                # # 转换数据类型
                # foggy_images_np = foggy_images.astype(np.float32)
                # # NumPy的表示转换为PyTorch的表示
                # foggy_images_tensor = torch.from_numpy(foggy_images_np).cuda(opt.local_rank)
            optimizer.zero_grad()
            # AODfoggy_images_tensor_permuted:   AOD网络输出的图
            #                 fpn_outs_iamges:   原图 detecion特征list
            #           fpn_outs_foggy_images:   filter图 detecion特征list
            #      outputs_foggy_iamges_label:   预测的标签
            images_tensor_permuted, dexximg, foggy_images_feature_list, fpn_outs_clean_images_list, outputs_foggy_iamges_label  = model_train(opt, images, foggy_images, Type = 'train')  
            # print(filtered_img.shape) # torch.Size([4, 3, 640, 640])
            # loss_AOD_value = loss_AOD(dexximg, images_tensor_permuted)


            loss_feature_value = loss_feature(foggy_images_feature_list['dark3'], fpn_outs_clean_images_list['dark3'])
            

            losss_detection_value = loss_detection(outputs_foggy_iamges_label, bboxes) #  detection loss: box class

            # loss = loss_AOD_value + loss_feature_value +  losss_detection_value
            loss = 0.2 * loss_feature_value +  0.8 * losss_detection_value
            loss.backward()

            optimizer.step()

            loss += loss
            # loss_AOD_value += loss_AOD_value
            loss_feature_value += loss_feature_value
            losss_detection_value += losss_detection_value 


            if ema:
                ema.update(model_train)
              

            if RANK in {-1, 0}:
                # pbar.set_postfix(**{'lr' : get_lr(optimizer),'loss'  : loss / (iteration + 1),'AOD_loss': loss_AOD_value / (iteration + 1), 'Feature_loss': loss_feature_value / (iteration + 1),  'detection_loss': losss_detection_value / (iteration + 1)})
                pbar.set_postfix(**{'lr' : get_lr(optimizer),'loss'  : loss / (iteration + 1), 'Feature_loss': loss_feature_value / (iteration + 1),  'detection_loss': losss_detection_value / (iteration + 1)})
                pbar.update(1)

               
                writer.add_scalar('lr', get_lr(optimizer), epoch)


                writer.add_scalar('loss', loss / (iteration + 1), epoch)


                writer.add_scalar('Feature_loss', loss_feature_value / (iteration + 1), epoch)


                writer.add_scalar('detection_loss', losss_detection_value / (iteration + 1), epoch)


        if RANK in {-1, 0}:
            pbar.close()
            print('Finish Train')
        
        # if ema:
        #     model_train_eval = ema.ema
        # else:
        #     model_train_eval = model_train.eval()
        if (epoch + 1) % opt.eval_period == 0 or epoch + 1 == opt.UnFreeze_Epoch:
            print('Start Validation')
            model_train.eval()
            if RANK in {-1, 0}:
                pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{opt.UnFreeze_Epoch}',postfix=dict,mininterval=0.3)
            for iteration, batch in enumerate(gen_val):
                if iteration >= epoch_step_val:
                    break
                images, bboxes, foggy_images = batch[0], batch[1], batch[2]
                with torch.no_grad():
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda(opt.local_rank)

                    # bboxes = [torch.from_numpy(ann.numpy()).type(torch.FloatTensor).cpu() for ann in bboxes]
                    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor).cuda(opt.local_rank) for ann in bboxes]
                    foggy_images = torch.from_numpy(foggy_images).type(torch.FloatTensor).cuda(opt.local_rank)
                    # # 转换数据类型
                    # foggy_images_np = foggy_images.astype(np.float32)
                    # # NumPy的表示转换为PyTorch的表示
                    # foggy_images_tensor = torch.from_numpy(foggy_images_np).cuda(opt.local_rank)
                    optimizer.zero_grad()
                    # AODfoggy_images_tensor_permuted:   AOD网络输出的图
                    #                 fpn_outs_iamges:   原图 detecion特征list
                    #           fpn_outs_foggy_images:   filter图 detecion特征list
                    #      outputs_foggy_iamges_label:   预测的标签
        
                    images_tensor_permuted, dexximg, foggy_images_feature_list, fpn_outs_clean_images_list, outputs_foggy_iamges_label  = model_train(opt, images, foggy_images, Type = 'val')  
                    # print(filtered_img.shape) # torch.Size([4, 3, 640, 640])

                    # val_loss_AOD_value = loss_AOD(dexximg, images_tensor_permuted)
                    val_loss_feature_value = loss_feature(foggy_images_feature_list['dark3'], fpn_outs_clean_images_list['dark3'])
                    val_losss_detection_value = loss_detection(outputs_foggy_iamges_label, bboxes) #  detection loss: box class
                    val_loss = val_loss_AOD_value + val_loss_feature_value + val_losss_detection_value

                    val_loss += val_loss
                    # val_loss_AOD_value += val_loss_AOD_value
                    val_loss_feature_value += val_loss_feature_value
                    val_losss_detection_value += val_losss_detection_value 

  

                    # 绘图
                    if RANK == 0 and iteration== 2:

                        ori_image2 = images # torch.Size([B, 640, 640, 3])
                        ori_image3 = ori_image2.to("cpu").detach().numpy()# (b, 640, 640, 3])
                        ori_image4 = np.clip(ori_image3*255, 0, 255)# (b, 640, 640, 3])
                        #-------------保存图片----------#
                        cv2.imwrite("img/{}--1ori_image.png".format(epoch), ori_image4[0])
                    
                        foggy_images2 = foggy_images # torch.Size([2, 640, 640, 3])
                        foggy_images3 = foggy_images2.to("cpu").detach().numpy()# (2, 640, 640, 3)
                        foggy_images4 = np.clip(foggy_images3*255, 0, 255)# (b, 1330, 13303, 3)
                        #-------------保存图片----------#
                        cv2.imwrite("img/{}--2foggy_image.png".format(epoch), foggy_images4[0])
                    
                        dexximg2 = dexximg.permute(0,2,3,1) # torch.Size([2, 3, 640, 640]) --- torch.Size([B, 640, 640, 3])
                        dexximg3 = dexximg2.to("cpu").detach().numpy()# (2, 3, 640, 640)
                        dexximg4 = np.clip(dexximg3*255, 0, 255)# (b, 1330, 13303, 3)
                        #-------------保存图片----------#
                        cv2.imwrite("img/{}--3dexximg.png".format(epoch), dexximg4[0])




                    # val_loss += val_loss
                    if RANK in {-1, 0}:

                        # pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),'AOD_loss': val_loss_AOD_value / (iteration + 1), 'Feature_loss': loss_feature_value / (iteration + 1),  'detection_loss': losss_detection_value / (iteration + 1)})
                        pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 'Feature_loss': loss_feature_value / (iteration + 1),  'detection_loss': losss_detection_value / (iteration + 1)})
                        # wandb.log({
                        #     'val_loss'  : val_loss / (iteration + 1),'AOD_loss': val_loss_AOD_value / (iteration + 1), 'Feature_loss': loss_feature_value / (iteration + 1),  'detection_loss': losss_detection_value / (iteration + 1)
                        # })
                        pbar.update(1)

                        writer.add_scalar('val_loss', val_loss / (iteration + 1), epoch)

                        # print('val_loss')

                        # print('val_loss_AOD')

                        writer.add_scalar('val_loss_feature', loss_feature_value / (iteration + 1), epoch)  

                        # print('val_loss_feature')

                        writer.add_scalar('val_losss_detection', losss_detection_value / (iteration + 1), epoch) 

                        # print('val_losss_detection')

            if RANK in {-1, 0}:
                pbar.close()
                # 关闭writer
                writer.close()
                print('Finish Validation') 

            if dist.get_rank() == 0:
                losses_cpu = loss.detach().cpu()
                loss = losses_cpu.numpy()


                # loss_AOD_value_cpu = loss_AOD_value.detach().cpu()
                # loss_AOD_value = loss_AOD_value_cpu.numpy()

                # loss_feature_value_cpu = loss_feature_value.detach().cpu()
                # loss_feature_value = loss_feature_value_cpu.numpy()

                # losss_detection_value_cpu = losss_detection_value .detach().cpu()
                # losss_detection_value = losss_detection_value_cpu.numpy()



                val_losses_cpu = val_loss.detach().cpu()
                val_loss = val_losses_cpu.numpy()



                # val_loss_AOD_value_cpu = val_loss_AOD_value.detach().cpu()
                # val_loss_AOD_value = val_loss_AOD_value_cpu.numpy()


                # val_loss_feature_value_cpu = val_loss_feature_value.detach().cpu()
                # val_loss_feature_value  = val_loss_feature_value_cpu.numpy()


                # val_losss_detection_value_cpu = val_losss_detection_value.detach().cpu()  
                # val_losss_detection_value = val_losss_detection_value_cpu.numpy()

                # loss_history.append_loss(epoch + 1, psnr_train / epoch_step, loss / epoch_step, Dehazy_loss / epoch_step)          

                loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step)           
                print('Epoch:'+ str(epoch + 1) + '/' + str(opt.UnFreeze_Epoch))
                print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
                #-----------------------------------------------#
                #   保存权值
                #-----------------------------------------------#
                if ema:
                    save_state_dict = ema.ema.state_dict()
                else:
                    save_state_dict = model.state_dict()
                if (epoch + 1) % opt.save_period == 0 or epoch + 1 == opt.UnFreeze_Epoch:

                    # torch.save(model.state_dict(), 'logs/ep%03d-panr_train%.3f-loss%.3f-dehazy_loss%.3f.pth' % (epoch + 1, psnr_train / epoch_step, loss / epoch_step, Dehazy_loss / epoch_step))   
                    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)) 
                if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
                    print('Save best model to best_epoch_weights.pth')
                    torch.save(save_state_dict, os.path.join(opt.save_dir, "best_epoch_weights.pth"))
                    # wandb.save("best_epoch_weights.pth")
                    
                torch.save(save_state_dict, os.path.join(opt.save_dir, "last_epoch_weights.pth"))
        if opt.distributed:
            dist.barrier()
    if RANK == 0:
        loss_history.writer.close()
        
 




if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


    
# python -m torch.distributed.launch --nproc_per_node=2 train_multi_gpu.py