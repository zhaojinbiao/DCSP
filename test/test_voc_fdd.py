import os
import random
import cv2
import torch
import torch.distributed as dist
import argparse
from pathlib import Path
import time
import torch.nn as nn
import datetime
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from functools import partial
import torch.nn.functional as F
import xml.etree.ElementTree as ET


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
from utils.utils import get_lr, LossHistory, detect_image
from utils.psnr_ssim import batch_PSNR
from ptflops import get_model_complexity_info
from utils.utils_bbox import decode_outputs, non_max_suppression, draw_boxes
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import colorsys
from utils.utils import *
from utils.utils_map import get_map
from torchvision.utils import save_image


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
print('WORLD_SIZE:{}'.format(WORLD_SIZE))
print('LOCAL_RANK:{}'.format(LOCAL_RANK))
print('RANK:{}'.format(RANK))

def parse_opt(known=False):
    parser = argparse.ArgumentParser() 
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=3407, help='Global training seed')
    parser.add_argument('--classes_path', type=str, default='datasets/classes/voc_fdd_classes.txt', help='分类标签路径')
    parser.add_argument('--model_path', type=str, default='train_weitht/last_epoch_weights.pth', help='加载模型路径') # 'pth/yolox_s.pth'
    # parser.add_argument('--model_path1', type=str, default='pth/dehazer.pth', help='加载模型路径') # 



    parser.add_argument('--val_fog_annotation_path', type=str, default='datasets/FDDtest/2007_train.txt', help='雾图验证集标注')
    parser.add_argument('--val_clean_annotation_path', type=str, default='datasets/FDDtest/2007_train.txt', help='干净图验证集标注')

    parser.add_argument('--vocfog_traindata_dir', dest='vocfog_traindata_dir', default='datasets/FDDtest/VOC2007/JPEGImages/',help='train the dir contains ten levels synthetic foggy images')
    parser.add_argument('--vocfog_valdata_dir', dest='vocfog_valdata_dir', default='datasets/FDDtest/VOC2007/JPEGImages/',help='雾图路径')
    # 验证的--val_annotation_path --val_clear_annotation_path 因为我没有使用验证集，所以没有这两个路径
    # parser.add_argument('--clear_annotation_path', type=str, default='2007_train.txt', help='2007_train.txt')
    # parser.add_argument('--val_clear_annotation_path', type=str, default='2007_val.txt', help='2007_val.txt')
    parser.add_argument('--input_shape', type=list, default=[640,640], help='输入图片resize大小')
    parser.add_argument('--mosaic', type=bool, default=False, help='是否mosaic')
    parser.add_argument('--batch_size', type=int, default= '1', help='根据是否freeze动态调整batch_size')
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
    parser.add_argument('--distributed', type=bool, default=False, help='')
    parser.add_argument('--sync_bn', type=bool, default=False,help='')
    parser.add_argument('--eval_flag',type=bool,default=True,help='')
    parser.add_argument('--eval_period',type=int,default=1,help='')
    parser.add_argument('--phi',type=str,default='s',help='指定model类型S-X')
    parser.add_argument('--font_path', type=str, default='datasets/simhei.ttf', help='字体文件')

    parser.add_argument('--psnr', type=bool, default=False, help='')
    parser.add_argument('--detection', type=bool, default=False, help='')
    #-------------------------------------------------------------------------------------------------------------------------gai
    parser.add_argument('--test_out_path', type=str, default='datasets/FDDtest/test_out', help='推理map的路径')

    parser.add_argument('--MINOVERLAP', type=list, default='[0.5]', help='平均预测精度')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):

    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_git_status()
        # check_requirements()
    # defin model
    class_names, num_classes = get_classes(opt.classes_path)
#-------------------------------------------------------------------defin model-----------------------------------------------------------------------------
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
    
#--------------------------------------------------------------------DDP model----------------------------------------------------------------------------------
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        ngpus_per_node  = torch.cuda.device_count()
        # dist.init_process_group(backend='nccl', world_size = WORLD_SIZE, rank=LOCAL_RANK)
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

        model_test = model.cuda(LOCAL_RANK)
        model_test = torch.nn.parallel.DistributedDataParallel(model_test, device_ids=[LOCAL_RANK], find_unused_parameters=True)
    # model = torch.nn.parallel.DistributedDataParallel(model.cuda(LOCAL_RANK), device_ids=[LOCAL_RANK])
    # model_dict = model_test.state_dict()

    # pretrained_dict1 = torch.load('pth/last_epoch_weights.pth', map_location = device)
    model_test = model.cuda()
    model_dict1 = model_test.state_dict()
    pretrained_dict1 = torch.load(opt.model_path)

#--------------------------------------------------------------------load weights----------------------------------------------------------------------------------
    # for name, param in pretrained_dict1.items():
    load_key1, no_load_key1, temp_dict1 = [], [], {}
    for k, v in pretrained_dict1.items():
        if k in model_dict1.keys() and np.shape(model_dict1[k]) == np.shape(v):
            temp_dict1[k] = v
            load_key1.append(k)
        else:
            no_load_key1.append(k)
    print('load_key1:{}'.format(len(load_key1)))
    print('no_load_key1:{}'.format(len(no_load_key1)))
    print('-'*50)
    model_dict1.update(temp_dict1)
    model_test.load_state_dict(model_dict1)
    print('-'*10)

#--------------------------------------------------------------------load data----------------------------------------------------------------------------------
    with open(opt.val_fog_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()

    with open(opt.val_clean_annotation_path, encoding='utf-8') as f:
        val_clear_lines = f.readlines()

    num_val     = len(val_lines)

    test_dataset     = YoloDataset(val_lines, val_clear_lines, opt.input_shape, num_classes, train = False)

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if opt.distributed:
        test_sampler   = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=True,)
        batch_size     = batch_size // ngpus_per_node
        shuffle        = False
    else:
        test_sampler   = None
        shuffle        = True

    gen_test       = DataLoader(test_dataset, shuffle = False, batch_size = 1, num_workers = opt.num_workers, pin_memory=True, 
                                drop_last=True, collate_fn=yolo_dataset_collate,sampler=test_sampler, 
                                worker_init_fn=partial(worker_init_fn, rank=RANK, seed=opt.seed))
     
    # 手动设置
    clean_dir(os.path.join(opt.test_out_path, 'ground-truth'))
    clean_dir(os.path.join(opt.test_out_path, 'detection-results'))
    clean_dir(os.path.join(opt.test_out_path, 'images-optional'))
    clean_dir(os.path.join(opt.test_out_path, 'psnr_results'))
    clean_dir(os.path.join(opt.test_out_path, 'dehazy_out_results'))


#--------------------------------------------------------------------tiao zheng model mood----------------------------------------------------------------------------------
    model_train     = model.train()
    model_train.eval()

    psnr_meter = AverageMeter()
    time_meter = AverageMeter()

    num_data = test_dataset.__len__()

    for iteration, batch in enumerate(gen_test):
            if iteration >= num_data:
                break
            # images 雾图 == foggy_images  names,图片的id 有真实干净的图才有psnr，只有fog-test才有psnr
            # import pdb
            # pdb.set_trace()
            with torch.no_grad():
                images, bboxes, foggy_images, names = batch[0], batch[1], batch[2], batch[3]
                image       = Image.open(names[0])
                image_shape = np.array(np.shape(image)[0:2])
                start_time = time.time()
                images_tensor = torch.from_numpy(images).type(torch.FloatTensor).cuda()

                bboxes = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in bboxes]
                foggy_images = torch.from_numpy(foggy_images).type(torch.FloatTensor).cuda()
                images_tensor_permuted, dexximg, foggy_images_feature_list, outputs_foggy_iamges_label  = model_train(opt, images_tensor, foggy_images, Type = 'test')  

                times = time.time() - start_time
                image_id = names[0].split('/')[-1].split('.')[0]



            if opt.psnr:
                f = open(os.path.join(opt.test_out_path, "psnr_results/", "psnr.txt"),"a")
                # pred_clip = torch.clamp(dexximg, 0, 1)
                # pred_clip = dexximg.permute(0,2,3,1) # [1, 3, 640, 640]
                pred_clip = dexximg
                cur_psnr = get_metrics(pred_clip, images_tensor)
                psnr_meter.update(cur_psnr, 1)
                time_meter.update(times, 1)

                print('Iteration[' + str(iteration+1) + '/' + str(len(test_dataset)) + ']' + '  Processing image... ' + names[0] + '   PSNR: ' + str(cur_psnr) + '  Time ' + str(times))
                f.write('Iteration[' + str(iteration+1) + '/' + str(len(test_dataset)) + ']' + '  Processing image... ' + names[0] + '   PSNR: ' + str(cur_psnr) + ' Time: ' + str(times) + '\n')
                f.close()
            if opt.detection:
                    d = open(os.path.join(opt.test_out_path, "detection-results/" + image_id +".txt"),"w")
                    # 解码
                    outputs = outputs_foggy_iamges_label
                    outputs = decode_outputs(outputs, opt.input_shape)
                    results = non_max_suppression(outputs, num_classes, opt.input_shape, image_shape, letterbox_image = True, conf_thres = 0.0001, nms_thres = 0.555)
                    
                    if results[0] is None: 
                        return image

                    top_label   = np.array(results[0][:, 6], dtype = 'int32')
                    top_conf    = results[0][:, 4] * results[0][:, 5]
                    top_boxes   = results[0][:, :4]
                    # font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                    # thickness   = int(max((image.size[0] + image.size[1]) // np.mean(opt.input_shape), 1))
                    for i, c in list(enumerate(top_label)):
                        predicted_class = class_names[int(c)]
                        box             = top_boxes[i]
                        score           = str(top_conf[i])

                        top, left, bottom, right = box
                        if predicted_class not in class_names:
                            continue
                        d.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
                    d.close()
                    print("Get predict result done.")
                
                    #  获取GT
                    d_gt = open(os.path.join(opt.test_out_path, "ground-truth/"+image_id+".txt"), "w")
                    # -------------------------------------------------------------------------------------------------------------------gai
                    root = ET.parse(os.path.join("datasets/FDDtest/VOC2007", "Annotations/"+image_id+".xml")).getroot()
                    for obj in root.findall('object'):
                        difficult_flag = False
                        if obj.find('difficult')!=None:
                            difficult = obj.find('difficult').text
                            if int(difficult)==1:
                                difficult_flag = True
                        obj_name = obj.find('name').text
                        if obj_name not in class_names:
                            continue
                        bndbox  = obj.find('bndbox')
                        left    = bndbox.find('xmin').text
                        top     = bndbox.find('ymin').text
                        right   = bndbox.find('xmax').text
                        bottom  = bndbox.find('ymax').text

                        if difficult_flag:
                            d_gt.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                        else:
                            d_gt.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                    print("Get ground truth result done.")
                   
            if opt.psnr:
                
                ori_image2 = images_tensor # torch.Size([B, 640, 640, 3])
                ori_image3 = ori_image2.to("cpu").detach().numpy()# (b, 640, 640, 3])
                ori_image4 = np.clip(ori_image3*255, 0, 255)# (b, 640, 640, 3])
                #-------------保存图片----------#
                cv2.imwrite(opt.test_out_path + '/dehazy_out_results' + '/' + image_id + '_gt.jpg', ori_image4[0])
            
                foggy_images2 = foggy_images # torch.Size([2, 640, 640, 3])
                foggy_images3 = foggy_images2.to("cpu").detach().numpy()# (2, 640, 640, 3)
                foggy_images4 = np.clip(foggy_images3*255, 0, 255)# (b, 1330, 13303, 3)
                #-------------保存图片----------#
                cv2.imwrite(opt.test_out_path + '/dehazy_out_results' + '/' + image_id + '_img.jpg', foggy_images4[0])
            
                dexximg2 = dexximg.permute(0,2,3,1) # torch.Size([2, 3, 640, 640]) --- torch.Size([B, 640, 640, 3])
                dexximg3 = dexximg2.to("cpu").detach().numpy()# (2, 3, 640, 640)
                dexximg4 = np.clip(dexximg3*255, 0, 255)# (b, 1330, 13303, 3)
                #-------------保存图片----------#
                cv2.imwrite(opt.test_out_path + '/dehazy_out_results' + '/' + image_id + '_restored.jpg', dexximg4[0])


    if opt.psnr:
        f = open(os.path.join(opt.test_out_path, "psnr_results/", "psnr.txt"),"a")
        print('Average: PSNR: {:.4f} Time: {:.4f}'.format(psnr_meter.average(), time_meter.average()))
        f.write('Average: PSNR: ' + str(psnr_meter.average()) + ' Time: ' + str(time_meter.average()) + '\n')
        f.close()
    print('Successfully save results to: ' + "test_out/psnr_results/psnr.txt")
    MINOVERLAP = [0.5]
    if opt.detection:
        # 计算map
        for i in MINOVERLAP:
            get_map(i, draw_plot = True, path = opt.test_out_path)
        print("Get map done.")



        #         # 绘图
        #         if RANK == 0 and iteration== 2:

        #             ori_image2 = images # torch.Size([B, 640, 640, 3])
        #             ori_image3 = ori_image2.to("cpu").detach().numpy()# (b, 640, 640, 3])
        #             ori_image4 = np.clip(ori_image3*255, 0, 255)# (b, 640, 640, 3])
        #             #-------------保存图片----------#
        #             cv2.imwrite("img/{}--1ori_image.png".format(epoch), ori_image4[0])
                
        #             foggy_images2 = foggy_images # torch.Size([2, 640, 640, 3])
        #             foggy_images3 = foggy_images2.to("cpu").detach().numpy()# (2, 640, 640, 3)
        #             foggy_images4 = np.clip(foggy_images3*255, 0, 255)# (b, 1330, 13303, 3)
        #             #-------------保存图片----------#
        #             cv2.imwrite("img/{}--2foggy_image.png".format(epoch), foggy_images4[0])
                
        #             dexximg2 = dexximg.permute(0,2,3,1) # torch.Size([2, 3, 640, 640]) --- torch.Size([B, 640, 640, 3])
        #             dexximg3 = dexximg2.to("cpu").detach().numpy()# (2, 3, 640, 640)
        #             dexximg4 = np.clip(dexximg3*255, 0, 255)# (b, 1330, 13303, 3)
        #             #-------------保存图片----------#
        #             cv2.imwrite("img/{}--3dexximg.png".format(epoch), dexximg4[0])



        #         # map




        #         # val_loss += val_loss
        #         if RANK in {-1, 0}:
        #             # pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
        #             pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),'AOD_loss': val_loss_AOD_value / (iteration + 1), 'Feature_loss': loss_feature_value / (iteration + 1),  'detection_loss': losss_detection_value / (iteration + 1)})

        # if RANK in {-1, 0}:
        #     pbar.close()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    dist.destroy_process_group()