import numpy as np
import torch
from torchvision.ops import nms, boxes

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def decode_outputs(outputs, input_shape):
    grids   = []
    strides = []
    hw      = [x.shape[-2:] for x in outputs]
    #---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    #---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    #---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率
    #---------------------------------------------------#
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        #---------------------------#
        #   根据特征层的高宽生成网格点
        #---------------------------#   
        grid_y, grid_x  = torch.meshgrid([torch.arange(h), torch.arange(w)])
        #---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        #---------------------------#   
        grid            = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape           = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    #---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    #---------------------------#
    grids               = torch.cat(grids, dim=1).type(outputs.type())
    strides             = torch.cat(strides, dim=1).type(outputs.type())
    #------------------------#
    #   根据网格点进行解码
    #------------------------#
    outputs[..., :2]    = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4]   = torch.exp(outputs[..., 2:4]) * strides
    #-----------------#
    #   归一化
    #-----------------#
    outputs[..., [0,2]] = outputs[..., [0,2]] / input_shape[1]
    outputs[..., [1,3]] = outputs[..., [1,3]] / input_shape[0]
    return outputs


from PIL import  ImageFont, ImageDraw
def draw_boxes(image, outputs, font_file, class_names, colors_list=[]):
    """
    在图片上画框
    Args:
        image: 要画框的图片，PIL.Image.open的返回值
        outputs: 一个列表，NMS后的结果，其中的坐标为归一化后的坐标
        font_file:字体文件路径
        class_names:类名列表
        colors_list:颜色列表

    Returns:

    """
    # 根据图片的宽，动态调整字体大小
    font_size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
    # font_size = np.floor(3e-2 * 640 + 0.5).astype('int32')
    font = ImageFont.truetype(font=font_file, size=font_size)  # 创建字体对象，包括字体和字号
    draw = ImageDraw.Draw(image)                # 将letterbox_img作为画布

    for output in outputs:                      # ouput是每张图片的检测结果，当然这里batch_size为1就是了
        if output is not None:
            for obj in output:                  # 一张图片可能有多个目标，obj就是其中之一
                """从obj中获得信息"""
                box = obj[:4] * 640             # 将归一化后的坐标转化为输入图片（letterbox_img）中的坐标
                cls_index = int(obj[6])         # 类别索引
                score = obj[4] * obj[5]         # score，可以理解为类别置信度
                x1, y1, x2, y2 = map(int, box)          # 转化为整数
                pred_class = class_names[cls_index]     # 目标类别名称
                color = 'red'                           # TODO 具体使用时，还得改成colors_list[cls_index]

                """组建要显示的文字信息"""
                label = ' {} {:.2f}'.format(pred_class, score)
                # print(label, x1, y1, x2, y2)

                """获得文字的尺寸"""
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                """防止文字背景框在上边缘越界"""
                if y1 - label_size[1] >= 0:
                    text_origin = np.array([x1, y1 - label_size[1]])
                else:
                    # 如果越界，则将文字信息写在边框内部
                    text_origin = np.array([x1, y1 + 1])

                """绘制边框"""
                thickness = 2               # 边框厚度
                for i in range(thickness):  # 根据厚度确定循环的执行次数
                    draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=color)  # colors[cls_index]

                """绘制文字框"""
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)   # 背景
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)              # 文字
    del draw

    return image

def non_max_suppression_my(prediction, conf_thres=0.5, nms_thres=0.4):
    """
        Args: 
                prediction: 模型预测的结果（经解码后的数据）outputs[batch_size, 8400, 85]
                            如果要预测80个类别,那么prediction的维度为[bathc_size, num_anchors, 85]
                nms_thres:  NMS阈值
                conf_thres: 置信度阈值
                num_anchor = totle [w * h]
    """
    # 将预测结果的格式中心坐标和宽高转换成左上角右下角的坐标
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2
    box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2
    box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2]/2
    box_corner[:,:,3] = prediction[:,:,1] + prediction[:,:,3]/2
    prediction[:,:,:4] = box_corner[:,:,:4]
    #prediction (bath_size, totle[w * h], 4 + 1 + num_classes)
    
    output = [None for _ in range(len(prediction))]# len(prediction))是batch_size，即图片数量,也就是返回张量的第一个维度
    
    
    # 对图片进行循环 prediction [batch_size, num_anchors, 4 + 1 +num_classes]
    for image_i, image_pred in enumerate(prediction):
        """第一轮过滤"""
        # 利用置信度进行第一轮筛选        
        # image_pred[num_anchors, 4(左上角右下角) + 1（obj） + num_classes]
        # 利用目标置信度（即对应的预测框存在要检测的目标的概率）做第一轮过滤
        image_pred = image_pred[image_pred[:,4]>=conf_thres]
    
        # class_conf 类别概率最大值[num_anchors, 1] 
        # class_pred 预测类别在nam_class中的索引[num_anchors, 1]

        #如果当前图片中，所有目标的置信度都小于阈值,那么就进行下一轮循环，检测下一张图片
        if not image_pred.size(0):
            continue
        
        # 目标置信度乘以各个类别的概率，并对结果取最大值，获得各个预测框的score
        score = image_pred[:,4] * image_pred[:,5:].max(1)[0]
        # image_pred[:,4]是置信度， image_pred[:,5:].max(1)[0]是各个类别的概率最大值

        # 将image_pred中的预测框按score从大到小排序
        image_pred = image_pred[(-score).argsort()]
        # argsort()是将(-score)中的元素从小到大排序，返回排序后索引
        # 将(-score)中的元素从小到大排序，实际上是对score从大到小排序
        # 将排序后的索引放入image_pred中作为索引，实际上是对本张图片中预测出来的目标，按score从大到小排序

        # 获得第一轮过滤后的各个预测框的类别概率最大值及其索引
        class_confs, class_preds = image_pred[:,5:].max(1, keepdim=True)
        # class_confs 类别概率最大值，class_preds 预测类别在80个类别中的索引

        # 将各个目标框的上下角点坐标、目标置信度、类别置信度、类别索引串起来
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # 经过上条命令之后，
        # [num_anchors, x1, y1, x2, y2, obj_conf] + [num_anchors, class_conf] + [num_anchors, class_pred] 
        # detections的维度变为（num_anchors, 7）
        

        """第二轮过滤"""
        keep_boxes = []     # 用来存储符合要求的目标框

        while detections.size(0):   # 如果detections中还有目标
            """以下标注是执行第一轮循环时的标注，后面几轮以此类推"""
            # 获得与第一个box（最大score对应的box）具有高重叠的预测框的布尔索引
            from utilss import bbox_iou
            # bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4])返回值的维度为(num_objects, )
            # bbox_iou的返回值与非极大值抑制的阈值相比较，获得布尔索引
            # 即剩下的边框中，只有detection[0]的iou大于nms_thres的，才抑制，即认为这些边框与detection[0]检测的是同一个目标
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres

            # 获得与第一个box相同类别的预测框的索引
            label_match = detections[0, -1] == detections[:, -1]
            # 布尔索引，获得所有与detection[0]相同类别的对象的索引
            
            # 获得需要抑制的预测框的布尔索引
            invalid = large_overlap & label_match   # &是位运算符，两个布尔索引进行位运算
            # 经过第一轮筛选后的剩余预测框，如果同时满足和第一个box有高重叠、类别相同这两个条件，那么就该被抑制
            # 这些应该被抑制的边框，其对应的索引即为无效索引

            # 获得被抑制预测框的置信度
            weights = detections[invalid, 4:5]

            # 加权获得最后的预测框坐标
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            # 上面的命令是将当前边框，和被抑制的边框进行加权，
            # 类似于好几个边框都检测到了同一张人脸，将这几个边框的左上角点横坐标x进行加权（按照置信度加权），
            # 获得最后边框的x，对左上角点的纵坐标y，以及右下角点的横纵坐标也进行加权处理
            # 其他的obj_conf, class_conf, class_pred则使用当前box的

            keep_boxes += [detections[0]]       # 将第一个box加入到 keep_boxes 中
            detections = detections[~invalid]   # 去掉无效的预测框，更新detections

        if keep_boxes:                          # 如果keep_boxes不是空列表
            output[image_i] = torch.stack(keep_boxes)   # 将目标堆叠，然后加入到列表
            # 假设NMS之后，第i张图中有num_obj个目标，那么torch.stack(keep_boxes)的结果是就是一个(num_obj, 7)的张量，没有图片索引

        # 如果keep_boxes为空列表，那么output[image_i]则未被赋值，保留原来的值（原来的为None）
    return output

def non_max_suppression(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    #----------------------------------------------------------#
    box_corner          = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    
    output = [None for _ in range(len(prediction))]
    #----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        
        nms_out_index = boxes.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thres,
        )

        output[i]   = detections[nms_out_index]

        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]
            
        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data
            
        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
        
        if output[i] is not None:
            output[i]           = output[i].detach().cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output
