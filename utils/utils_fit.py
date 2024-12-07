import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, fp16, scaler, save_period, save_dir, local_rank=0):
    
    loss        = 0  
    val_loss    = 0
    Dehazy_loss = 0
    model_train.train()
    print('Start Train...')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar: 
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets, clearimgs = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor).cuda()

            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()

            if not fp16:
                # loss_value = 0.2 * loss_value + 0.8 * loss_dehazy
                # loss_value.backward()
                # loss_dehazy.backward()
                # optimizer.step()            
                
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)

                #----------------------#
                #   计算损失
                #----------------------#
                loss_dehazy = yolo_loss(outputs, clearimgs)

                #----------------------#
                #   反向传播
                #----------------------#
                loss_dehazy.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model_train(images)
                    #----------------------#
                    #   计算损失
                    #----------------------#
                    loss_value = yolo_loss(outputs, targets)

                #----------------------#
                #   反向传播
                #----------------------#
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            if ema:
                ema.update(model_train)

            loss += loss_dehazy.item()
            Dehazy_loss += loss_dehazy.item()

            # 添加进度条后面要显示的其他信息
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                                'Dehazy_loss': Dehazy_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')
    print('Start Validation')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets, clearimgs = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor).cuda()
                #----------------------#
                #   清零梯度
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = yolo_loss(outputs, clearimgs)
                # for l in range(len(outputs)-1):
                #     loss_item = yolo_loss(outputs[l], targets)
                #     loss_value_all  += loss_item
                # loss_value = loss_value_all

            val_loss += loss_value.item()
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        # eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))


    # loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    # print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    # print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
    #     torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))     
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))