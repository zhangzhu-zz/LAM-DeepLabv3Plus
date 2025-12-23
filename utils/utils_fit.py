import os
import numpy as np

import torch
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score, fast_hist, per_Accuracy


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    total_hist      = np.zeros((num_classes, num_classes))  # 训练混淆矩阵

    val_loss        = 0
    val_f_score     = 0
    val_hist        = np.zeros((num_classes, num_classes))  # 验证混淆矩阵

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   计算损失
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)
                # 将输出转换为预测类别
                preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()  # 预测类别
                labels_flat = pngs.cpu().numpy().flatten()  # 真实标签
                total_hist += fast_hist(labels_flat, preds, num_classes)  # 累加混淆矩阵

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   计算损失
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)
                    #   计算混淆矩阵
                    preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()  # 预测类别
                    labels_flat = pngs.cpu().numpy().flatten()  # 真实标签
                    total_hist += fast_hist(labels_flat, preds, num_classes)  # 累加混淆矩阵 
                    
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
            
        if local_rank == 0:
            total_accuracy = per_Accuracy(total_hist) * 100  # 计算训练准确率
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'accuracy'  : total_accuracy,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(imgs)
            #----------------------#
            #   计算损失
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            labels_flat = pngs.cpu().numpy().flatten()
            val_hist += fast_hist(labels_flat, preds, num_classes)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
            if local_rank == 0:
                val_accuracy = per_Accuracy(val_hist) * 100  # 计算验证准确率
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'accuracy'  : val_accuracy,
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        total_accuracy = per_Accuracy(total_hist) * 100
        val_accuracy = per_Accuracy(val_hist) * 100
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, total_accuracy, val_accuracy)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f || Train Accuracy: %.2f%% || Val Accuracy: %.2f%%' % (total_loss / epoch_step, val_loss / epoch_step_val, total_accuracy, val_accuracy))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f-acc%.2f-val_acc%.2f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val, total_accuracy, val_accuracy)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))