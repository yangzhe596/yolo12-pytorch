import os
import time

import torch
from tqdm import tqdm

from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0, time_analyse=False):
    loss        = 0
    val_loss    = 0

    # 耗时统计变量
    if time_analyse:
        time_data_total = 0.0
        time_forward_total = 0.0
        time_backward_total = 0.0
        time_batch_start = time.time()

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # 数据加载耗时统计
        if time_analyse:
            time_data = time.time() - time_batch_start

        images, bboxes = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            if time_analyse:
                torch.cuda.synchronize() if cuda else None
                time_forward_start = time.time()
            # dbox, cls, origin_cls, anchors, strides
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, bboxes)
            if time_analyse:
                torch.cuda.synchronize() if cuda else None
                time_forward = time.time() - time_forward_start
                time_backward_start = time.time()
            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
            if time_analyse:
                torch.cuda.synchronize() if cuda else None
                time_backward = time.time() - time_backward_start
        else:
            from torch.cuda.amp import autocast
            if time_analyse:
                torch.cuda.synchronize() if cuda else None
                time_forward_start = time.time()
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)
                loss_value = yolo_loss(outputs, bboxes)

            if time_analyse:
                torch.cuda.synchronize() if cuda else None
                time_forward = time.time() - time_forward_start
                time_backward_start = time.time()
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            if time_analyse:
                torch.cuda.synchronize() if cuda else None
                time_backward = time.time() - time_backward_start
        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        # 耗时统计累加与打印
        if time_analyse:
            time_data_total += time_data
            time_forward_total += time_forward
            time_backward_total += time_backward

            # 每10个batch打印一次平均值
            if (iteration + 1) % 50 == 0 and local_rank == 0:
                avg_data = time_data_total / (iteration + 1) * 1000
                avg_forward = time_forward_total / (iteration + 1) * 1000
                avg_backward = time_backward_total / (iteration + 1) * 1000
                pbar.write(f'[Time Analyse] iter {iteration + 1}: '
                           f'data={avg_data:.1f}ms, forward={avg_forward:.1f}ms, backward={avg_backward:.1f}ms')

            # 记录下一个batch开始时间
            time_batch_start = time.time()

        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train_eval(images)
            loss_value  = yolo_loss(outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
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