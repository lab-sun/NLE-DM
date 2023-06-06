"""
The training and validation modules for network
"""

import torch
import torch.nn.functional
import train_utils.distributed_utils as utils
import numpy as np
from sklearn.metrics import f1_score


def criterion(input1, input2, target1, target2, device):
    losses = {}
    """
    for network to jointly predict action and reason, class_weights=[1, 1, 2, 2];
    for network to jointly predict action and description, class_weights=[1, 3, 2, 2]
    """
    class_weights1 = [1, 1, 2, 2]  # for act_rea
    # class_weights = [1, 3, 2, 2] # for act_des
    w1 = torch.FloatTensor(class_weights1).to(device)

    losses['action'] = torch.nn.functional.binary_cross_entropy_with_logits(input1, target1, weight=w1)
    losses['reason'] = torch.nn.functional.binary_cross_entropy_with_logits(input2, target2)

    return 0.5 * losses['action'] + 1.0 * losses['reason']


def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0

        Target_ActionArr = []
        Target_ReasonArr = []
        Pre_ActionArr = []
        Pre_ReasonArr = []

        Action_overall = []
        Reason_overall = []

        for image, target in data_loader:
            # calculate the validation loss
            image = image.to(device)
            target[0] = target[0].to(device)
            target[1] = target[1].to(device)
            output = model(image)
            output1 = output[0].to(device)
            output2 = output[1].to(device)
            loss = criterion(output1, output2, target[0], target[1], device)
            val_loss += loss  # val loss for one epoch

            # calculate the F1 score
            # Pretreat the predictions in order to calculate the F1 score (from Yiran Xu et al.)
            predict_action = torch.sigmoid(output1) > 0.5
            predict_reason = torch.sigmoid(output2) > 0.5

            # Overall F1 score (F1_all)
            # Action Part
            f1_overall_action = f1_score(target[0].cpu().numpy(), predict_action.cpu().numpy(), average='samples')
            Action_overall.append(f1_overall_action)
            # Reason Part
            f1_overall_reason = f1_score(target[1].cpu().data.numpy(), predict_reason.cpu().data.numpy(), average='samples')
            Reason_overall.append(f1_overall_reason)

            # category & average F1 score (F1 & mF1)
            Target_ActionArr.append(target[0].cpu().numpy())
            Pre_ActionArr.append(predict_action.cpu().numpy())
            Target_ReasonArr.append(target[1].cpu().numpy())
            Pre_ReasonArr.append(predict_reason.cpu().numpy())

        ActionArr = List2Arr(Target_ActionArr)
        Pre_ActionArr = List2Arr(Pre_ActionArr)
        ReasonArr = List2Arr(Target_ReasonArr)
        Pre_ReasonArr = List2Arr(Pre_ReasonArr)

        f1_action = f1_score(ActionArr, Pre_ActionArr, average=None)  # Action Category Acc
        f1_reason = f1_score(ReasonArr, Pre_ReasonArr, average=None)  # Reason Category Acc
        f1_action_average = np.mean(f1_action)  # Action Average Acc
        f1_reason_average = np.mean(f1_reason)  # Action Average Acc

    return val_loss.item(), np.mean(Action_overall), np.mean(Reason_overall), f1_action, f1_action_average, f1_reason, f1_reason_average

def evaluate_nu(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        threshold = 0.5
        targets_acts = []
        targets_desc = []
        output_acts = []
        output_desc = []

        act_category = [0.0] * 4
        desc_category = [0.0] * 6

        for image, target in data_loader:
            # calculate the validation loss
            image = image.to(device)
            target0 = target[0].to(device)
            target1 = target[1].to(device)

            output = model(image)
            output0 = output[0].to(device)
            output1 = output[1].to(device)
            loss = criterion(output0, output1, target0, target1, device)
            val_loss += loss  # val loss for one epoch

            """
            Calculate the F1 score
            """
            # Pretreat the predictions in order to calculate the F1 score (from Yiran Xu et al.)
            # Pretreat the prediction & target in order to calculate the F1 score
            predict_act = torch.sigmoid(output0) > threshold
            predict_desc = torch.sigmoid(output1) > threshold
            predict_act = predict_act.cpu().numpy()
            predict_desc = predict_desc.cpu().numpy()
            target0_numpy = target0.cpu().numpy()
            target1_numpy = target1.cpu().numpy()

            targets_acts.append(target0_numpy)
            output_acts.append(predict_act)
            targets_desc.append(target1_numpy)
            output_desc.append(predict_desc)

        targets_desc = List2List(targets_desc)
        output_desc = List2List(output_desc)
        targets_acts = List2List(targets_acts)
        output_acts = List2List(output_acts)


        # Select each action/description category and calculate their F1 scores
        for i in range(4):
            act_category[i] = f1_score(targets_acts[i::4], output_acts[i::4])

        for i in range(6):
            desc_category[i] = f1_score(targets_desc[i::6], output_desc[i::6])

        f1_overall_act = f1_score(targets_acts, output_acts, average='macro')
        f1_overall_desc = f1_score(targets_desc, output_desc, average='macro')

    return val_loss.item(), act_category, desc_category, \
           f1_overall_act, f1_overall_desc, np.mean(act_category), np.mean(desc_category)

def List2Arr(List):
    Arr1 = np.array(List[:-1]).reshape(-1, List[0].shape[1])
    Arr2 = np.array(List[-1]).reshape(-1, List[0].shape[1])

    return np.vstack((Arr1, Arr2))

def List2List(List):
    Arr1 = np.array(List[:-1]).reshape(-1, List[0].shape[1])
    Arr2 = np.array(List[-1]).reshape(-1, List[0].shape[1])
    Arr = np.vstack((Arr1, Arr2))

    return [i for item in Arr for i in item]

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        target[0] = target[0].to(device)
        target[1] = target[1].to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            output1 = output[0].to(device)
            output2 = output[1].to(device)
            loss = criterion(output1, output2, target[0], target[1], device)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].globals, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        return a learning rate factor according to the number of steps
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)

            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
